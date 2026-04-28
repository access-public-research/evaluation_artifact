import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils.io import ensure_dir
from ..utils.stats import ci95_mean, cvar_top_fraction


@dataclass
class RunMatch:
    dataset: str
    regime: str
    seed: int
    epoch: int
    tag: str
    run_dir: Path


def _infer_prefix(family: str, meta: dict) -> str:
    prefix = str(meta.get("prefix", "")).strip()
    if prefix:
        return prefix
    lname = family.lower()
    if "diff" in lname:
        return "diff"
    if "hash" in lname:
        return "hash"
    if "proj" in lname:
        return "proj"
    if "conf" in lname:
        return "conf"
    raise ValueError(f"Could not infer prefix for family={family}.")


def _load_partitions(
    eval_root: Path,
    family: str,
    bank: str,
    split: str,
    num_parts: int,
) -> List[np.ndarray]:
    fam_dir = eval_root / family
    meta = json.loads((fam_dir / "meta.json").read_text(encoding="utf-8"))
    prefix = _infer_prefix(family, meta)

    split_dir = split
    if split in {"val", "validation"}:
        split_dir = "validation"
    elif split == "val_skew":
        split_dir = "val_skew"
    elif split == "train":
        split_dir = "train"

    base = fam_dir / f"bank{bank}" / split_dir
    out: List[np.ndarray] = []
    for m in range(int(num_parts)):
        matches = list(base.glob(f"{prefix}_m{m:02d}_K*.npy"))
        if not matches:
            raise FileNotFoundError(
                f"Missing partition file for family={family}, bank={bank}, split={split_dir}, m={m}"
            )
        out.append(np.load(matches[0]))
    return out


def _cvar(vals: np.ndarray, q: float) -> float:
    return cvar_top_fraction(vals, q)


def _cell_indices(cells: np.ndarray, min_cell: int) -> List[np.ndarray]:
    idxs: List[np.ndarray] = []
    for k in np.unique(cells):
        idx = np.where(cells == k)[0]
        if idx.size >= int(min_cell):
            idxs.append(idx)
    return idxs


def _find_worst_cell_stats(
    losses: np.ndarray,
    partitions: List[np.ndarray],
    cvar_q: float,
    min_cell: int,
    clip_loss: Optional[float],
) -> Dict[str, float]:
    worst_cvar = float("-inf")
    worst_clip_rate = float("nan")
    worst_cell_size = 0

    for cells in partitions:
        idxs = _cell_indices(cells.astype(np.int64), min_cell=min_cell)
        for idx in idxs:
            vals = losses[idx]
            c = _cvar(vals, cvar_q)
            if np.isnan(c):
                continue
            if c > worst_cvar:
                worst_cvar = float(c)
                worst_cell_size = int(idx.size)
                if clip_loss is None:
                    worst_clip_rate = 0.0
                else:
                    worst_clip_rate = float(np.mean(vals > float(clip_loss)))

    if not np.isfinite(worst_cvar):
        return {
            "anchor_worst_cell_cvar": float("nan"),
            "anchor_worst_cell_clip_rate": float("nan"),
            "anchor_worst_cell_size": 0,
            "anchor_rho_cvar_clip": float("nan"),
        }

    rho = float(min(worst_clip_rate / float(cvar_q), 1.0)) if np.isfinite(worst_clip_rate) else float("nan")
    return {
        "anchor_worst_cell_cvar": worst_cvar,
        "anchor_worst_cell_clip_rate": worst_clip_rate,
        "anchor_worst_cell_size": worst_cell_size,
        "anchor_rho_cvar_clip": rho,
    }


def _resolve_run_dir(seed_root: Path, tag: str) -> Path:
    exact = seed_root / tag
    if exact.exists():
        return exact
    candidates = sorted([p.name for p in seed_root.iterdir() if p.is_dir()])
    raise FileNotFoundError(
        f"Could not resolve exact run dir in {seed_root} for tag='{tag}'. Available tags: {candidates}"
    )


def _load_epoch_clip_alpha(run_dir: Path) -> Dict[int, float]:
    """Return per-epoch active clip alpha when logged (dynamic clipping)."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    out: Dict[int, float] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = int(rec.get("epoch", -1))
        if ep <= 0:
            continue
        a = rec.get("clip_alpha_active")
        if a is None:
            continue
        try:
            a_f = float(a)
        except (TypeError, ValueError):
            continue
        if np.isfinite(a_f):
            out[ep] = a_f
    return out


def _build_tag_map(df_pockets: pd.DataFrame) -> Dict[Tuple[str, int, int], List[str]]:
    tag_map: Dict[Tuple[str, int, int], List[str]] = {}
    sub = df_pockets[df_pockets["split"] == "val"][["regime", "seed", "epoch", "tag"]].drop_duplicates()
    for (regime, seed, epoch), grp in sub.groupby(["regime", "seed", "epoch"]):
        tags = sorted({str(t) for t in grp["tag"].tolist()})
        tag_map[(str(regime), int(seed), int(epoch))] = tags
    return tag_map


def _mean_ci(vals: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    if vals.size == 1:
        return m, 0.0
    ci = ci95_mean(vals)
    return m, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["celeba", "waterbirds", "camelyon17"])
    ap.add_argument("--selected_rows_csv", required=True)
    ap.add_argument(
        "--selection_mode",
        default="selected_best_proxy",
        help="Filter selected_rows by selection_mode. Use 'all' to disable filtering.",
    )
    ap.add_argument("--pockets_csv", required=True)
    ap.add_argument("--families", default="teacher_difficulty")
    ap.add_argument("--banks", default="A,B")
    ap.add_argument("--regimes", default="rcgdro,rcgdro_softclip_p95_a10,rcgdro_softclip_p97_a10,rcgdro_softclip_p99_a10,rcgdro_softclip_p95_a10_wb,rcgdro_softclip_p97_a10_wb,rcgdro_softclip_p99_a10_wb,rcgdro_softclip_p95_a10_cam,rcgdro_softclip_p97_a10_cam,rcgdro_softclip_p99_a10_cam")
    ap.add_argument("--min_cell", type=int, default=20)
    ap.add_argument("--cvar_q", type=float, default=0.1)
    ap.add_argument("--split", default="val_skew")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = str(cfg["dataset"]["name"])
    backbone = str(cfg["embeddings"]["backbone"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_root = Path(cfg["project"]["runs_dir"])

    eval_version = cfg.get("partitions", {}).get("eval_version")
    if eval_version:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}" / str(eval_version)
    else:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}"
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval banks at {eval_root}")

    df_sel = pd.read_csv(args.selected_rows_csv)
    sel_mode = str(args.selection_mode).strip().lower()
    if "selection_mode" in df_sel.columns and sel_mode not in {"", "all", "any"}:
        df_sel = df_sel[df_sel["selection_mode"].astype(str) == str(args.selection_mode)].copy()

    keep_regimes = {r.strip() for r in str(args.regimes).split(",") if r.strip()}
    df_sel = df_sel[df_sel["regime"].isin(keep_regimes)].copy()
    if df_sel.empty:
        raise ValueError(
            "No selected rows after filtering. "
            f"selection_mode={args.selection_mode}, regimes={sorted(keep_regimes)}"
        )
    required_cols = {"regime", "seed", "epoch"}
    missing = sorted(required_cols.difference(df_sel.columns))
    if missing:
        raise ValueError(
            f"selected_rows_csv is missing required columns: {missing}. "
            "Expected row-level table (not a regime-level summary)."
        )

    df_pockets = pd.read_csv(args.pockets_csv)
    tag_map = _build_tag_map(df_pockets)

    families = [f.strip() for f in str(args.families).split(",") if f.strip()]
    banks = [b.strip() for b in str(args.banks).split(",") if b.strip()]

    # Pre-load partitions per family/bank once.
    parts_cache: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for fam in families:
        fam_meta = json.loads((eval_root / fam / "meta.json").read_text(encoding="utf-8"))
        M = int(fam_meta.get("num_partitions", 1))
        for bank in banks:
            parts_cache[(fam, bank)] = _load_partitions(
                eval_root=eval_root,
                family=fam,
                bank=bank,
                split=str(args.split),
                num_parts=M,
            )

    rows: List[Dict] = []
    for _, r in df_sel.iterrows():
        regime = str(r["regime"])
        seed = int(r["seed"])
        epoch = int(r["epoch"])
        key = (regime, seed, epoch)
        row_tag = str(r["tag"]).strip() if ("tag" in df_sel.columns and pd.notna(r.get("tag"))) else ""
        if row_tag:
            tag = row_tag
        else:
            candidates = tag_map.get(key, [])
            if not candidates:
                raise KeyError(f"No tag found for dataset={dataset_name}, regime={regime}, seed={seed}, epoch={epoch}")
            if len(candidates) != 1:
                raise ValueError(
                    f"Ambiguous tags for dataset={dataset_name}, regime={regime}, seed={seed}, epoch={epoch}: {candidates}"
                )
            tag = candidates[0]

        seed_root = runs_root / dataset_name / regime / f"seed{seed}"
        run_dir = _resolve_run_dir(seed_root=seed_root, tag=tag)
        epoch_clip_alpha = _load_epoch_clip_alpha(run_dir)

        val_loss = np.load(run_dir / "val_loss_by_epoch.npy", mmap_mode="r")
        losses = np.asarray(val_loss[epoch - 1], dtype=np.float64)

        run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        clip_loss = run_cfg.get("clip_loss")
        clip_alpha = run_cfg.get("clip_alpha", 1.0)
        if clip_loss is None:
            training = run_cfg.get("training", {})
            clip_loss = training.get("clip_loss")
            clip_alpha = training.get("clip_alpha", clip_alpha)

        # Some baseline runs persist clip fields as zeros; treat clipping as active
        # only for explicit softclip regimes with valid positive threshold and 0<alpha<1.
        try:
            clip_loss_num = float(clip_loss) if clip_loss is not None else float("nan")
        except (TypeError, ValueError):
            clip_loss_num = float("nan")
        try:
            clip_alpha_num = float(clip_alpha) if clip_alpha is not None else float("nan")
        except (TypeError, ValueError):
            clip_alpha_num = float("nan")
        clip_alpha_ep = float(epoch_clip_alpha.get(epoch, clip_alpha_num))

        clip_active = bool(np.isfinite(clip_loss_num) and (clip_loss_num > 0.0) and np.isfinite(clip_alpha_ep))

        if not clip_active:
            frac_clip = 0.0
            mean_excess = float("nan")
            distortion_mass = 0.0
            clip_loss_out = float("nan")
            clip_alpha_out = 1.0
        else:
            clip_loss_out = clip_loss_num
            clip_alpha_out = clip_alpha_ep
            mask = losses > clip_loss_out
            frac_clip = float(np.mean(mask))
            mean_excess = float(np.mean(losses[mask] - clip_loss_out)) if np.any(mask) else 0.0
            distortion_mass = float((1.0 - clip_alpha_out) * frac_clip * mean_excess)

        # Worst-cell clipping concentration on anchor partitions.
        anchor_stats_rows: List[Dict[str, float]] = []
        for fam in families:
            for bank in banks:
                stats = _find_worst_cell_stats(
                    losses=losses,
                    partitions=parts_cache[(fam, bank)],
                    cvar_q=float(args.cvar_q),
                    min_cell=int(args.min_cell),
                    clip_loss=None if np.isnan(clip_loss_out) else clip_loss_out,
                )
                stats["family"] = fam
                stats["bank"] = bank
                anchor_stats_rows.append(stats)

        # Select the anchor/bank with highest worst-cell CVaR.
        anchor_stats_rows = sorted(
            anchor_stats_rows,
            key=lambda z: -np.inf if not np.isfinite(z["anchor_worst_cell_cvar"]) else z["anchor_worst_cell_cvar"],
            reverse=True,
        )
        best_anchor = anchor_stats_rows[0] if anchor_stats_rows else {
            "family": "na",
            "bank": "na",
            "anchor_worst_cell_cvar": float("nan"),
            "anchor_worst_cell_clip_rate": float("nan"),
            "anchor_worst_cell_size": 0,
            "anchor_rho_cvar_clip": float("nan"),
        }

        tail_selected = r.get("tail_worst_cvar", np.nan)
        if not np.isfinite(pd.to_numeric(tail_selected, errors="coerce")):
            tail_selected = best_anchor["anchor_worst_cell_cvar"]

        rows.append(
            {
                "dataset": dataset_name,
                "regime": regime,
                "seed": seed,
                "epoch_selected": epoch,
                "tag": tag,
                "run_dir": str(run_dir),
                "clip_loss": clip_loss_out,
                "clip_alpha": clip_alpha_out,
                "frac_clipped_selected": frac_clip,
                "mean_excess_selected": mean_excess,
                "distortion_mass_selected": distortion_mass,
                "tail_worst_cvar_selected": float(tail_selected),
                "perf_selected": float(
                    r.get("test_hosp_2_acc", r.get("oracle_wg_acc", r.get("test_acc", np.nan)))
                ),
                "anchor_family": best_anchor["family"],
                "anchor_bank": best_anchor["bank"],
                "anchor_worst_cell_cvar": best_anchor["anchor_worst_cell_cvar"],
                "anchor_worst_cell_clip_rate": best_anchor["anchor_worst_cell_clip_rate"],
                "anchor_worst_cell_size": best_anchor["anchor_worst_cell_size"],
                "anchor_rho_cvar_clip": best_anchor["anchor_rho_cvar_clip"],
            }
        )

    df_rows = pd.DataFrame(rows)

    # Tail deltas versus the family baseline within the selected rows.
    if not df_rows[df_rows["regime"] == "rcgdro"].empty:
        baseline_regime = "rcgdro"
    elif not df_rows[df_rows["regime"] == "erm"].empty:
        baseline_regime = "erm"
    else:
        baseline_regime = ""

    if baseline_regime:
        base_tail = float(df_rows[df_rows["regime"] == baseline_regime]["tail_worst_cvar_selected"].mean())
        df_rows["baseline_regime"] = baseline_regime
        df_rows["tail_delta_vs_baseline"] = df_rows["tail_worst_cvar_selected"] - base_tail
    else:
        df_rows["baseline_regime"] = ""
        df_rows["tail_delta_vs_baseline"] = np.nan

    # Summary by regime.
    srows: List[Dict] = []
    for regime, grp in df_rows.groupby("regime"):
        frac_m, frac_ci = _mean_ci(grp["frac_clipped_selected"].to_numpy())
        mex_m, mex_ci = _mean_ci(grp["mean_excess_selected"].to_numpy())
        dist_m, dist_ci = _mean_ci(grp["distortion_mass_selected"].to_numpy())
        rho_m, rho_ci = _mean_ci(grp["anchor_rho_cvar_clip"].to_numpy())
        tail_m, tail_ci = _mean_ci(grp["tail_worst_cvar_selected"].to_numpy())
        taild_m, taild_ci = _mean_ci(grp["tail_delta_vs_baseline"].to_numpy())
        srows.append(
            {
                "dataset": dataset_name,
                "regime": regime,
                "baseline_regime": baseline_regime,
                "n": int(grp.shape[0]),
                "frac_clipped_mean": frac_m,
                "frac_clipped_ci95": frac_ci,
                "mean_excess_mean": mex_m,
                "mean_excess_ci95": mex_ci,
                "distortion_mass_mean": dist_m,
                "distortion_mass_ci95": dist_ci,
                "rho_worst_mean": rho_m,
                "rho_worst_ci95": rho_ci,
                "tail_worst_cvar_mean": tail_m,
                "tail_worst_cvar_ci95": tail_ci,
                "tail_delta_vs_baseline_mean": taild_m,
                "tail_delta_vs_baseline_ci95": taild_ci,
            }
        )
    df_summary = pd.DataFrame(srows).sort_values("regime")

    out_dir = artifacts_dir / "metrics"
    ensure_dir(out_dir)
    suf = str(args.out_suffix).strip()
    suf = f"_{suf}" if suf else ""
    out_rows = out_dir / f"{dataset_name}_tail_distortion_rows{suf}.csv"
    out_summary = out_dir / f"{dataset_name}_tail_distortion_summary{suf}.csv"
    df_rows.to_csv(out_rows, index=False)
    df_summary.to_csv(out_summary, index=False)
    print(f"[tail-distortion] wrote rows: {out_rows}")
    print(f"[tail-distortion] wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
