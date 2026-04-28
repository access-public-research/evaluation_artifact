import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils.stats import ci95_mean
from .make_properness_plots import _discover_runs, _select_epoch
from .phase0_eval import _apply_objective_transform, _infer_prefix, _load_partitions, _proxy_metrics


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(ci95_mean(x))


def _resolve_eval_root(cfg: Dict) -> Path:
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]
    eval_version = cfg.get("partitions", {}).get("eval_version")
    if eval_version:
        return artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}" / str(eval_version)
    return artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}"


def _load_family_spec(eval_root: Path, family: str) -> Tuple[str, int, int]:
    meta_path = eval_root / family / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing family meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prefix = _infer_prefix(family, meta)
    return prefix, int(meta.get("num_cells")), int(meta.get("num_partitions", 1))


def _group_phase0(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["regime", "seed", "tag", "family", "epoch"], as_index=False).agg(
        {
            "proxy_worst_loss_min": "mean",
            "proxy_worst_loss_clip_min": "mean",
            "val_overall_loss": "mean",
        }
    )


def _select_rows(
    phase0: pd.DataFrame,
    runs_root: Path,
    dataset: str,
    regimes: List[str],
    family: str,
    selection_metric_mode: str,
    tag_filter: str,
) -> Dict[Tuple[str, int], Dict[str, object]]:
    runs = _discover_runs(runs_root, dataset, regimes)
    tag_filters = [t.strip() for t in str(tag_filter).split(",") if t.strip()]
    if tag_filters:
        runs = [r for r in runs if any(t in r.tag for t in tag_filters)]
    if not runs:
        raise FileNotFoundError(f"No runs found for regimes={regimes} tag_filter={tag_filter!r}")

    out: Dict[Tuple[str, int], Dict[str, object]] = {}
    for run in runs:
        proxy_df = phase0[
            (phase0["regime"] == run.regime)
            & (phase0["seed"] == run.seed)
            & (phase0["tag"] == run.tag)
            & (phase0["family"] == family)
        ].copy()
        if proxy_df.empty:
            continue
        epoch = _select_epoch(proxy_df, run.regime, mode=selection_metric_mode)
        out[(run.regime, int(run.seed))] = {
            "tag": run.tag,
            "run_dir": run.run_dir,
            "epoch": int(epoch),
        }
    return out


def _aggregate_proxy_metrics(
    *,
    eval_root: Path,
    family: str,
    banks: List[str],
    prefix: str,
    num_cells: int,
    num_parts: int,
    losses: np.ndarray,
    correct: np.ndarray,
    min_cell: int,
    cvar_q: float,
    clip_loss: float,
    clip_alpha: float,
) -> tuple[float, float]:
    ce_vals = []
    clip_vals = []
    losses_clip = _apply_objective_transform(losses, clip_loss=clip_loss, clip_alpha=clip_alpha)
    for bank in banks:
        parts = _load_partitions(eval_root, family, bank, "val_skew", prefix, num_parts)
        ce = _proxy_metrics(
            losses=losses,
            correct=correct,
            parts=parts,
            num_cells=num_cells,
            min_cell=min_cell,
            cvar_q=cvar_q,
        )
        cc = _proxy_metrics(
            losses=losses_clip,
            correct=correct,
            parts=parts,
            num_cells=num_cells,
            min_cell=min_cell,
            cvar_q=cvar_q,
        )
        ce_vals.append(float(ce["worst_loss"]))
        clip_vals.append(float(cc["worst_loss"]))
    return float(np.mean(ce_vals)), float(np.mean(clip_vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline_regime", required=True)
    ap.add_argument("--softclip_regime", required=True)
    ap.add_argument("--metrics_suffix", required=True)
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--selection_metric_mode", default="auto", choices=["auto", "proxy_unclipped", "proxy_clip"])
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--min_cell", type=int, default=20)
    ap.add_argument("--cvar_q", type=float, default=0.1)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_root = Path(cfg["project"]["runs_dir"])
    eval_root = _resolve_eval_root(cfg)
    family = str(args.proxy_family)
    prefix, num_cells, num_parts = _load_family_spec(eval_root, family)
    banks = list(cfg.get("partitions", {}).get("eval_banks", {}).get("banks", ["A", "B"]))

    phase0_path = artifacts_dir / "metrics" / f"{dataset}_{cfg['embeddings']['backbone']}_phase0_val_metrics_{args.metrics_suffix}.csv"
    phase0 = _group_phase0(pd.read_csv(phase0_path))
    selected = _select_rows(
        phase0=phase0,
        runs_root=runs_root,
        dataset=dataset,
        regimes=[str(args.baseline_regime), str(args.softclip_regime)],
        family=family,
        selection_metric_mode=str(args.selection_metric_mode),
        tag_filter=str(args.tag_filter),
    )

    rows: List[Dict[str, object]] = []
    baseline_regime = str(args.baseline_regime)
    softclip_regime = str(args.softclip_regime)
    common_seeds = sorted(
        set(seed for regime, seed in selected if regime == baseline_regime)
        & set(seed for regime, seed in selected if regime == softclip_regime)
    )
    if not common_seeds:
        raise RuntimeError("No overlapping seeds between baseline and softclip selections.")

    for seed in common_seeds:
        base = selected[(baseline_regime, seed)]
        soft = selected[(softclip_regime, seed)]

        soft_cfg = json.loads((Path(soft["run_dir"]) / "config.json").read_text(encoding="utf-8"))
        clip_loss = float(soft_cfg.get("clip_loss", 0.0) or 0.0)
        clip_alpha = float(soft_cfg.get("clip_alpha", 0.0) or 0.0)
        if not np.isfinite(clip_loss) or clip_loss <= 0:
            raise ValueError(f"Softclip regime {softclip_regime} seed{seed} is missing a valid clip_loss.")

        base_losses = np.load(Path(base["run_dir"]) / "val_loss_by_epoch.npy", mmap_mode="r")
        base_corr = np.load(Path(base["run_dir"]) / "val_correct_by_epoch.npy", mmap_mode="r")
        soft_losses = np.load(Path(soft["run_dir"]) / "val_loss_by_epoch.npy", mmap_mode="r")
        soft_corr = np.load(Path(soft["run_dir"]) / "val_correct_by_epoch.npy", mmap_mode="r")

        base_epoch = int(base["epoch"])
        soft_epoch = int(soft["epoch"])
        base_loss_e = np.asarray(base_losses[base_epoch - 1], dtype=np.float64)
        base_corr_e = np.asarray(base_corr[base_epoch - 1], dtype=np.float64)
        soft_loss_e = np.asarray(soft_losses[soft_epoch - 1], dtype=np.float64)
        soft_corr_e = np.asarray(soft_corr[soft_epoch - 1], dtype=np.float64)

        base_proxy_ce, base_proxy_clip = _aggregate_proxy_metrics(
            eval_root=eval_root,
            family=family,
            banks=banks,
            prefix=prefix,
            num_cells=num_cells,
            num_parts=num_parts,
            losses=base_loss_e,
            correct=base_corr_e,
            min_cell=int(args.min_cell),
            cvar_q=float(args.cvar_q),
            clip_loss=clip_loss,
            clip_alpha=clip_alpha,
        )
        soft_proxy_ce, soft_proxy_clip = _aggregate_proxy_metrics(
            eval_root=eval_root,
            family=family,
            banks=banks,
            prefix=prefix,
            num_cells=num_cells,
            num_parts=num_parts,
            losses=soft_loss_e,
            correct=soft_corr_e,
            min_cell=int(args.min_cell),
            cvar_q=float(args.cvar_q),
            clip_loss=clip_loss,
            clip_alpha=clip_alpha,
        )

        rows.append(
            {
                "seed": int(seed),
                "baseline_regime": baseline_regime,
                "softclip_regime": softclip_regime,
                "baseline_tag": str(base["tag"]),
                "softclip_tag": str(soft["tag"]),
                "baseline_epoch": base_epoch,
                "softclip_epoch": soft_epoch,
                "clip_loss": clip_loss,
                "clip_alpha": clip_alpha,
                "baseline_val_loss": float(np.mean(base_loss_e)),
                "softclip_val_loss": float(np.mean(soft_loss_e)),
                "baseline_proxy_ce": base_proxy_ce,
                "softclip_proxy_ce": soft_proxy_ce,
                "baseline_proxy_clip": base_proxy_clip,
                "softclip_proxy_clip": soft_proxy_clip,
                "delta_clip_proxy_soft_minus_base": float(soft_proxy_clip - base_proxy_clip),
                "delta_val_loss_soft_minus_base": float(np.mean(soft_loss_e) - np.mean(base_loss_e)),
            }
        )

    df = pd.DataFrame(rows).sort_values("seed")
    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_rows, index=False)

    deltas_proxy = pd.to_numeric(df["delta_clip_proxy_soft_minus_base"], errors="coerce").to_numpy()
    deltas_val = pd.to_numeric(df["delta_val_loss_soft_minus_base"], errors="coerce").to_numpy()
    proxy_mean, proxy_ci = _mean_ci(deltas_proxy)
    val_mean, val_ci = _mean_ci(deltas_val)
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "baseline_regime": baseline_regime,
                "softclip_regime": softclip_regime,
                "n": int(df.shape[0]),
                "delta_clip_proxy_mean": proxy_mean,
                "delta_clip_proxy_ci": proxy_ci,
                "delta_val_loss_mean": val_mean,
                "delta_val_loss_ci": val_ci,
                "n_proxy_better": int(np.sum(deltas_proxy < 0)),
                "n_proxy_better_and_val_loss_worse": int(np.sum((deltas_proxy < 0) & (deltas_val > 0))),
            }
        ]
    )
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
