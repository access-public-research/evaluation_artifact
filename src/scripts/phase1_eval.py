import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils.io import ensure_dir
from ..utils.stats import cvar_top_fraction


@dataclass
class RunInfo:
    dataset: str
    regime: str
    seed: int
    tag: str
    run_dir: Path


def discover_runs(runs_root: Path, dataset: str, regimes: Iterable[str]) -> List[RunInfo]:
    out: List[RunInfo] = []
    for regime in regimes:
        regime_dir = runs_root / dataset / regime
        if not regime_dir.exists():
            continue
        for seed_dir in sorted(regime_dir.glob("seed*")):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except Exception:
                continue
            for tag_dir in sorted(seed_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                if not (tag_dir / "config.json").exists():
                    continue
                out.append(RunInfo(dataset=dataset, regime=regime, seed=seed, tag=tag_dir.name, run_dir=tag_dir))
    return out


def _parse_epoch_subset(spec: str, max_epochs: int) -> List[int]:
    spec = (spec or "").strip().lower()
    if not spec or spec == "all":
        return list(range(1, max_epochs + 1))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i = max(1, int(lo))
            hi_i = min(max_epochs, int(hi))
            out.extend(list(range(lo_i, hi_i + 1)))
        else:
            idx = int(part)
            if 1 <= idx <= max_epochs:
                out.append(idx)
    return sorted(set(out))


def _load_partitions(eval_root: Path, family: str, bank: str, split: str, prefix: str, num_parts: int):
    split = split.lower()
    if split == "val_skew":
        split = "val_skew"
    elif split in {"val", "validation"}:
        split = "validation"
    elif split == "train":
        split = "train"
    else:
        raise ValueError(f"Unknown split: {split}")

    parts = []
    base = eval_root / family / f"bank{bank}" / split
    for m in range(int(num_parts)):
        matches = list(base.glob(f"{prefix}_m{m:02d}_K*.npy"))
        if not matches:
            raise FileNotFoundError(f"Missing partition for {family} bank{bank} split={split} m={m}")
        parts.append(np.load(matches[0]))
    return parts


def _infer_prefix(family: str, meta: dict) -> str:
    prefix = str(meta.get("prefix", "")).strip()
    if prefix:
        return prefix
    name = family.lower()
    if "hash" in name:
        return "hash"
    if "proj" in name:
        return "proj"
    if "conf" in name:
        return "conf"
    if "diff" in name or "difficulty" in name:
        return "diff"
    raise ValueError(f"Could not infer prefix for family {family}. Add prefix to meta.json.")


def _load_families(eval_root: Path, include: List[str] | None) -> List[Tuple[str, str, int, int]]:
    fams: List[Tuple[str, str, int, int]] = []
    for fam_dir in sorted(eval_root.iterdir()):
        if not fam_dir.is_dir():
            continue
        name = fam_dir.name
        if include and name not in include:
            continue
        meta_path = fam_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        prefix = _infer_prefix(name, meta)
        K = int(meta.get("num_cells"))
        M = int(meta.get("num_partitions", 1))
        fams.append((name, prefix, K, M))
    if include and not fams:
        raise FileNotFoundError(f"No eval families found for include={include} under {eval_root}.")
    return fams


def _cell_indices(cells: np.ndarray, num_cells: int, min_cell: int) -> Tuple[List[np.ndarray], np.ndarray]:
    idxs: List[np.ndarray] = []
    counts = np.zeros((int(num_cells),), dtype=np.int64)
    for k in range(int(num_cells)):
        idx = np.where(cells == k)[0]
        counts[k] = int(idx.size)
        if idx.size >= int(min_cell):
            idxs.append(idx)
    return idxs, counts


def _cvar(vals: np.ndarray, q: float) -> float:
    return cvar_top_fraction(vals, q)


def _within_cell_metrics(
    losses: np.ndarray,
    correct: np.ndarray | None,
    idxs: List[np.ndarray],
    q: float,
) -> Dict[str, float]:
    if not idxs:
        return {
            "within_cvar_mean": float("nan"),
            "within_cvar_weighted": float("nan"),
            "worst_cell_cvar": float("nan"),
            "worst_cell_mean_loss": float("nan"),
            "worst_cell_mean_acc": float("nan"),
        }

    cvars = []
    weights = []
    mean_losses = []
    mean_accs = []
    for idx in idxs:
        vals = losses[idx]
        c = _cvar(vals, q)
        cvars.append(c)
        weights.append(float(idx.size))
        mean_losses.append(float(np.mean(vals)))
        if correct is not None:
            mean_accs.append(float(np.mean(correct[idx])))

    within_mean = float(np.mean(cvars))
    within_weighted = float(np.average(cvars, weights=weights)) if weights else within_mean
    worst_cvar = float(np.max(cvars))
    worst_mean_loss = float(np.max(mean_losses)) if mean_losses else float("nan")
    worst_mean_acc = float(np.min(mean_accs)) if mean_accs else float("nan")

    return {
        "within_cvar_mean": within_mean,
        "within_cvar_weighted": within_weighted,
        "worst_cell_cvar": worst_cvar,
        "worst_cell_mean_loss": worst_mean_loss,
        "worst_cell_mean_acc": worst_mean_acc,
    }


def _oracle_worst_group_acc(correct: np.ndarray, groups: np.ndarray) -> float:
    worst = 1.0
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        acc = float(np.mean(correct[mask]))
        worst = min(worst, acc)
    return float(worst)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regimes", default="erm,rcgdro")
    ap.add_argument("--min_cell", type=int, default=20)
    ap.add_argument("--cvar_q", type=float, default=0.1)
    ap.add_argument("--train_epoch_subset", default="1-5,26-30")
    ap.add_argument("--train_families", default="global_hash,within_label_hash")
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--exclude_tag_filter", default="")
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--families", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
    if not feat_dir.exists():
        raise FileNotFoundError(f"Missing embeddings at {feat_dir}.")

    eval_version = cfg.get("partitions", {}).get("eval_version")
    if eval_version:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}" / str(eval_version)
    else:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}"
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval banks at {eval_root}. Run build_eval_banks first.")

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    runs_root = Path(cfg["project"]["runs_dir"])
    runs = discover_runs(runs_root, dataset_name, regimes)
    if not runs:
        raise FileNotFoundError(f"No runs found under {runs_root / dataset_name} for regimes={regimes}")

    tag_filters = [t.strip() for t in args.tag_filter.split(",") if t.strip()]
    exclude_tag_filter = args.exclude_tag_filter.strip()
    if tag_filters:
        runs = [r for r in runs if any(t in r.tag for t in tag_filters)]
    if exclude_tag_filter:
        runs = [r for r in runs if exclude_tag_filter not in r.tag]

    include = [s.strip() for s in args.families.split(",") if s.strip()]
    families = _load_families(eval_root, include if include else None)
    tf_raw = [f.strip() for f in args.train_families.split(",") if f.strip()]
    if "all" in tf_raw or "*" in tf_raw:
        train_fams = {f[0] for f in families}
    else:
        train_fams = set(tf_raw)

    eval_cfg = cfg.get("partitions", {}).get("eval_banks", {})
    banks = eval_cfg.get("banks", ["A", "B"])

    y_val = np.load(feat_dir / "y_val_skew.npy")
    g_val = np.load(feat_dir / "g_val_skew.npy")

    out_dir = artifacts_dir / "metrics"
    ensure_dir(out_dir)
    suffix = str(args.out_suffix).strip()
    suffix = f"_{suffix}" if suffix else ""
    out_pockets = out_dir / f"{dataset_name}_{backbone}_phase1_pockets{suffix}.csv"
    out_range = out_dir / f"{dataset_name}_{backbone}_phase1_proxy_range{suffix}.csv"
    out_spearman = out_dir / f"{dataset_name}_{backbone}_phase1_spearman_eval{suffix}.csv"
    out_churn = out_dir / f"{dataset_name}_{backbone}_phase1_bin_churn{suffix}.csv"
    if out_pockets.exists() and out_range.exists() and out_spearman.exists() and not int(args.overwrite):
        print("[phase1] outputs already exist; use --overwrite 1 to regenerate.")
        return

    rows: List[Dict] = []
    churn_rows: List[Dict] = []
    for run in runs:
        val_loss = np.load(run.run_dir / "val_loss_by_epoch.npy")
        val_correct = np.load(run.run_dir / "val_correct_by_epoch.npy")
        E, N = val_loss.shape
        if int(y_val.shape[0]) != int(N):
            raise ValueError(f"Val size mismatch for {run.run_dir}: val arrays={N} labels={y_val.shape[0]}")

        # Precompute oracle worst-group accuracy per epoch.
        oracle_wg = np.zeros((E,), dtype=np.float64)
        for e in range(E):
            oracle_wg[e] = _oracle_worst_group_acc(val_correct[e], g_val)

        # Load train losses (if needed).
        train_loss = None
        train_epochs = []
        if train_fams:
            train_loss_path = run.run_dir / "train_loss_by_epoch.npy"
            if train_loss_path.exists():
                train_loss = np.load(train_loss_path, mmap_mode="r")
                train_epochs = _parse_epoch_subset(args.train_epoch_subset, int(train_loss.shape[0]))

        for fam_name, prefix, K, M in families:
            for bank in banks:
                parts_val = _load_partitions(eval_root, fam_name, bank, "val_skew", prefix, M)
                idxs_val = []
                for cells in parts_val:
                    idxs, _counts = _cell_indices(cells.astype(np.int64), K, int(args.min_cell))
                    idxs_val.append(idxs)

                # Val metrics per epoch.
                for e in range(E):
                    losses_e = val_loss[e].astype(np.float64, copy=False)
                    correct_e = val_correct[e].astype(np.float64, copy=False)
                    for m, idxs in enumerate(idxs_val):
                        metrics = _within_cell_metrics(losses_e, correct_e, idxs, float(args.cvar_q))
                        rows.append(
                            {
                                "dataset": run.dataset,
                                "regime": run.regime,
                                "seed": int(run.seed),
                                "tag": run.tag,
                                "epoch": int(e + 1),
                                "split": "val",
                                "family": fam_name,
                                "bank": bank,
                                "partition": int(m),
                                "within_cvar_mean": metrics["within_cvar_mean"],
                                "within_cvar_weighted": metrics["within_cvar_weighted"],
                                "worst_cell_cvar": metrics["worst_cell_cvar"],
                                "worst_cell_mean_loss": metrics["worst_cell_mean_loss"],
                                "worst_cell_mean_acc": metrics["worst_cell_mean_acc"],
                                "oracle_wg_acc": float(oracle_wg[e]),
                            }
                        )

                # Train metrics (subset of epochs, proxy families only).
                if train_loss is not None and fam_name in train_fams and train_epochs:
                    parts_train = _load_partitions(eval_root, fam_name, bank, "train", prefix, M)
                    idxs_train = []
                    for cells in parts_train:
                        idxs, _counts = _cell_indices(cells.astype(np.int64), K, int(args.min_cell))
                        idxs_train.append(idxs)
                    for e in train_epochs:
                        e_idx = int(e - 1)
                        losses_e = np.asarray(train_loss[e_idx], dtype=np.float64)
                        for m, idxs in enumerate(idxs_train):
                            metrics = _within_cell_metrics(losses_e, None, idxs, float(args.cvar_q))
                            rows.append(
                                {
                                    "dataset": run.dataset,
                                    "regime": run.regime,
                                    "seed": int(run.seed),
                                    "tag": run.tag,
                                    "epoch": int(e),
                                    "split": "train",
                                    "family": fam_name,
                                    "bank": bank,
                                    "partition": int(m),
                                    "within_cvar_mean": metrics["within_cvar_mean"],
                                    "within_cvar_weighted": metrics["within_cvar_weighted"],
                                    "worst_cell_cvar": metrics["worst_cell_cvar"],
                                    "worst_cell_mean_loss": metrics["worst_cell_mean_loss"],
                                    "worst_cell_mean_acc": float("nan"),
                                    "oracle_wg_acc": float("nan"),
                                }
                            )

    df = pd.DataFrame(rows)
    df.to_csv(out_pockets, index=False)

    # Proxy score dynamics: per run, per family/bank on val.
    df_val = df[df["split"] == "val"].copy()
    agg = df_val.groupby(
        ["dataset", "regime", "seed", "tag", "family", "bank", "epoch"],
        as_index=False,
    ).mean(numeric_only=True)
    range_rows: List[Dict] = []
    for (dataset, regime, seed, tag, family, bank), grp in agg.groupby(
        ["dataset", "regime", "seed", "tag", "family", "bank"]
    ):
        worst_loss = grp["worst_cell_mean_loss"].to_numpy()
        worst_acc = grp["worst_cell_mean_acc"].to_numpy()
        cvar = grp["worst_cell_cvar"].to_numpy()
        range_rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": int(seed),
                "tag": tag,
                "family": family,
                "bank": bank,
                "worst_loss_var": float(np.nanvar(worst_loss)),
                "worst_loss_iqr": float(np.nanpercentile(worst_loss, 75) - np.nanpercentile(worst_loss, 25)),
                "worst_acc_var": float(np.nanvar(worst_acc)),
                "worst_acc_iqr": float(np.nanpercentile(worst_acc, 75) - np.nanpercentile(worst_acc, 25)),
                "worst_cvar_var": float(np.nanvar(cvar)),
                "worst_cvar_iqr": float(np.nanpercentile(cvar, 75) - np.nanpercentile(cvar, 25)),
            }
        )
    pd.DataFrame(range_rows).to_csv(out_range, index=False)

    # Spearman across seeds per epoch (val only).
    spearman_rows: List[Dict] = []
    for (dataset, regime, tag, family, bank, epoch), grp in agg.groupby(
        ["dataset", "regime", "tag", "family", "bank", "epoch"]
    ):
        proxy_loss = grp["worst_cell_mean_loss"].to_numpy()
        proxy_acc = grp["worst_cell_mean_acc"].to_numpy()
        proxy_cvar = grp["worst_cell_cvar"].to_numpy()
        oracle = grp["oracle_wg_acc"].to_numpy()
        spearman_rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "tag": tag,
                "family": family,
                "bank": bank,
                "epoch": int(epoch),
                "spearman_proxy_loss": _spearman(proxy_loss, oracle),
                "spearman_proxy_acc": _spearman(proxy_acc, oracle),
                "spearman_proxy_cvar": _spearman(proxy_cvar, oracle),
                "n_seeds": int(grp.shape[0]),
            }
        )
    pd.DataFrame(spearman_rows).to_csv(out_spearman, index=False)

    # Optional bin churn output (for adaptive confidence bins).
    for run in runs:
        metrics_path = run.run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "train_bin_churn" not in rec:
                continue
            churn_rows.append(
                {
                    "dataset": run.dataset,
                    "regime": run.regime,
                    "seed": int(run.seed),
                    "tag": run.tag,
                    "epoch": int(rec.get("epoch", 0)),
                    "train_bin_churn": float(rec.get("train_bin_churn", 0.0)),
                }
            )
    if churn_rows:
        pd.DataFrame(churn_rows).to_csv(out_churn, index=False)

    print(f"[phase1] wrote {out_pockets}")
    print(f"[phase1] wrote {out_range}")
    print(f"[phase1] wrote {out_spearman}")
    if churn_rows:
        print(f"[phase1] wrote {out_churn}")


if __name__ == "__main__":
    main()
