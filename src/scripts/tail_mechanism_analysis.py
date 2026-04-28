import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import load_config
from ..utils.io import ensure_dir
from ..utils.stats import ci95_mean


@dataclass
class RunInfo:
    regime: str
    seed: int
    tag: str
    run_dir: Path


def _discover_runs(
    runs_root: Path,
    dataset: str,
    regimes: List[str],
    tag_filters: List[str],
) -> Dict[Tuple[str, int], RunInfo]:
    out: Dict[Tuple[str, int], RunInfo] = {}
    for regime in regimes:
        reg_dir = runs_root / dataset / regime
        if not reg_dir.exists():
            continue
        for seed_dir in sorted(reg_dir.glob("seed*")):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except Exception:
                continue
            candidates = []
            for tag_dir in sorted(seed_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                if not (tag_dir / "config.json").exists():
                    continue
                if tag_filters and not any(t in tag_dir.name for t in tag_filters):
                    continue
                candidates.append(tag_dir)
            if not candidates:
                continue
            # Prefer the first match (tags are unique in this setup).
            tag_dir = candidates[0]
            out[(regime, seed)] = RunInfo(regime=regime, seed=seed, tag=tag_dir.name, run_dir=tag_dir)
    return out


def _mean_ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    mean = float(x.mean())
    if x.size == 1:
        return mean, 0.0
    std = float(x.std(ddof=1))
    ci = ci95_mean(x)
    return mean, float(ci)


def _load_bins(eval_root: Path, family: str, bank: str, split: str) -> Tuple[np.ndarray, int]:
    base = eval_root / family / f"bank{bank}" / split
    matches = list(base.glob("diff_m00_K*.npy"))
    if not matches:
        raise FileNotFoundError(f"Missing difficulty bins at {base}")
    bins = np.load(matches[0])
    name = matches[0].name
    k_str = name.split("K")[-1].split(".")[0]
    K = int(k_str)
    return bins.astype(np.int64), K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--clip_threshold", type=float, default=0.810722)
    ap.add_argument("--difficulty_family", default="teacher_difficulty")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]
    runs_root = Path(cfg["project"]["runs_dir"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    figures_dir = Path(cfg["project"]["figures_dir"])
    ensure_dir(figures_dir)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    tag_filters = [t.strip() for t in args.tag_filter.split(",") if t.strip()]
    runs = _discover_runs(runs_root, dataset, regimes, tag_filters)
    if not runs:
        raise FileNotFoundError("No runs found for regimes/tag filter.")

    summary = pd.read_csv(args.summary_csv)
    summary = summary[summary["regime"].isin(regimes)].copy()
    if tag_filters:
        summary = summary[summary["regime"].isin(regimes)]

    eval_version = cfg.get("partitions", {}).get("eval_version")
    if eval_version:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset}_{backbone}" / str(eval_version)
    else:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset}_{backbone}"
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval banks at {eval_root}")

    banks = cfg.get("partitions", {}).get("eval_banks", {}).get("banks", ["A", "B"])
    rows = []

    for _, row in summary.iterrows():
        regime = row["regime"]
        seed = int(row["seed"])
        epoch = int(row["epoch"])
        run = runs.get((regime, seed))
        if run is None:
            continue
        run_cfg = json.loads((run.run_dir / "config.json").read_text())
        clip_loss = float(run_cfg.get("clip_loss", 0.0) or 0.0)
        threshold = float(args.clip_threshold) if clip_loss <= 0 else float(clip_loss)

        val_loss = np.load(run.run_dir / "val_loss_by_epoch.npy")
        losses = val_loss[epoch - 1].astype(np.float64, copy=False)

        overall_clip_rate = float(np.mean(losses > threshold))

        top_rates = []
        for bank in banks:
            bins, K = _load_bins(eval_root, args.difficulty_family, bank, "val_skew")
            top_start = int(np.floor(0.9 * K))
            top_mask = bins >= top_start
            if top_mask.sum() == 0:
                continue
            top_rates.append(float(np.mean(losses[top_mask] > threshold)))

        if top_rates:
            top_clip_rate = float(np.mean(top_rates))
        else:
            top_clip_rate = np.nan

        ratio = float(top_clip_rate / (overall_clip_rate + 1e-8)) if np.isfinite(top_clip_rate) else np.nan

        rows.append(
            {
                "regime": regime,
                "seed": seed,
                "epoch": epoch,
                "clip_loss_active": clip_loss,
                "clip_threshold_used": threshold,
                "overall_clip_rate": overall_clip_rate,
                "top10_clip_rate": top_clip_rate,
                "top10_ratio": ratio,
            }
        )

    df = pd.DataFrame(rows)
    out_suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_csv = figures_dir / f"{dataset}_clip_rate_top10{out_suffix}.csv"
    df.to_csv(out_csv, index=False)

    # Aggregate summary + plot
    agg_rows = []
    for regime, sub in df.groupby("regime"):
        mean_overall, ci_overall = _mean_ci(sub["overall_clip_rate"])
        mean_top, ci_top = _mean_ci(sub["top10_clip_rate"])
        mean_ratio, ci_ratio = _mean_ci(sub["top10_ratio"])
        agg_rows.append(
            {
                "regime": regime,
                "overall_clip_rate_mean": mean_overall,
                "overall_clip_rate_ci": ci_overall,
                "top10_clip_rate_mean": mean_top,
                "top10_clip_rate_ci": ci_top,
                "top10_ratio_mean": mean_ratio,
                "top10_ratio_ci": ci_ratio,
                "n": int(sub.shape[0]),
            }
        )

    df_agg = pd.DataFrame(agg_rows).sort_values("regime")
    out_agg = figures_dir / f"{dataset}_clip_rate_top10_summary{out_suffix}.csv"
    df_agg.to_csv(out_agg, index=False)

    # Plot ratio with CI
    plt.figure(figsize=(7.5, 4.5), dpi=160)
    x = np.arange(len(df_agg))
    plt.bar(x, df_agg["top10_ratio_mean"], yerr=df_agg["top10_ratio_ci"], capsize=4)
    plt.xticks(x, df_agg["regime"], rotation=20)
    plt.ylabel("Clip rate ratio (top10 / overall)")
    plt.title("Tail suppression diagnostic (teacher difficulty top decile)")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{dataset}_clip_rate_top10_ratio{out_suffix}.png")
    plt.close()


if __name__ == "__main__":
    main()
