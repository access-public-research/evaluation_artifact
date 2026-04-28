import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


def _discover_runs(runs_root: Path, dataset: str, regimes: List[str], tag_filters: List[str]) -> List[RunInfo]:
    out: List[RunInfo] = []
    for regime in regimes:
        reg_dir = runs_root / dataset / regime
        if not reg_dir.exists():
            continue
        for seed_dir in sorted(reg_dir.glob("seed*")):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except Exception:
                continue
            for tag_dir in sorted(seed_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                if not (tag_dir / "config.json").exists():
                    continue
                if tag_filters and not any(t in tag_dir.name for t in tag_filters):
                    continue
                out.append(RunInfo(regime=regime, seed=seed, tag=tag_dir.name, run_dir=tag_dir))
    return out


def _mean_ci(x: np.ndarray) -> (float, float):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    runs_root = Path(cfg["project"]["runs_dir"])
    figures_dir = Path(cfg["project"]["figures_dir"])
    ensure_dir(figures_dir)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    tag_filters = [t.strip() for t in args.tag_filter.split(",") if t.strip()]

    runs = _discover_runs(runs_root, dataset, regimes, tag_filters)
    if not runs:
        raise FileNotFoundError("No runs found for given regimes/tag filters.")

    rows = []
    for run in runs:
        metrics_path = run.run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        import json
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "regime": run.regime,
                    "seed": run.seed,
                    "epoch": int(rec.get("epoch", 0)),
                    "train_loss": float(rec.get("train_loss", np.nan)),
                    "val_loss": float(rec.get("val_loss", np.nan)),
                    "val_acc": float(rec.get("val_acc", np.nan)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No metrics found to plot.")

    out_suffix = f"_{args.out_suffix}" if args.out_suffix else ""

    # Aggregate by regime/epoch
    agg_rows = []
    for (regime, epoch), sub in df.groupby(["regime", "epoch"]):
        for metric in ["train_loss", "val_loss", "val_acc"]:
            mean, ci = _mean_ci(sub[metric])
            agg_rows.append(
                {
                    "regime": regime,
                    "epoch": int(epoch),
                    "metric": metric,
                    "mean": mean,
                    "ci": ci,
                }
            )
    agg = pd.DataFrame(agg_rows)

    out_csv = figures_dir / f"{dataset}_learning_curves{out_suffix}.csv"
    agg.to_csv(out_csv, index=False)

    def _plot(metric: str, ylabel: str, filename: str):
        plt.figure(figsize=(7.5, 4.5), dpi=160)
        for regime in regimes:
            sub = agg[(agg["regime"] == regime) & (agg["metric"] == metric)].sort_values("epoch")
            if sub.empty:
                continue
            x = sub["epoch"].to_numpy()
            y = sub["mean"].to_numpy()
            ci = sub["ci"].to_numpy()
            plt.plot(x, y, linewidth=2, label=regime)
            plt.fill_between(x, y - ci, y + ci, alpha=0.15)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{dataset}: {ylabel} over epochs")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figures_dir / filename)
        plt.close()

    _plot("val_acc", "Val accuracy", f"{dataset}_learning_val_acc{out_suffix}.png")
    _plot("val_loss", "Val loss", f"{dataset}_learning_val_loss{out_suffix}.png")
    _plot("train_loss", "Train loss", f"{dataset}_learning_train_loss{out_suffix}.png")


if __name__ == "__main__":
    main()
