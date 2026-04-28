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


def _discover_runs(runs_root: Path, dataset: str, regimes: List[str]) -> List[RunInfo]:
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
                out.append(RunInfo(regime=regime, seed=seed, tag=tag_dir.name, run_dir=tag_dir))
    return out


def _mean_ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    mean = float(x.mean())
    if x.size == 1:
        return mean, 0.0
    return mean, ci95_mean(x)


def _select_epoch(df: pd.DataFrame, regime: str, mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode == "proxy_unclipped":
        metric = df["proxy_worst_loss_min"]
    elif mode == "proxy_clip":
        metric = df["proxy_worst_loss_clip_min"]
        metric_vals = pd.to_numeric(metric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(metric_vals).any():
            metric = df["proxy_worst_loss_min"]
    else:
        # Backward-compatible default behavior.
        if "clip" in regime:
            metric = df["proxy_worst_loss_clip_min"]
            metric_vals = pd.to_numeric(metric, errors="coerce").to_numpy(dtype=np.float64)
            # Some run types (e.g., finetune re-runs) may not log clip-aware proxy values.
            # Fall back to the unclipped proxy selector when clip-aware values are unavailable.
            if not np.isfinite(metric_vals).any():
                metric = df["proxy_worst_loss_min"]
        else:
            metric = df["proxy_worst_loss_min"]
    idx = int(metric.idxmin())
    return int(df.loc[idx, "epoch"])


def _load_frac_clipped(run_dir: Path, epoch: int) -> Tuple[float, float, float]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return np.nan, np.nan, np.nan
    train_fc = np.nan
    val_fc = np.nan
    clip_alpha_active = np.nan
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if int(rec.get("epoch", -1)) == int(epoch):
            train_fc = rec.get("train_frac_clipped", np.nan)
            val_fc = rec.get("val_frac_clipped", np.nan)
            clip_alpha_active = rec.get("clip_alpha_active", np.nan)
            break
    return float(train_fc), float(val_fc), float(clip_alpha_active)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--metrics_suffix", required=True)
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--tail_family", default="teacher_difficulty")
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--fixed_epoch", type=int, default=-1)
    ap.add_argument(
        "--selection_metric_mode",
        default="auto",
        choices=["auto", "proxy_unclipped", "proxy_clip"],
        help="Checkpoint selector metric: auto (legacy), proxy_unclipped (stationary), or proxy_clip.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]
    runs_root = Path(cfg["project"]["runs_dir"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    figures_dir = Path(cfg["project"]["figures_dir"])
    ensure_dir(figures_dir)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    runs = _discover_runs(runs_root, dataset, regimes)
    tag_filters = [t.strip() for t in args.tag_filter.split(",") if t.strip()]
    if tag_filters:
        runs = [r for r in runs if any(t in r.tag for t in tag_filters)]

    if not runs:
        raise FileNotFoundError("No runs matched tag filter/regimes.")

    suffix = args.metrics_suffix
    val_path = artifacts_dir / "metrics" / f"{dataset}_{backbone}_phase0_val_metrics_{suffix}.csv"
    pockets_path = artifacts_dir / "metrics" / f"{dataset}_{backbone}_phase1_pockets_{suffix}.csv"
    if not val_path.exists() or not pockets_path.exists():
        raise FileNotFoundError("Missing phase0/phase1 metrics for suffix.")

    df_val = pd.read_csv(val_path)
    df_pockets = pd.read_csv(pockets_path)

    if tag_filters:
        df_val = df_val[df_val["tag"].apply(lambda t: any(f in t for f in tag_filters))]
        df_pockets = df_pockets[df_pockets["tag"].apply(lambda t: any(f in t for f in tag_filters))]

    df_pockets = df_pockets[df_pockets["split"] == "val"].copy()

    # Aggregate across banks
    df_val = df_val.groupby(["regime", "seed", "tag", "family", "epoch"], as_index=False).agg(
        {
            "proxy_worst_loss_min": "mean",
            "proxy_worst_loss_clip_min": "mean",
            "proxy_worst_acc_min": "mean",
            "val_overall_acc": "mean",
            "val_overall_loss": "mean",
        }
    )
    df_pockets = df_pockets.groupby(["regime", "seed", "tag", "family", "epoch"], as_index=False).agg(
        {
            "oracle_wg_acc": "mean",
            "within_cvar_mean": "mean",
            "worst_cell_cvar": "mean",
            "worst_cell_mean_loss": "mean",
        }
    )

    rows = []
    for run in runs:
        proxy_df = df_val[
            (df_val["regime"] == run.regime)
            & (df_val["seed"] == run.seed)
            & (df_val["tag"] == run.tag)
            & (df_val["family"] == args.proxy_family)
        ]
        if proxy_df.empty:
            continue
        if int(args.fixed_epoch) > 0:
            epoch = int(args.fixed_epoch)
            if int(epoch) not in set(proxy_df["epoch"].astype(int).tolist()):
                continue
        else:
            epoch = _select_epoch(proxy_df, run.regime, mode=args.selection_metric_mode)

        proxy_row = proxy_df[proxy_df["epoch"] == epoch].iloc[0]
        tail_row = df_pockets[
            (df_pockets["regime"] == run.regime)
            & (df_pockets["seed"] == run.seed)
            & (df_pockets["tag"] == run.tag)
            & (df_pockets["family"] == args.tail_family)
            & (df_pockets["epoch"] == epoch)
        ]
        if tail_row.empty:
            continue
        tail_row = tail_row.iloc[0]

        frac_train, frac_val, clip_alpha_active = _load_frac_clipped(run.run_dir, epoch)

        rows.append(
            {
                "regime": run.regime,
                "seed": run.seed,
                "epoch": epoch,
                "frac_clipped_train": frac_train,
                "frac_clipped_val": frac_val,
                "clip_alpha_active": clip_alpha_active,
                "proxy_worst_loss": float(proxy_row["proxy_worst_loss_min"]),
                "proxy_worst_loss_clip": float(proxy_row["proxy_worst_loss_clip_min"]),
                "proxy_worst_acc": float(proxy_row["proxy_worst_acc_min"]),
                "val_overall_acc": float(proxy_row["val_overall_acc"]),
                "val_overall_loss": float(proxy_row["val_overall_loss"]),
                "oracle_wg_acc": float(tail_row["oracle_wg_acc"]),
                "tail_within_cvar": float(tail_row["within_cvar_mean"]),
                "tail_worst_cvar": float(tail_row["worst_cell_cvar"]),
                "tail_worst_loss": float(tail_row["worst_cell_mean_loss"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No matching rows found to plot.")

    out_suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_csv = figures_dir / f"{dataset}_properness_summary{out_suffix}.csv"
    df.to_csv(out_csv, index=False)

    # Aggregate for plotting
    plot_rows = []
    for regime, sub in df.groupby("regime"):
        fx, fx_ci = _mean_ci(sub["frac_clipped_val"])
        y_proxy = sub["proxy_worst_loss_clip"] if "clip" in regime else sub["proxy_worst_loss"]
        p_mean, p_ci = _mean_ci(y_proxy)
        t_mean, t_ci = _mean_ci(sub["tail_worst_cvar"])
        wg_mean, wg_ci = _mean_ci(sub["oracle_wg_acc"])
        plot_rows.append(
            {
                "regime": regime,
                "frac_mean": fx,
                "frac_ci": fx_ci,
                "proxy_mean": p_mean,
                "proxy_ci": p_ci,
                "tail_mean": t_mean,
                "tail_ci": t_ci,
                "wg_mean": wg_mean,
                "wg_ci": wg_ci,
            }
        )

    plot_df = pd.DataFrame(plot_rows)

    def _scatter_plot(x_col, x_ci, y_col, y_ci, ylabel, out_name):
        plt.figure(figsize=(6.5, 4.5), dpi=160)
        for _, row in plot_df.iterrows():
            plt.errorbar(
                row[x_col],
                row[y_col],
                xerr=row[x_ci],
                yerr=row[y_ci],
                fmt="o",
                capsize=4,
                label=row["regime"],
            )
        plt.xlabel("Fraction clipped (val)")
        plt.ylabel(ylabel)
        plt.title(f"Properness axis: {ylabel}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figures_dir / out_name)
        plt.close()

    _scatter_plot("frac_mean", "frac_ci", "proxy_mean", "proxy_ci", "Proxy worst loss (selected)", f"{dataset}_properness_proxy{out_suffix}.png")
    _scatter_plot("frac_mean", "frac_ci", "tail_mean", "tail_ci", "Tail worst-cell CVaR", f"{dataset}_properness_tail{out_suffix}.png")
    _scatter_plot("frac_mean", "frac_ci", "wg_mean", "wg_ci", "Oracle worst-group acc", f"{dataset}_properness_wg{out_suffix}.png")


if __name__ == "__main__":
    main()
