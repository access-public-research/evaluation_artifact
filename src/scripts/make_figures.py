import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import load_config
from ..metrics.proxy_eval import aggregate_proxy_metrics, snr_between_total_multi
from ..utils.io import ensure_dir
from ..utils.stats import ci95_mean


@dataclass
class RunRef:
    regime: str
    seed: int
    tag: str
    run_dir: Path


def _mean_ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    mean = float(x.mean())
    if x.size == 1:
        return mean, 0.0
    std = float(x.std(ddof=1))
    ci = ci95_mean(x)
    return mean, float(ci)


def _discover_run_refs(runs_root: Path, dataset: str, regimes: Iterable[str]) -> List[RunRef]:
    out: List[RunRef] = []
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
                out.append(RunRef(regime=regime, seed=seed, tag=tag_dir.name, run_dir=tag_dir))
    return out


def _corr_curve(df: pd.DataFrame, regime: str, score_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for ep, sub in df[df["regime"] == regime].groupby("epoch"):
        x = sub[score_col].to_numpy()
        y = sub[target_col].to_numpy()
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            rho = np.nan
        else:
            rho = float(spearmanr(x, y).correlation)
        rows.append({"regime": regime, "epoch": int(ep), "rho": rho})
    return pd.DataFrame(rows).sort_values("epoch")


def _plot_corr(df_metrics: pd.DataFrame, regime: str, out_path: Path):
    curve = _corr_curve(df_metrics, regime, "val_proxy_worst_acc", "test_worst_group_acc")
    plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.plot(curve["epoch"], curve["rho"], linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.ylim(-1.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Spearman corr (across seeds)")
    plt.title(f"Critic informativeness vs epoch ({regime})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _selection_with_oracle(df_sel: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "regime", "seed", "tag"]
    oracle = (
        df_sel.groupby(keys)[["oracle_best_wg_acc", "oracle_best_overall_acc"]]
        .first()
        .reset_index()
    )
    oracle = oracle.rename(
        columns={
            "oracle_best_wg_acc": "test_worst_group_acc",
            "oracle_best_overall_acc": "test_overall_acc",
        }
    )
    oracle["method"] = "oracle"
    # Fill any missing columns with NaN for safe alignment.
    for col in df_sel.columns:
        if col not in oracle.columns:
            oracle[col] = np.nan
    return pd.concat([df_sel, oracle[df_sel.columns]], ignore_index=True)


def _bar_with_ci(df: pd.DataFrame, regime: str, methods: List[str], value_col: str, title: str, out_path: Path):
    sub = df[(df["regime"] == regime) & (df["method"].isin(methods))]
    means = []
    cis = []
    for m in methods:
        vals = sub[sub["method"] == m][value_col].to_numpy()
        mean, ci = _mean_ci(vals)
        means.append(mean)
        cis.append(ci)

    x = np.arange(len(methods))
    plt.figure(figsize=(8.0, 4.8), dpi=160)
    plt.bar(x, means, yerr=cis, capsize=4)
    plt.xticks(x, methods, rotation=20)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_flattening(df_metrics: pd.DataFrame, out_path: Path):
    agg = (
        df_metrics.groupby(["regime", "epoch"])[["val_proxy_between_loss", "val_dec_between_loss"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = [str(a) if (b is None or str(b) == "") else f"{a}_{b}" for a, b in agg.columns]

    regimes = sorted(df_metrics["regime"].unique())
    plt.figure(figsize=(9.0, 4.8), dpi=160)
    for regime in regimes:
        sub = agg[agg["regime"] == regime].sort_values("epoch")
        x = sub["epoch"].to_numpy()
        y = sub["val_proxy_between_loss_mean"].to_numpy()
        ystd = sub["val_proxy_between_loss_std"].to_numpy()
        plt.plot(x, y, linewidth=2, label=f"{regime} (proxy)")
        plt.fill_between(x, y - ystd, y + ystd, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("Between/total variance ratio (loss)")
    plt.title("Flattening on proxy partitions")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _stratified_sample_indices(g: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64)
    n = int(n)
    uniq, counts = np.unique(g, return_counts=True)
    props = counts / counts.sum()
    raw = props * n
    base = np.floor(raw).astype(int)
    rem = n - int(base.sum())
    if rem > 0:
        order = np.argsort(raw - base)[::-1]
        for i in range(rem):
            base[order[i % len(order)]] += 1

    idx_all = np.arange(len(g), dtype=np.int64)
    out = []
    for gid, k in zip(uniq, base):
        if k <= 0:
            continue
        pool = idx_all[g == gid]
        k = min(k, int(pool.size))
        out.append(rng.choice(pool, size=k, replace=False))
    sel = np.concatenate(out) if out else np.array([], dtype=np.int64)
    rng.shuffle(sel)
    return sel


def _subset_val_metrics(
    val_loss: np.ndarray,
    val_correct: np.ndarray,
    idx: np.ndarray,
    proxy_parts: List[np.ndarray],
    proxy_K: int,
    snr_trials: int,
    snr_seed: int,
):
    idx = np.asarray(idx, dtype=np.int64)
    parts_sub = [p[idx] for p in proxy_parts]
    E = int(val_loss.shape[0])
    overall_acc = np.zeros(E, dtype=np.float64)
    overall_loss = np.zeros(E, dtype=np.float64)
    proxy_worst_acc = np.zeros(E, dtype=np.float64)
    proxy_worst_loss = np.zeros(E, dtype=np.float64)
    snr = np.zeros(E, dtype=np.float64)
    for e in range(E):
        losses_e = val_loss[e, idx].astype(np.float64, copy=False)
        correct_e = val_correct[e, idx].astype(np.float64, copy=False)
        overall_acc[e] = float(correct_e.mean())
        overall_loss[e] = float(losses_e.mean())
        wacc, wloss, _bt_l, _bt_c = aggregate_proxy_metrics(losses_e, correct_e, parts_sub, proxy_K)
        proxy_worst_acc[e] = wacc
        proxy_worst_loss[e] = wloss
        snr[e] = snr_between_total_multi(correct_e, parts_sub, proxy_K, null_trials=snr_trials, seed=snr_seed + e)
    return overall_acc, overall_loss, proxy_worst_acc, proxy_worst_loss, snr


def _phase_diagram(
    cfg,
    dataset_name: str,
    backbone: str,
    df_metrics: pd.DataFrame,
    runs_root: Path,
    out_path: Path,
    tag_filter: str = "",
    exclude_tag_filter: str = "",
):
    regimes = ["erm", "rcgdro"]
    run_refs = _discover_run_refs(runs_root, dataset_name, regimes)
    if tag_filter:
        run_refs = [r for r in run_refs if tag_filter in str(r.tag)]
    if exclude_tag_filter:
        run_refs = [r for r in run_refs if exclude_tag_filter not in str(r.tag)]
    if not run_refs:
        return

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"

    g_val = np.load(feat_dir / "g_val_skew.npy")
    proxy_cfg = cfg["partitions"]["proxy"]
    proxy_K = int(proxy_cfg["num_cells"])
    part_base = artifacts_dir / "partitions" / f"{dataset_name}_{backbone}"
    part_version = cfg.get("partitions", {}).get("version")
    part_version_dir = part_base / str(part_version) if part_version else None
    parts_cache = {}

    sizes = [int(x) for x in cfg.get("analysis", {}).get("val_size_sweep", [200, 400, 800, 1200, 1600, 2000])]
    resamples = int(cfg.get("analysis", {}).get("val_size_resamples", 5))
    snr_trials = int(cfg.get("analysis", {}).get("snr_null_trials_sweep", 10))
    tail_lambda = float(cfg["selectors"].get("tailmoderated_lambda", 0.5))
    snr_threshold = float(cfg["selectors"].get("router", {}).get("snr_threshold", 1.5))

    perf_rows = []
    snr_rows = []
    for run in run_refs:
        run_dir = run.run_dir
        val_loss = np.load(run_dir / "val_loss_by_epoch.npy")
        val_correct = np.load(run_dir / "val_correct_by_epoch.npy")

        run_cfg = json.loads((run_dir / "config.json").read_text())
        part_root_cfg = run_cfg.get("partition_root")
        if part_root_cfg:
            part_root = Path(part_root_cfg)
        elif part_base.exists():
            part_root = part_base
        elif part_version_dir and part_version_dir.exists():
            part_root = part_version_dir
        else:
            raise FileNotFoundError(f"Could not resolve partitions for {run_dir}")

        proxy_key = str(part_root)
        if proxy_key not in parts_cache:
            proxy_dir = part_root / "proxy" / "val_skew"
            parts_cache[proxy_key] = [
                np.load(proxy_dir / f"hash_m{m:02d}_K{proxy_K}.npy")
                for m in range(int(proxy_cfg["num_partitions"]))
            ]
        proxy_parts = parts_cache[proxy_key]

        df_run = (
            df_metrics[
                (df_metrics["regime"] == run.regime)
                & (df_metrics["seed"] == run.seed)
                & (df_metrics["tag"] == run.tag)
            ]
            .sort_values("epoch")
            .reset_index(drop=True)
        )
        if df_run.empty:
            continue
        test_wg = df_run["test_worst_group_acc"].to_numpy()
        test_overall = df_run["test_overall_acc"].to_numpy()

        for n in sizes:
            n_eff = min(int(n), int(g_val.size))
            for r in range(resamples):
                rng = np.random.default_rng(int(run.seed) * 10000 + int(n_eff) * 100 + r)
                idx = _stratified_sample_indices(g_val, n_eff, rng)
                overall_acc, overall_loss, proxy_acc, proxy_loss, snr = _subset_val_metrics(
                    val_loss, val_correct, idx, proxy_parts, proxy_K, snr_trials=snr_trials, snr_seed=int(run.seed) * 1000 + r
                )

                tail_score = overall_acc - tail_lambda * proxy_loss
                idx_proxy = int(np.nanargmax(proxy_acc))
                idx_tail = int(np.nanargmax(tail_score))
                idx_router = idx_proxy if snr[idx_proxy] >= snr_threshold else idx_tail

                snr_rows.append(
                    {
                        "regime": run.regime,
                        "seed": run.seed,
                        "val_size": n_eff,
                        "resample": r,
                        "snr_mean": float(np.nanmean(snr)),
                        "snr_proxy": float(snr[idx_proxy]),
                    }
                )

                for method, idx_sel in [
                    ("proxy_acc", idx_proxy),
                    ("tailmoderated", idx_tail),
                    ("router", idx_router),
                ]:
                    perf_rows.append(
                        {
                            "regime": run.regime,
                            "seed": run.seed,
                            "val_size": n_eff,
                            "resample": r,
                            "method": method,
                            "selected_epoch": int(idx_sel + 1),
                            "test_worst_group_acc": float(test_wg[idx_sel]),
                            "test_overall_acc": float(test_overall[idx_sel]),
                            "snr_selected": float(snr[idx_sel]),
                        }
                    )

    if not perf_rows or not snr_rows:
        return

    df_perf = pd.DataFrame(perf_rows)
    df_snr = pd.DataFrame(snr_rows)

    snr_agg = df_snr.groupby(["regime", "val_size"])["snr_mean"].mean().reset_index()
    perf_agg = (
        df_perf.groupby(["regime", "val_size", "method"])["test_worst_group_acc"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(11.0, 4.8), dpi=160)
    ax1 = plt.subplot(1, 2, 1)
    for regime, sub in snr_agg.groupby("regime"):
        sub = sub.sort_values("val_size")
        ax1.plot(sub["val_size"], sub["snr_mean"], marker="o", linewidth=2, label=regime)
    ax1.set_xlabel("Validation size")
    ax1.set_ylabel("Correctness SNR (mean)")
    ax1.set_title("Observability rises with val size")
    ax1.legend(frameon=False)

    ax2 = plt.subplot(1, 2, 2)
    sub_perf = perf_agg[perf_agg["regime"] == "rcgdro"]
    for method, sub in sub_perf.groupby("method"):
        sub = sub.sort_values("val_size")
        ax2.plot(sub["val_size"], sub["test_worst_group_acc"], marker="o", linewidth=2, label=method)
    ax2.set_xlabel("Validation size")
    ax2.set_ylabel("Test worst-group acc")
    ax2.set_title("Phase transition in selector reliability (rcgdro)")
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--exclude_tag_filter", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]

    metrics_dir = Path(cfg["project"]["artifacts_dir"]) / "metrics"
    run_metrics_path = metrics_dir / f"{dataset_name}_{backbone}_run_metrics.csv"
    sel_path = metrics_dir / f"{dataset_name}_{backbone}_selection_summary.csv"
    if not run_metrics_path.exists() or not sel_path.exists():
        raise FileNotFoundError(
            f"Missing metrics files. Run eval first:\n"
            f"  python -m src.scripts.eval_runs --config {args.config} --dataset {args.dataset}"
        )

    df_metrics = pd.read_csv(run_metrics_path)
    df_sel = pd.read_csv(sel_path)
    if args.tag_filter:
        mask_m = df_metrics["tag"].astype(str).str.contains(str(args.tag_filter))
        mask_s = df_sel["tag"].astype(str).str.contains(str(args.tag_filter))
        df_metrics = df_metrics[mask_m].reset_index(drop=True)
        df_sel = df_sel[mask_s].reset_index(drop=True)
        if df_metrics.empty or df_sel.empty:
            raise ValueError(f"No rows matched tag_filter='{args.tag_filter}'.")
    if args.exclude_tag_filter:
        mask_m = ~df_metrics["tag"].astype(str).str.contains(str(args.exclude_tag_filter))
        mask_s = ~df_sel["tag"].astype(str).str.contains(str(args.exclude_tag_filter))
        df_metrics = df_metrics[mask_m].reset_index(drop=True)
        df_sel = df_sel[mask_s].reset_index(drop=True)
        if df_metrics.empty or df_sel.empty:
            raise ValueError(f"All rows excluded by exclude_tag_filter='{args.exclude_tag_filter}'.")

    fig_root = Path(cfg["project"]["figures_dir"]) / f"{dataset_name}_{backbone}"
    if args.tag_filter:
        safe_tag = str(args.tag_filter).replace(" ", "_")
        fig_root = fig_root.parent / f"{fig_root.name}_tag-{safe_tag}"
    ensure_dir(fig_root)

    # 1) Selection hardness (with oracle upper bound).
    df_sel_oracle = _selection_with_oracle(df_sel)
    methods_hard = ["overall", "loss", "hybrid", "proxy_acc", "tailmoderated", "router", "oracle"]
    for regime in sorted(df_sel_oracle["regime"].unique()):
        out = fig_root / f"selection_hardness_{regime}.png"
        _bar_with_ci(
            df_sel_oracle,
            regime=regime,
            methods=methods_hard,
            value_col="test_worst_group_acc",
            title=f"Selection hardness ({regime})",
            out_path=out,
        )

    # 2/3) Critic informativeness.
    _plot_corr(df_metrics, regime="erm", out_path=fig_root / "critic_corr_erm.png")
    _plot_corr(df_metrics, regime="rcgdro", out_path=fig_root / "critic_corr_rcgdro.png")

    # 4) Mechanism: flattening.
    _plot_flattening(df_metrics, out_path=fig_root / "flattening_proxy_loss.png")

    # 5) Router win (focus on rcgdro).
    methods_router = ["overall", "hybrid", "proxy_acc", "tailmoderated", "router", "oracle"]
    _bar_with_ci(
        df_sel_oracle,
        regime="rcgdro",
        methods=methods_router,
        value_col="test_worst_group_acc",
        title="Router vs portfolio (rcgdro)",
        out_path=fig_root / "router_win_rcgdro.png",
    )

    # 6) Phase diagram (most meaningful on CelebA).
    if dataset_name == "celeba":
        runs_root = Path(cfg["project"]["runs_dir"])
        _phase_diagram(
            cfg,
            dataset_name=dataset_name,
            backbone=backbone,
            df_metrics=df_metrics,
            runs_root=runs_root,
            out_path=fig_root / "phase_diagram_val_size.png",
            tag_filter=str(args.tag_filter),
            exclude_tag_filter=str(args.exclude_tag_filter),
        )

    print(f"[make_figures] wrote figures under {fig_root}")


if __name__ == "__main__":
    main()
