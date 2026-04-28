import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _ci95(x: np.ndarray) -> float:
    return ci95_mean(np.asarray(x, dtype=np.float64))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache_glob",
        default="replication_rcg/artifacts/metrics/camelyon17_transition_canonical_20260305_cache_camelyon17_loo_h*.npz",
    )
    ap.add_argument("--alpha", type=float, default=0.1, help="Softclip slope alpha used for clipped examples.")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_activation_inflection_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    caches = sorted(Path(".").glob(str(args.cache_glob)))
    if not caches:
        raise FileNotFoundError(f"No caches found for glob: {args.cache_glob}")

    rows: List[dict] = []
    alpha = float(args.alpha)
    for cp in caches:
        z = np.load(cp)
        fold = cp.stem.split("_cache_")[-1]
        # Recover hospital id from fold suffix.
        h = int(fold.split("_h")[-1])
        seeds = z["seeds"].astype(int)
        c0 = z["correct_base"].astype(bool)
        c95 = z["correct_p95"].astype(bool)
        c97 = z["correct_p97"].astype(bool)
        c99 = z["correct_p99"].astype(bool)
        l95 = z["loss_p95"].astype(np.float32)
        l97 = z["loss_p97"].astype(np.float32)
        l99 = z["loss_p99"].astype(np.float32)
        t95 = z["tau_p95"].astype(np.float32)
        t97 = z["tau_p97"].astype(np.float32)
        t99 = z["tau_p99"].astype(np.float32)

        for si, seed in enumerate(seeds):
            state = np.full(c0.shape[1], "fail_p99", dtype=object)
            state[c0[si]] = "base_ok"
            state[(~c0[si]) & c95[si]] = "rec_p95"
            state[(~c95[si]) & c97[si]] = "rec_p97"
            state[(~c97[si]) & c99[si]] = "rec_p99"

            active = {
                "base": np.ones((c0.shape[1],), dtype=np.float32),
                "p95": (l95[si] <= t95[si]).astype(np.float32),
                "p97": (l97[si] <= t97[si]).astype(np.float32),
                "p99": (l99[si] <= t99[si]).astype(np.float32),
            }
            weight = {
                "base": np.ones((c0.shape[1],), dtype=np.float32),
                "p95": np.where(active["p95"] > 0.5, 1.0, alpha).astype(np.float32),
                "p97": np.where(active["p97"] > 0.5, 1.0, alpha).astype(np.float32),
                "p99": np.where(active["p99"] > 0.5, 1.0, alpha).astype(np.float32),
            }

            for st in ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]:
                m = state == st
                if not np.any(m):
                    continue
                n = int(m.sum())
                for reg in ["base", "p95", "p97", "p99"]:
                    rows.append(
                        {
                            "fold": fold,
                            "holdout_hospital": h,
                            "seed": int(seed),
                            "state": st,
                            "regime": reg,
                            "n": n,
                            "active_frac": float(active[reg][m].mean()),
                            "mean_weight_ratio": float(weight[reg][m].mean()),
                        }
                    )

    df = pd.DataFrame(rows)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows_csv = out_prefix.with_name(out_prefix.name + "_rows.csv")
    df.to_csv(rows_csv, index=False)

    g = (
        df.groupby(["fold", "holdout_hospital", "state", "regime"])
        .agg(
            n_seed=("seed", "nunique"),
            active_frac_mean=("active_frac", "mean"),
            mean_weight_ratio=("mean_weight_ratio", "mean"),
        )
        .reset_index()
    )
    # Add CI across seeds by fold/state/regime.
    cis = (
        df.groupby(["fold", "holdout_hospital", "state", "regime"])["active_frac"]
        .apply(lambda s: _ci95(s.to_numpy(dtype=np.float64)))
        .reset_index(name="active_frac_ci95")
    )
    g = g.merge(cis, on=["fold", "holdout_hospital", "state", "regime"], how="left")
    fold_csv = out_prefix.with_name(out_prefix.name + "_fold_summary.csv")
    g.to_csv(fold_csv, index=False)

    pooled = (
        df.groupby(["state", "regime"])
        .agg(
            n=("seed", "count"),
            active_frac_mean=("active_frac", "mean"),
            mean_weight_ratio=("mean_weight_ratio", "mean"),
        )
        .reset_index()
    )
    pooled_ci = (
        df.groupby(["state", "regime"])["active_frac"]
        .apply(lambda s: _ci95(s.to_numpy(dtype=np.float64)))
        .reset_index(name="active_frac_ci95")
    )
    pooled = pooled.merge(pooled_ci, on=["state", "regime"], how="left")
    pooled_csv = out_prefix.with_name(out_prefix.name + "_pooled_summary.csv")
    pooled.to_csv(pooled_csv, index=False)

    # Inflection deltas per fold/seed/state.
    piv = df.pivot_table(index=["fold", "holdout_hospital", "seed", "state"], columns="regime", values="active_frac").reset_index()
    piv["delta_p95_to_p97"] = piv["p97"] - piv["p95"]
    piv["delta_p97_to_p99"] = piv["p99"] - piv["p97"]
    inflect_csv = out_prefix.with_name(out_prefix.name + "_inflection_rows.csv")
    piv.to_csv(inflect_csv, index=False)

    # Plot pooled activation curves.
    order_states = ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]
    order_reg = ["base", "p95", "p97", "p99"]
    x = np.arange(len(order_reg))
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axs = axs.ravel()
    for i, st in enumerate(order_states):
        ax = axs[i]
        s = pooled[pooled["state"] == st].set_index("regime")
        y = [float(s.loc[r, "active_frac_mean"]) if r in s.index else np.nan for r in order_reg]
        e = [float(s.loc[r, "active_frac_ci95"]) if r in s.index else np.nan for r in order_reg]
        ax.plot(x, y, marker="o")
        ax.fill_between(x, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e), alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(order_reg)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(st)
        ax.set_ylabel("Active fraction")
    if len(order_states) < len(axs):
        axs[-1].axis("off")
    png = out_prefix.with_name(out_prefix.name + "_pooled_curves.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print("[activation-inflection] wrote:")
    print(f" - {rows_csv}")
    print(f" - {fold_csv}")
    print(f" - {pooled_csv}")
    print(f" - {inflect_csv}")
    print(f" - {png}")


if __name__ == "__main__":
    main()
