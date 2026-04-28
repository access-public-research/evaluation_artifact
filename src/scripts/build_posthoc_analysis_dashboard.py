import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loo_fold_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_fold_summary_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument(
        "--loo_conditionality_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_conditionality_camloo_foldcal_a10_10s_20260305_fold_features.csv",
    )
    ap.add_argument(
        "--weighting_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_objective_weighting_signflip_20260305_summary.csv",
    )
    ap.add_argument(
        "--grad_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_tail_core_grad_ratio_20260305_summary.csv",
    )
    ap.add_argument(
        "--ls_effect_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_labelsmooth_effect_size_n10_20260304.csv",
    )
    ap.add_argument(
        "--focal_effect_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_focal_effect_size_n10_20260304.csv",
    )
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/posthoc_dashboard_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    loo = _read(args.loo_fold_summary_csv)
    cond = _read(args.loo_conditionality_csv)
    wsum = _read(args.weighting_summary_csv)
    gsum = _read(args.grad_summary_csv)
    ls = _read(args.ls_effect_csv)
    fc = _read(args.focal_effect_csv)

    # Consolidated key table.
    key_rows: List[dict] = []
    # LOO aggregate
    n_tot = int(loo["n_seeds"].sum())
    succ = int(loo["seeds_late_top10_gt_early"].sum())
    key_rows.append(
        {
            "block": "loo_staging",
            "metric": "late_top10_gt_early_success",
            "value": f"{succ}/{n_tot} ({succ/max(n_tot,1):.3f})",
        }
    )
    key_rows.append(
        {
            "block": "loo_staging",
            "metric": "mean_diff_top10_late_minus_early",
            "value": f"{loo['mean_diff_top10_rec_late_minus_early'].mean():.6f}",
        }
    )

    for _, r in cond.sort_values("holdout_hospital").iterrows():
        key_rows.append(
            {
                "block": "loo_conditionality",
                "metric": f"h{int(r['holdout_hospital'])}_tail_ratio_99_90_over_95_50",
                "value": f"{float(r['teacher_tail_ratio_99_90_over_95_50']):.6f}",
            }
        )

    # Weighting/grad summaries (selected representative rows)
    for target in ["rcgdro_softclip_p95_a10_cam", "erm_labelsmooth_e10_cam", "erm_focal_g2_cam", "erm"]:
        sw = wsum[wsum["regime"] == target]
        sg = gsum[gsum["regime"] == target]
        if not sw.empty:
            key_rows.append(
                {
                    "block": "weighting",
                    "metric": f"{target}_tail_over_core_ratio",
                    "value": f"{float(sw['tail_over_core_ratio'].iloc[0]):.6f}",
                }
            )
        if not sg.empty:
            key_rows.append(
                {
                    "block": "grad_ratio",
                    "metric": f"{target}_tail_over_core_grad_ratio",
                    "value": f"{float(sg['tail_over_core_grad_ratio'].iloc[0]):.6f}",
                }
            )

    # LS/Focal effect deltas
    for df, name in [(ls, "labelsmooth"), (fc, "focal")]:
        for _, r in df[df["regime"] != "erm"].iterrows():
            key_rows.append(
                {
                    "block": f"{name}_effect",
                    "metric": f"{r['regime']}_tail_delta_vs_erm",
                    "value": f"{float(r['delta_tail_vs_erm']):.6f}",
                }
            )

    key = pd.DataFrame(key_rows)
    key_csv = out_prefix.with_name(out_prefix.name + "_key_metrics.csv")
    key.to_csv(key_csv, index=False)

    # Small dashboard figure.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # 1) LOO fold success by holdout hospital
    ax = axs[0, 0]
    tmp = loo.sort_values("holdout_hospital")
    x = tmp["holdout_hospital"].to_numpy(dtype=int)
    y = (tmp["seeds_late_top10_gt_early"] / tmp["n_seeds"]).to_numpy(dtype=float)
    ax.bar(x, y, color="tab:blue", alpha=0.85)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Holdout hospital")
    ax.set_ylabel("Success rate")
    ax.set_title("LOO: late>early hardness success by fold")

    # 2) Conditionality scatter
    ax = axs[0, 1]
    ax.scatter(
        cond["teacher_tail_ratio_99_90_over_95_50"],
        cond["stage_success_rate"],
        s=70,
        c=cond["holdout_hospital"],
        cmap="viridis",
    )
    for _, r in cond.iterrows():
        ax.annotate(f"h{int(r['holdout_hospital'])}", (r["teacher_tail_ratio_99_90_over_95_50"], r["stage_success_rate"]), fontsize=8)
    ax.set_xlabel("Teacher tail ratio (q99-q90)/(q95-q50)")
    ax.set_ylabel("Stage success rate")
    ax.set_title("Conditionality: harder-tailed folds stage more often")

    # 3) Weighting sign flip summary
    ax = axs[1, 0]
    keep = wsum[wsum["regime"].isin(["rcgdro_softclip_p95_a10_cam", "erm_labelsmooth_e10_cam", "erm_focal_g2_cam", "erm"])].copy()
    keep = keep.sort_values("regime")
    ax.bar(np.arange(keep.shape[0]), keep["tail_over_core_ratio"], color="tab:orange", alpha=0.85)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(np.arange(keep.shape[0]))
    ax.set_xticklabels(keep["regime"], rotation=30, ha="right")
    ax.set_ylabel("Tail/Core implicit-weight ratio")
    ax.set_title("Objective weighting summary")

    # 4) Tail deltas for LS/Focal
    ax = axs[1, 1]
    ls_sub = ls[ls["regime"] != "erm"][["regime", "delta_tail_vs_erm"]].copy()
    fc_sub = fc[fc["regime"] != "erm"][["regime", "delta_tail_vs_erm"]].copy()
    merged = pd.concat([ls_sub.assign(family="labelsmooth"), fc_sub.assign(family="focal")], ignore_index=True)
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    for fam, col in [("labelsmooth", "tab:green"), ("focal", "tab:red")]:
        s = merged[merged["family"] == fam]
        ax.scatter(np.arange(s.shape[0]), s["delta_tail_vs_erm"], label=fam, c=col, s=55)
    ax.set_xlabel("Regime index within family")
    ax.set_ylabel("Tail delta vs ERM")
    ax.set_title("LS/Focal tail direction (negative = tail improves)")
    ax.legend(loc="best", fontsize=8)

    png = out_prefix.with_name(out_prefix.name + "_panel.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print("[posthoc-dashboard] wrote:")
    print(f" - {key_csv}")
    print(f" - {png}")


if __name__ == "__main__":
    main()
