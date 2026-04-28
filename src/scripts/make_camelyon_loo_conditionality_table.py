import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aggregate_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_aggregate_sign_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument(
        "--fold_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_fold_summary_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument(
        "--switch_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_domain_switch_summary_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument(
        "--out_tex",
        default="paper/neurips2026_selection_risk/tables/table_camelyon_loo_conditionality.tex",
    )
    args = ap.parse_args()

    agg = pd.read_csv(args.aggregate_csv)
    fold = pd.read_csv(args.fold_csv)
    sw = pd.read_csv(args.switch_csv)

    top10 = agg[agg["metric"] == "top10_recover_late_gt_early"].iloc[0]
    success_min = int(fold["seeds_late_top10_gt_early"].min())
    success_max = int(fold["seeds_late_top10_gt_early"].max())
    switch_rate = float(sw["switch"].mean())

    lines = [
        "\\begin{tabular}{lcc}",
        "  \\toprule",
        "  Quantity & Value & Interpretation \\\\",
        "  \\midrule",
        f"  Pooled seed-fold late-harder rate & {int(top10['successes'])}/{int(top10['trials'])} ({float(top10['success_rate']):.3f}) & Exploratory pooled units \\\\",
        f"  One-sided pooled sign test $p$ & {float(top10['p_one_sided']):.3f} & Dependent within fold \\\\",
        f"  Per-fold success range & {success_min}/10 to {success_max}/10 & Heterogeneous across holdouts \\\\",
        f"  Mean hardness gap (late$-$early) & {float(top10['mean_diff_late_minus_early']):+.3f} & Late recoveries are harder on average \\\\",
        f"  Dominant-domain switch rate & {100.0 * switch_rate:.1f}\\% & Not a pure hospital-switch effect \\\\",
        "  \\bottomrule",
        "\\end{tabular}",
    ]
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
