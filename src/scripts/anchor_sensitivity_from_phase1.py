import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), 0.0
    mean = float(np.mean(x))
    ci = ci95_mean(x)
    return mean, ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_csv", required=True, help="selected rows csv (e.g., *_selected_vs_epoch30_rows_*.csv)")
    ap.add_argument("--pockets_csv", required=True, help="phase1 pockets csv")
    ap.add_argument("--families", default="teacher_difficulty,decoupled_proj")
    ap.add_argument("--metric", default="worst_cell_cvar")
    ap.add_argument("--selection_mode", default="selected_best_proxy")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows_df = pd.read_csv(args.rows_csv)
    pockets_df = pd.read_csv(args.pockets_csv)

    if "selection_mode" in rows_df.columns:
        rows_df = rows_df[rows_df["selection_mode"] == args.selection_mode].copy()
    merge_keys = ["regime", "seed", "epoch"]
    if "tag" in rows_df.columns:
        merge_keys.append("tag")
    rows_df = rows_df[merge_keys].drop_duplicates()

    pockets_df = pockets_df[pockets_df["split"] == "val"].copy()
    pockets_group_keys = ["regime", "seed", "family", "epoch"]
    if "tag" in pockets_df.columns:
        pockets_group_keys.append("tag")
    pockets_agg = (
        pockets_df.groupby(pockets_group_keys, as_index=False)
        .agg({args.metric: "mean"})
    )

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    rows_out = []
    for fam in families:
        fam_cols = [k for k in merge_keys if k in pockets_agg.columns] + [args.metric]
        fam_df = pockets_agg[pockets_agg["family"] == fam][fam_cols].copy()
        merged = rows_df.merge(fam_df, on=[k for k in merge_keys if k in fam_df.columns], how="left")
        merged["family"] = fam
        merged.rename(columns={args.metric: "tail_metric"}, inplace=True)
        rows_out.append(merged)

    merged_df = pd.concat(rows_out, ignore_index=True)
    summary_rows = []
    for (regime, family), sub in merged_df.groupby(["regime", "family"]):
        mean, ci = _mean_ci(sub["tail_metric"])
        summary_rows.append(
            {
                "regime": regime,
                "family": family,
                "n": int(sub["tail_metric"].notna().sum()),
                "tail_metric_mean": mean,
                "tail_metric_ci": ci,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["regime", "family"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    merged_df.to_csv(out_path.with_name(out_path.stem + "_rows.csv"), index=False)
    print(f"[anchor_sensitivity] wrote {out_path}")
    print(f"[anchor_sensitivity] wrote {out_path.with_name(out_path.stem + '_rows.csv')}")


if __name__ == "__main__":
    main()
