import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _ci95(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    return ci95_mean(v.to_numpy(dtype=np.float64))


def _summary(df_rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols: List[str] = [
        "proxy_selected",
        "tail_worst_cvar",
        "oracle_wg_acc",
        "val_overall_acc",
        "frac_clipped_val",
    ]
    groups = []
    for (selection_mode, regime), g in df_rows.groupby(["selection_mode", "regime"], dropna=False):
        row = {
            "selection_mode": selection_mode,
            "regime": regime,
            "n": int(g["seed"].nunique()),
        }
        for m in metric_cols:
            x = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(x.mean()) if x.notna().any() else np.nan
            row[f"{m}_sd"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else 0.0
            row[f"{m}_ci95"] = _ci95(x)
        groups.append(row)
    out = pd.DataFrame(groups).sort_values(["selection_mode", "regime"]).reset_index(drop=True)
    return out


def _choose_proxy_selected(out: pd.DataFrame) -> pd.Series:
    regime = out["regime"].astype(str).str.lower()
    if "proxy_worst_loss" not in out.columns:
        raise KeyError("proxy_worst_loss column is required")
    proxy_unclip = pd.to_numeric(out["proxy_worst_loss"], errors="coerce")
    proxy_clip = (
        pd.to_numeric(out["proxy_worst_loss_clip"], errors="coerce")
        if "proxy_worst_loss_clip" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=np.float64)
    )
    use_clip = regime.str.contains("clip")
    selected = proxy_unclip.copy()
    valid_clip = proxy_clip.notna()
    selected.loc[use_clip & valid_clip] = proxy_clip.loc[use_clip & valid_clip]
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_summary_csv", required=True)
    ap.add_argument("--fixed_summary_csv", required=True)
    ap.add_argument("--phase0_csv", required=True)
    ap.add_argument("--out_rows_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--fixed_label", default="fixed_epoch_30")
    args = ap.parse_args()

    df_sel = pd.read_csv(args.selected_summary_csv)
    df_fix = pd.read_csv(args.fixed_summary_csv)
    df_p0 = pd.read_csv(args.phase0_csv)

    # Tag is unique per (regime, seed, epoch) and needed by strict downstream diagnostics.
    tag_map = df_p0[["regime", "seed", "epoch", "tag"]].drop_duplicates()

    def _build_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        keep_cols = [
            "regime",
            "seed",
            "epoch",
            "proxy_worst_loss",
            "proxy_worst_loss_clip",
            "proxy_worst_acc",
            "tail_worst_cvar",
            "oracle_wg_acc",
            "val_overall_acc",
            "val_overall_loss",
            "frac_clipped_train",
            "frac_clipped_val",
            "clip_alpha_active",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        out = df[keep_cols].copy()
        out.insert(0, "selection_mode", mode)
        out["proxy_selected"] = _choose_proxy_selected(out)
        out = out.merge(tag_map, on=["regime", "seed", "epoch"], how="left")
        if out["tag"].isna().any():
            missing = out[out["tag"].isna()][["regime", "seed", "epoch"]].drop_duplicates()
            raise ValueError(f"Missing tag mapping for rows:\n{missing.to_string(index=False)}")
        return out

    rows_sel = _build_rows(df_sel, "selected_best_proxy")
    rows_fix = _build_rows(df_fix, args.fixed_label)
    rows = pd.concat([rows_sel, rows_fix], ignore_index=True)
    rows = rows[
        [
            "selection_mode",
            "regime",
            "seed",
            "epoch",
            "tag",
            "proxy_worst_loss",
            "proxy_worst_loss_clip",
            "proxy_worst_acc",
            "tail_worst_cvar",
            "oracle_wg_acc",
            "val_overall_acc",
            "val_overall_loss",
            "frac_clipped_train",
            "frac_clipped_val",
            "clip_alpha_active",
            "proxy_selected",
        ]
    ]

    out_rows = Path(args.out_rows_csv)
    out_summary = Path(args.out_summary_csv)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_rows, index=False)
    _summary(rows).to_csv(out_summary, index=False)
    print(f"[selected-vs-fixed] wrote {out_rows}")
    print(f"[selected-vs-fixed] wrote {out_summary}")


if __name__ == "__main__":
    main()
