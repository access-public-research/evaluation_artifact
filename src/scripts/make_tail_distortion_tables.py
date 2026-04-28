import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def _regime_label(regime: str) -> str:
    r = str(regime).lower()
    if r == "rcgdro":
        return "Baseline"
    if "p95" in r:
        return "P95"
    if "p97" in r:
        return "P97"
    if "p99" in r:
        return "P99"
    return regime


def _dataset_label(dataset: str) -> str:
    d = str(dataset).lower()
    if d == "celeba":
        return "CelebA"
    if d == "waterbirds":
        return "Waterbirds"
    if d == "camelyon17":
        return "Camelyon17"
    return dataset


def _fmt_pm(mean: float, ci: float, nd: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(ci):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} $\\pm$ {ci:.{nd}f}"


def _fmt_float(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def _write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    reg_order = {"Baseline": 0, "P95": 1, "P97": 2, "P99": 3}
    ds_order = {"CelebA": 0, "Waterbirds": 1, "Camelyon17": 2}

    dfx = df.copy()
    dfx["dataset_label"] = dfx["dataset"].map(_dataset_label)
    dfx["regime_label"] = dfx["regime"].map(_regime_label)
    dfx["ord_ds"] = dfx["dataset_label"].map(ds_order).fillna(99)
    dfx["ord_rg"] = dfx["regime_label"].map(reg_order).fillna(99)
    dfx = dfx.sort_values(["ord_ds", "ord_rg"]).reset_index(drop=True)

    lines: List[str] = []
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Regime & FracClip & Mean Excess & Distortion Mass & Tail $\\Delta$ \\\\")
    lines.append("\\midrule")
    for _, r in dfx.iterrows():
        lines.append(
            " & ".join(
                [
                    str(r["dataset_label"]),
                    str(r["regime_label"]),
                    _fmt_pm(r["frac_clipped_mean"], r["frac_clipped_ci95"]),
                    _fmt_pm(r["mean_excess_mean"], r["mean_excess_ci95"]),
                    _fmt_pm(r["distortion_mass_mean"], r["distortion_mass_ci95"]),
                    _fmt_pm(r["tail_delta_vs_baseline_mean"], r["tail_delta_vs_baseline_ci95"]),
                ]
            )
            + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_corr_table(df: pd.DataFrame, out_path: Path) -> None:
    pred_map: Dict[str, str] = {
        "frac_clipped_selected": "FracClip",
        "distortion_mass_selected": "DistMass",
        "anchor_rho_cvar_clip": "$\\rho_{c^\\star}$",
    }
    ds_order = {"celeba": 0, "waterbirds": 1, "camelyon17": 2, "all": 3}
    pred_order = {"distortion_mass_selected": 0, "frac_clipped_selected": 1, "anchor_rho_cvar_clip": 2}

    dfx = df.copy()
    dfx = dfx[dfx["scope"].isin(["per_dataset", "pooled"])].copy()
    dfx["ord_ds"] = dfx["dataset"].map(ds_order).fillna(99)
    dfx["ord_pr"] = dfx["predictor"].map(pred_order).fillna(99)
    dfx = dfx.sort_values(["ord_ds", "ord_pr"]).reset_index(drop=True)

    lines: List[str] = []
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Scope & Predictor & Pearson & Spearman & $n$ \\\\")
    lines.append("\\midrule")
    for _, r in dfx.iterrows():
        scope = "Pooled" if r["dataset"] == "all" else _dataset_label(r["dataset"])
        pred = pred_map.get(str(r["predictor"]), str(r["predictor"]))
        lines.append(
            " & ".join(
                [
                    scope,
                    pred,
                    _fmt_float(r["pearson"], nd=3),
                    _fmt_float(r["spearman"], nd=3),
                    str(int(r["n"])) if np.isfinite(r["n"]) else "--",
                ]
            )
            + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csvs", required=True, help="Comma-separated *_tail_distortion_summary*.csv files.")
    ap.add_argument("--corr_csv", required=True)
    ap.add_argument("--out_summary_tex", required=True)
    ap.add_argument("--out_corr_tex", required=True)
    args = ap.parse_args()

    frames: List[pd.DataFrame] = []
    for p in [Path(x.strip()) for x in str(args.summary_csvs).split(",") if x.strip()]:
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    df_sum = pd.concat(frames, ignore_index=True)
    df_corr = pd.read_csv(args.corr_csv)

    out_summary = Path(args.out_summary_tex)
    out_corr = Path(args.out_corr_tex)
    ensure_dir(out_summary.parent)
    ensure_dir(out_corr.parent)

    _write_summary_table(df_sum, out_summary)
    _write_corr_table(df_corr, out_corr)
    print(f"[tail-distortion-tables] wrote {out_summary}")
    print(f"[tail-distortion-tables] wrote {out_corr}")


if __name__ == "__main__":
    main()

