from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "artifacts/metrics/camelyon17_gce_finetune_selector_summary_20260501.csv"
DEFAULT_ROWS = ROOT / "artifacts/metrics/camelyon17_gce_finetune_selector_rows_20260501.csv"
DEFAULT_OUT = ROOT / "paper/neurips2026_selection_risk/tables/table_camelyon17_gce_finetune_selector.tex"


def _fmt_delta(metric: str, mean: float, lo: float, hi: float) -> str:
    if metric in {"GCE validation proxy", "Validation accuracy"}:
        digits = 4
    elif metric == "Fixed held-out loss-tail":
        digits = 2
    else:
        digits = 3
    return f"{mean:+.{digits}f} [{lo:+.{digits}f},{hi:+.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(DEFAULT_ROWS))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rows_path = Path(args.rows)
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    rows_df = pd.read_csv(rows_path)
    expected_cols = {
        "seed",
        "selector",
        "epoch",
        "val_gce_proxy",
        "val_ce_loss",
        "val_acc",
        "heldout_loss",
        "heldout_acc",
        "heldout_ece",
        "heldout_brier",
        "fixed_heldout_loss_tail",
    }
    missing = expected_cols.difference(rows_df.columns)
    if missing:
        raise ValueError(f"Missing columns in rows artifact: {sorted(missing)}")

    df = pd.read_csv(args.summary)
    order = [
        "GCE validation proxy",
        "Validation CE loss",
        "Validation accuracy",
        "Held-out loss",
        "Fixed held-out loss-tail",
        "Held-out accuracy",
        "Held-out ECE",
        "Held-out Brier",
    ]
    df = df.set_index("metric").loc[order].reset_index()

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & $\Delta$ proxy-best $-$ val-loss-best & Count & Readout \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        metric = str(row["metric"])
        delta = _fmt_delta(metric, float(row["delta_mean"]), float(row["ci_low"]), float(row["ci_high"]))
        count = f"{int(row['count'])}/{int(row['n'])} {row['count_label']}"
        readout = str(row["interpretation"])
        lines.append(f"{metric} & {delta} & {count} & {readout} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
