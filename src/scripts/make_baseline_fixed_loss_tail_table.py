from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "artifacts/metrics/camelyon17_baseline_fixed_loss_tail_softclip_gce_summary_20260430.csv"
OUT_TEX = ROOT / "paper/neurips2026_selection_risk/tables/table_baseline_fixed_loss_tail.tex"


def _fmt_interval(mean: float, ci: float, digits: int) -> str:
    return f"{mean:+.{digits}f} [{mean - ci:+.{digits}f},{mean + ci:+.{digits}f}]"


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    rows = []
    for objective in ["SoftClip P95", "GCE q=0.7"]:
        sub = df[(df["objective"].eq(objective)) & (df["policy"].eq("proxy"))]
        if sub.empty:
            raise ValueError(f"Missing proxy row for {objective}")
        row = sub.iloc[0]
        rows.append(
            [
                objective,
                _fmt_interval(float(row["delta_fixed_tail_loss_mean"]), float(row["delta_fixed_tail_loss_ci"]), 3),
                _fmt_interval(float(row["delta_overall_loss_mean"]), float(row["delta_overall_loss_ci"]), 3),
                f"{int(row['delta_fixed_tail_loss_pos_count'])}/{int(row['n'])}",
            ]
        )

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Objective & $\Delta$Fixed loss-tail & $\Delta$Held-out loss & Harmful seeds \\",
        r"\midrule",
    ]
    for objective, fixed_tail, heldout_loss, seeds in rows:
        lines.append(f"{objective} & {fixed_tail} & {heldout_loss} & {seeds} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
