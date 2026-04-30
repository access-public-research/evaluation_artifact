from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "artifacts/metrics/camelyon_modern_selector_sensitivity_summary_20260430.csv"
OUT_TEX = ROOT / "paper/neurips2026_selection_risk/tables/table_camelyon_selector_averaging_sensitivity.tex"


POLICY_LABELS = {
    "proxy_best": "Proxy-best",
    "val_acc_best": "Val-acc-best",
    "avg_last5": "Last-5 avg.",
    "avg_val_loss5": "Val-loss-neigh. avg.",
}


def _fmt_ci(row: pd.Series, digits: int) -> str:
    mean = float(row["mean"])
    lo = float(row["ci_low"])
    hi = float(row["ci_high"])
    return f"{mean:+.{digits}f} [{lo:+.{digits}f},{hi:+.{digits}f}]"


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    rows = []
    for objective in ["SoftClip P95", "GCE q=0.7"]:
        for policy in ["proxy_best", "val_acc_best", "avg_last5", "avg_val_loss5"]:
            contrast = f"{policy} - val_loss_best"
            sub = df[(df["objective"].eq(objective)) & (df["contrast"].eq(contrast))].set_index("metric")
            if sub.empty:
                raise ValueError(f"Missing rows for {objective} / {contrast}")
            loss = sub.loc["test_loss"]
            ece = sub.loc["test_ece"]
            tail = sub.loc["teacher_tail_cvar"]
            rows.append(
                [
                    objective,
                    POLICY_LABELS[policy],
                    _fmt_ci(loss, 3),
                    _fmt_ci(ece, 4),
                    _fmt_ci(tail, 2),
                    f"{int(loss['harm_count'])}/{int(loss['n'])},"
                    f"{int(ece['harm_count'])}/{int(ece['n'])},"
                    f"{int(tail['harm_count'])}/{int(tail['n'])}",
                ]
            )

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Objective & Policy $-$ val-loss-best & $\Delta$Loss & $\Delta$ECE & $\Delta$Tail & Harm counts \\",
        r"\midrule",
    ]
    current = None
    for objective, policy, loss, ece, tail, counts in rows:
        obj_cell = objective if objective != current else ""
        current = objective
        lines.append(f"{obj_cell} & {policy} & {loss} & {ece} & {tail} & {counts} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
