from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAIRED_CSV = ROOT / "artifacts" / "metrics" / "acceptance_stepup_paired_effects_20260328.csv"
OUT_TEX = ROOT / "paper" / "neurips2026_selection_risk" / "tables" / "table_core_case_uncertainty.tex"


def _fmt_cell(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def main() -> None:
    df = pd.read_csv(PAIRED_CSV)

    suite_specs = [
        ("Camelyon17 ERM", "test_loss", "tail_cvar", "ECE"),
        ("Camelyon17 Finetune", "test_loss", "tail_cvar", "ECE"),
        ("CivilComments", "test_loss", "wilds_wg_loss", "ECE"),
    ]

    metric_label = {
        "test_loss": r"$\Delta$Held-out loss",
        "tail_cvar": r"$\Delta$Tail / reliability",
        "wilds_wg_loss": r"$\Delta$Tail / reliability",
        "ece": r"$\Delta$ECE",
    }

    def pick(suite: str, metric: str) -> pd.Series:
        row = df[(df["suite_label"] == suite) & (df["metric"] == metric)]
        if row.empty:
            raise ValueError(f"Missing {metric} for {suite} in {PAIRED_CSV}")
        return row.iloc[0]

    lines = [
        r"\begin{tabular}{lccc}",
        r"  \toprule",
        r"  Case & $\Delta$Held-out loss & $\Delta$Tail / reliability & $\Delta$ECE \\",
        r"  \midrule",
    ]

    for suite, loss_metric, rel_metric, ece_metric in suite_specs:
        loss = pick(suite, loss_metric)
        rel = pick(suite, rel_metric)
        ece = pick(suite, ece_metric.lower())
        rel_digits = 3 if rel_metric == "wilds_wg_loss" else 2
        lines.append(
            "  {suite} & {loss} & {rel} & {ece} \\\\".format(
                suite=suite,
                loss=_fmt_cell(loss["mean"], loss["ci_low"], loss["ci_high"], digits=3),
                rel=_fmt_cell(rel["mean"], rel["ci_low"], rel["ci_high"], digits=rel_digits),
                ece=_fmt_cell(ece["mean"], ece["ci_low"], ece["ci_high"], digits=3),
            )
        )

    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
