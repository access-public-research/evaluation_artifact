from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value):+.3f}"


def main() -> None:
    summary_csv = ROOT / "artifacts" / "metrics" / "camelyon17_bootstrap_selector_summary_extended_20260331.csv"
    out_tex = ROOT / "paper" / "neurips2026_selection_risk" / "tables" / "table_bootstrap_selector_comparison_extended.tex"

    summary = pd.read_csv(summary_csv).set_index("selection_policy")
    labels = {
        "baseline": "Baseline",
        "proxy_only": "Proxy-only",
        "val_loss_only": "Val-loss-only",
        "guardrail": "1.25x guardrail",
        "oracle_loss": "Oracle loss",
    }
    order = ["baseline", "proxy_only", "val_loss_only", "guardrail", "oracle_loss"]

    lines = [
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Selector & $\Delta$Held-out loss$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $\Delta$Tail CVaR$\downarrow$ & $\Delta$ECE$\downarrow$ \\",
        r"  \midrule",
    ]
    for policy in order:
        if policy not in summary.index:
            continue
        row = summary.loc[policy]
        lines.append(
            "  {label} & {dloss} & {dacc} & {dtail} & {dece} \\\\".format(
                label=labels.get(policy, policy),
                dloss=_fmt(row.get("delta_test_hosp_2_loss_vs_baseline")),
                dacc=_fmt(row.get("delta_test_hosp_2_acc_vs_baseline")),
                dtail=_fmt(row.get("delta_tail_worst_cvar_selected_vs_baseline")),
                dece=_fmt(row.get("delta_test_ece_vs_baseline")),
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
