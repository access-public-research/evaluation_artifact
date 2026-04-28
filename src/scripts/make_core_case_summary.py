from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"


def _row(df: pd.DataFrame, regime: str) -> pd.Series:
    sub = df.loc[df["regime"] == regime]
    if sub.empty:
        raise ValueError(f"Missing regime {regime}")
    return sub.iloc[0]


def _delta(summary_csv: Path, baseline_regime: str, target_regime: str, metric_col: str) -> float:
    df = pd.read_csv(summary_csv)
    return float(_row(df, target_regime)[metric_col] - _row(df, baseline_regime)[metric_col])


def _paired_effect(suite: str, metric: str) -> float:
    df = pd.read_csv(ART / "acceptance_stepup_paired_effects_20260328.csv")
    sub = df.loc[(df["suite_label"] == suite) & (df["metric"] == metric)]
    if sub.empty:
        raise ValueError(f"Missing paired effect for {suite}/{metric}")
    return float(sub.iloc[0]["mean"])


def _fmt_signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}"


def _fmt_cell(x: float | None, digits: int = 3, bold: bool = False) -> str:
    if x is None or pd.isna(x):
        return "---"
    text = _fmt_signed(float(x), digits=digits)
    return rf"\textbf{{{text}}}" if bold else text


def _build_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "case": "Camelyon17 ERM",
            "hazard": "persistent",
            "delta_acc": _delta(
                ART / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv",
                "erm",
                "erm_softclip_p95_a10_cam",
                "test_hosp_2_acc_mean",
            ),
            "delta_loss": _paired_effect("Camelyon17 ERM", "test_loss"),
            "delta_tail_rel": _paired_effect("Camelyon17 ERM", "tail_cvar"),
            "delta_ece": _paired_effect("Camelyon17 ERM", "ece"),
            "bold": False,
        }
    )

    rows.append(
        {
            "case": "CelebA ERM",
            "hazard": "transient",
            "delta_acc": _delta(
                ART / "celeba_test_wg_selected_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_oracle_wg_acc_mean",
            ),
            "delta_loss": _delta(
                ART / "celeba_test_wg_selected_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_oracle_wg_loss_mean",
            ),
            "delta_tail_rel": _delta(
                ART / "celeba_effect_size_erm_softclip_celeba_10s_20260325.csv",
                "erm",
                "erm_softclip_p95_a10",
                "tail_worst_cvar_mean",
            ),
            "delta_ece": _delta(
                ART / "celeba_selected_calibration_p95_20260327_summary.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_ece_mean",
            ),
            "bold": False,
        }
    )

    rows.append(
        {
            "case": "Camelyon17 Finetune",
            "hazard": r"persistent$^{\dagger}$",
            "delta_acc": _paired_effect("Camelyon17 Finetune", "test_acc"),
            "delta_loss": _paired_effect("Camelyon17 Finetune", "test_loss"),
            "delta_tail_rel": _paired_effect("Camelyon17 Finetune", "tail_cvar"),
            "delta_ece": _paired_effect("Camelyon17 Finetune", "ece"),
            "bold": True,
        }
    )

    rows.append(
        {
            "case": "CivilComments",
            "hazard": "persistent",
            "delta_acc": _paired_effect("CivilComments", "test_acc"),
            "delta_loss": _paired_effect("CivilComments", "test_loss"),
            "delta_tail_rel": _paired_effect("CivilComments", "wilds_wg_loss"),
            "delta_ece": _paired_effect("CivilComments", "ece"),
            "bold": False,
        }
    )

    acs = pd.read_csv(ART / "acs_income" / "phase3_selector_winsorized_p95.csv")
    acs_proxy = acs.loc[acs["selector"] == "Proxy-only"]
    rows.append(
        {
            "case": "ACSIncome",
            "hazard": "persistent",
            "delta_acc": None,
            "delta_loss": float(acs_proxy["delta_test_raw"].mean()),
            "delta_tail_rel": float(acs_proxy["delta_tail"].mean()),
            "delta_ece": None,
            "bold": False,
        }
    )

    return pd.DataFrame(rows)


def _write_tex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"  \toprule",
        r"  Case & Hazard & $\Delta$Acc$\uparrow$ & $\Delta$Loss$\downarrow$ & $\Delta$Tail/rel. diag.$\downarrow$ & $\Delta$ECE$\downarrow$ \\",
        r"  \midrule",
    ]
    for _, row in df.iterrows():
        bold = bool(row["bold"])
        lines.append(
            "  {case} & {hazard} & {acc} & {loss} & {tail} & {ece} \\\\".format(
                case=row["case"],
                hazard=row["hazard"],
                acc=_fmt_cell(row["delta_acc"], bold=bold),
                loss=_fmt_cell(row["delta_loss"], bold=bold),
                tail=_fmt_cell(row["delta_tail_rel"], bold=bold),
                ece=_fmt_cell(row["delta_ece"], bold=bold),
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = _build_rows()
    out_csv = ART / "core_case_summary_20260424.csv"
    out_tex = TABLES / "table_core_case_summary.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["bold"]).to_csv(out_csv, index=False)
    _write_tex(df, out_tex)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
