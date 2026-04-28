import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"


def _fmt(x: float, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


def _fmt_signed(x: float, nd: int = 3) -> str:
    return f"{float(x):+.{nd}f}"


def _proxy_value(row: pd.Series) -> float:
    v = row.get("proxy_worst_loss_clip_mean")
    if pd.notna(v):
        return float(v)
    return float(row["proxy_worst_loss_mean"])


def _load_delta_rows(
    suite: str,
    baseline_regime: str,
    regimes: list[str],
    selected_domain_csv: Path,
    selected_effect_csv: Path,
    fixed_domain_csv: Path,
    fixed_effect_csv: Path,
) -> list[dict]:
    sel_dom = pd.read_csv(selected_domain_csv)
    sel_eff = pd.read_csv(selected_effect_csv)
    fix_dom = pd.read_csv(fixed_domain_csv)
    fix_eff = pd.read_csv(fixed_effect_csv)

    sel_base_dom = sel_dom.loc[sel_dom["regime"] == baseline_regime].iloc[0]
    sel_base_eff = sel_eff.loc[sel_eff["regime"] == baseline_regime].iloc[0]
    fix_base_dom = fix_dom.loc[fix_dom["regime"] == baseline_regime].iloc[0]
    fix_base_eff = fix_eff.loc[fix_eff["regime"] == baseline_regime].iloc[0]

    rows = []
    for regime in regimes:
        sel_dom_row = sel_dom.loc[sel_dom["regime"] == regime].iloc[0]
        sel_eff_row = sel_eff.loc[sel_eff["regime"] == regime].iloc[0]
        fix_dom_row = fix_dom.loc[fix_dom["regime"] == regime].iloc[0]
        fix_eff_row = fix_eff.loc[fix_eff["regime"] == regime].iloc[0]

        rows.append(
            {
                "suite": suite,
                "regime": regime,
                "level": regime.split("_p")[1].split("_")[0],
                "frac_clipped_val_mean": float(sel_eff_row.get("frac_clipped_val_mean", float("nan"))),
                "selected_proxy_delta": _proxy_value(sel_eff_row) - _proxy_value(sel_base_eff),
                "selected_acc_delta": float(sel_dom_row["test_acc_mean"]) - float(sel_base_dom["test_acc_mean"]),
                "selected_loss_delta": float(sel_dom_row["test_loss_mean"]) - float(sel_base_dom["test_loss_mean"]),
                "selected_tail_delta": float(sel_eff_row["tail_worst_cvar_mean"]) - float(sel_base_eff["tail_worst_cvar_mean"]),
                "fixed_acc_delta": float(fix_dom_row["test_acc_mean"]) - float(fix_base_dom["test_acc_mean"]),
                "fixed_loss_delta": float(fix_dom_row["test_loss_mean"]) - float(fix_base_dom["test_loss_mean"]),
                "fixed_tail_delta": float(fix_eff_row["tail_worst_cvar_mean"]) - float(fix_base_eff["tail_worst_cvar_mean"]),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--finetune_suffix",
        default="finetune_cam_scivalid10s_20260326",
        help="Suite suffix for the finetune selected/fixed summaries.",
    )
    ap.add_argument(
        "--out_csv",
        default=str(METRICS / "camelyon_dose_response_summary_v32_20260422.csv"),
    )
    ap.add_argument(
        "--out_tex",
        default=str(TABLES / "table_camelyon_dose_response_v32_20260422.tex"),
    )
    args = ap.parse_args()

    rows = []
    rows.extend(
        _load_delta_rows(
            suite="Camelyon17 ERM",
            baseline_regime="erm",
            regimes=["erm_softclip_p95_a10_cam", "erm_softclip_p97_a10_cam", "erm_softclip_p99_a10_cam"],
            selected_domain_csv=METRICS / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv",
            selected_effect_csv=METRICS / "camelyon17_effect_size_erm_softclip_v11_10s_fix_20260228.csv",
            fixed_domain_csv=METRICS / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv",
            fixed_effect_csv=METRICS / "camelyon17_effect_size_erm_softclip_v11_10s_fix_fixed30_20260228.csv",
        )
    )

    ft_suffix = args.finetune_suffix
    rows.extend(
        _load_delta_rows(
            suite="Camelyon17 finetune",
            baseline_regime="rcgdro_finetune",
            regimes=[
                "rcgdro_softclip_p95_a10_cam_finetune",
                "rcgdro_softclip_p97_a10_cam_finetune",
                "rcgdro_softclip_p99_a10_cam_finetune",
            ],
            selected_domain_csv=METRICS / f"camelyon17_domain_acc_with_loss_{ft_suffix}_selected_summary.csv",
            selected_effect_csv=METRICS / f"camelyon17_effect_size_{ft_suffix}_selected.csv",
            fixed_domain_csv=METRICS / f"camelyon17_domain_acc_with_loss_{ft_suffix}_fixed10_summary.csv",
            fixed_effect_csv=METRICS / f"camelyon17_effect_size_{ft_suffix}_fixed10.csv",
        )
    )

    df = pd.DataFrame(rows)
    df["level_num"] = df["level"].astype(int)
    df = df.sort_values(["suite", "level_num"]).drop(columns=["level_num"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    display = {
        "erm_softclip_p95_a10_cam": "P95",
        "erm_softclip_p97_a10_cam": "P97",
        "erm_softclip_p99_a10_cam": "P99",
        "rcgdro_softclip_p95_a10_cam_finetune": "P95",
        "rcgdro_softclip_p97_a10_cam_finetune": "P97",
        "rcgdro_softclip_p99_a10_cam_finetune": "P99",
    }

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Suite & Regime & Frac clip & $\Delta$Loss$\downarrow$ & $\Delta$Tail$\downarrow$ \\",
        r"\midrule",
    ]
    current_suite = None
    for _, row in df.iterrows():
        suite = row["suite"]
        suite_cell = suite if suite != current_suite else ""
        current_suite = suite
        frac = "---" if pd.isna(row["frac_clipped_val_mean"]) else _fmt(row["frac_clipped_val_mean"], 3)
        lines.append(
            f"{suite_cell} & {display[row['regime']]} & {frac} & "
            f"{_fmt_signed(row['selected_loss_delta'], 3)} & "
            f"{_fmt_signed(row['selected_tail_delta'], 2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="ascii")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
