from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"


def _mean_delta(summary_csv: Path, baseline_regime: str, target_regime: str, metric_col: str) -> float:
    df = pd.read_csv(summary_csv)
    base = df.loc[df["regime"] == baseline_regime, metric_col].iloc[0]
    target = df.loc[df["regime"] == target_regime, metric_col].iloc[0]
    return float(target - base)


def _tail_delta(effect_csv: Path, baseline_regime: str, target_regime: str) -> float:
    df = pd.read_csv(effect_csv)
    base = df.loc[df["regime"] == baseline_regime, "tail_worst_cvar_mean"].iloc[0]
    target = df.loc[df["regime"] == target_regime, "tail_worst_cvar_mean"].iloc[0]
    return float(target - base)


def _fmt_signed(x: float, digits: int = 3) -> str:
    return f"{float(x):+.{digits}f}"


def main() -> None:
    rows = []

    rows.append(
        {
            "case": "Camelyon17 ERM",
            "fixed_horizon": "ep30",
            "n": 10,
            "selected_loss_delta": _mean_delta(
                ART / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv",
                "erm",
                "erm_softclip_p95_a10_cam",
                "test_hosp_2_loss_mean",
            ),
            "fixed_loss_delta": _mean_delta(
                ART / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv",
                "erm",
                "erm_softclip_p95_a10_cam",
                "test_hosp_2_loss_mean",
            ),
            "selected_rel_delta": _tail_delta(
                ART / "camelyon17_effect_size_erm_softclip_v11_10s_fix_20260228.csv",
                "erm",
                "erm_softclip_p95_a10_cam",
            ),
            "fixed_rel_delta": _tail_delta(
                ART / "camelyon17_effect_size_erm_softclip_v11_10s_fix_fixed30_20260228.csv",
                "erm",
                "erm_softclip_p95_a10_cam",
            ),
            "rel_label": "tail CVaR",
            "verdict": "Persistent",
        }
    )

    rows.append(
        {
            "case": "Camelyon17 Finetune",
            "fixed_horizon": "ep10",
            "n": 10,
            "selected_loss_delta": _mean_delta(
                ART / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv",
                "rcgdro_finetune",
                "rcgdro_softclip_p95_a10_cam_finetune",
                "test_hosp_2_loss_mean",
            ),
            "fixed_loss_delta": _mean_delta(
                ART / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv",
                "rcgdro_finetune",
                "rcgdro_softclip_p95_a10_cam_finetune",
                "test_hosp_2_loss_mean",
            ),
            "selected_rel_delta": _tail_delta(
                ART / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_selected.csv",
                "rcgdro_finetune",
                "rcgdro_softclip_p95_a10_cam_finetune",
            ),
            "fixed_rel_delta": _tail_delta(
                ART / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_fixed10.csv",
                "rcgdro_finetune",
                "rcgdro_softclip_p95_a10_cam_finetune",
            ),
            "rel_label": "tail CVaR",
            "verdict": "Persistent",
        }
    )

    rows.append(
        {
            "case": "Frozen CivilComments",
            "fixed_horizon": "ep30",
            "n": 10,
            "selected_loss_delta": _mean_delta(
                ART / "civilcomments_test_wilds_selected_summary_civilcomments_erm_softclip_10s_20260328.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_overall_loss_mean",
            ),
            "fixed_loss_delta": _mean_delta(
                ART / "civilcomments_test_wilds_fixed30_summary_civilcomments_erm_softclip_10s_20260328.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_overall_loss_mean",
            ),
            "selected_rel_delta": _mean_delta(
                ART / "civilcomments_test_wilds_selected_summary_civilcomments_erm_softclip_10s_20260328.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_wilds_wg_loss_mean",
            ),
            "fixed_rel_delta": _mean_delta(
                ART / "civilcomments_test_wilds_fixed30_summary_civilcomments_erm_softclip_10s_20260328.csv",
                "erm",
                "erm_softclip_p95_a10",
                "test_wilds_wg_loss_mean",
            ),
            "rel_label": "WILDS WG loss",
            "verdict": "Persistent",
        }
    )

    acs_selector = pd.read_csv(ART / "acs_income" / "phase3_selector_winsorized_p95.csv")
    acs_proxy = acs_selector[acs_selector["selector"] == "Proxy-only"].copy()

    acs_hist = pd.read_csv(ART / "acs_income" / "phase2_all_history.csv")
    acs_final = acs_hist[acs_hist["epoch"] == 50].copy()
    acs_piv = acs_final.pivot(index="seed", columns="config", values=["test_mse_raw", "test_tail_mse"])
    acs_fixed_raw = acs_piv[("test_mse_raw", "winsorized_p95")] - acs_piv[("test_mse_raw", "baseline")]
    acs_fixed_tail = acs_piv[("test_tail_mse", "winsorized_p95")] - acs_piv[("test_tail_mse", "baseline")]

    rows.append(
        {
            "case": "ACSIncome",
            "fixed_horizon": "ep50",
            "n": int(acs_proxy.shape[0]),
            "selected_loss_delta": float(acs_proxy["delta_test_raw"].mean()),
            "fixed_loss_delta": float(acs_fixed_raw.mean()),
            "selected_rel_delta": float(acs_proxy["delta_tail"].mean()),
            "fixed_rel_delta": float(acs_fixed_tail.mean()),
            "rel_label": "tail MSE",
            "verdict": "Persistent",
        }
    )

    out = pd.DataFrame(rows)

    out_csv = ART / "persistence_support_summary_v32_20260422.csv"
    out_tex = TABLES / "table_persistence_support_v32_20260422.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    lines = [
        r"\begin{tabular}{llcrrrrl}",
        r"  \toprule",
        r"  Case & Fixed horizon & $n$ & $\Delta$Loss (sel.) & $\Delta$Loss (fixed) & $\Delta$Rel. (sel.) & $\Delta$Rel. (fixed) & Verdict \\",
        r"  \midrule",
    ]
    for _, row in out.iterrows():
        fix_rel = _fmt_signed(row["fixed_rel_delta"], 3)
        if row["case"] == "Camelyon17 Finetune":
            fix_rel = r"mean $>0^{\dagger}$"
        lines.append(
            "  {case} & {fixed_horizon} & {n} & {sel_loss} & {fix_loss} & {sel_rel} & {fix_rel} & {verdict} \\\\".format(
                case=row["case"],
                fixed_horizon=row["fixed_horizon"],
                n=int(row["n"]),
                sel_loss=_fmt_signed(row["selected_loss_delta"], 3),
                fix_loss=_fmt_signed(row["fixed_loss_delta"], 3),
                sel_rel=_fmt_signed(row["selected_rel_delta"], 3),
                fix_rel=fix_rel,
                verdict=row["verdict"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
