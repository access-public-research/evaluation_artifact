import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SuitePaths:
    case_label: str
    regimes: tuple[str, ...]
    selected_effect_csv: Path
    fixed_effect_csv: Path
    selected_loss_csv: Path
    fixed_loss_csv: Path
    loss_col: str


REGIME_LABEL = {
    "erm_softclip_p95_a10": "P95",
    "erm_softclip_p95_a10_cam": "P95",
    "rcgdro_softclip_p95_a10": "P95",
    "rcgdro_softclip_p95_a10_cam": "P95",
    "rcgdro_softclip_p95_a10_cam_finetune": "P95",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suites(root: Path) -> list[SuitePaths]:
    art = root / "artifacts" / "metrics"
    figs = root / "figures"
    return [
        SuitePaths(
            case_label="CelebA ERM",
            regimes=("erm_softclip_p95_a10",),
            selected_effect_csv=art / "celeba_effect_size_erm_softclip_celeba_10s_20260325.csv",
            fixed_effect_csv=art / "celeba_effect_size_erm_softclip_celeba_10s_20260325_fixed30.csv",
            selected_loss_csv=art / "celeba_test_wg_selected_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
            fixed_loss_csv=art / "celeba_test_wg_fixed30_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
            loss_col="test_oracle_wg_loss_mean",
        ),
        SuitePaths(
            case_label="Camelyon17 ERM",
            regimes=("erm_softclip_p95_a10_cam",),
            selected_effect_csv=figs / "camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_20260228.csv",
            fixed_effect_csv=figs / "camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_fixed30_20260228.csv",
            selected_loss_csv=art / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv",
            fixed_loss_csv=art / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv",
            loss_col="test_hosp_2_loss_mean",
        ),
        SuitePaths(
            case_label="Camelyon17 Finetune",
            regimes=("rcgdro_softclip_p95_a10_cam_finetune",),
            selected_effect_csv=art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_selected.csv",
            fixed_effect_csv=art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_fixed10.csv",
            selected_loss_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv",
            fixed_loss_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv",
            loss_col="test_hosp_2_loss_mean",
        ),
        SuitePaths(
            case_label="CelebA Adaptive PseudoGroupDRO",
            regimes=("rcgdro_softclip_p95_a10",),
            selected_effect_csv=art / "celeba_effect_size_v7confclip_p60_p95_p97_p99_10s.csv",
            fixed_effect_csv=figs / "celeba_properness_summary_v7confclip_p60_p95_p97_p99_10s_fixed30.csv",
            selected_loss_csv=art / "celeba_test_wg_selected_with_loss_summary_v7confclip_p60_p95_p97_p99_10s_20260325.csv",
            fixed_loss_csv=art / "celeba_test_wg_fixed30_with_loss_summary_v7confclip_p60_p95_p97_p99_10s_20260326.csv",
            loss_col="test_oracle_wg_loss_mean",
        ),
        SuitePaths(
            case_label="Camelyon17 Adaptive PseudoGroupDRO",
            regimes=("rcgdro_softclip_p95_a10_cam",),
            selected_effect_csv=art / "camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv",
            fixed_effect_csv=art / "camelyon17_selected_vs_epoch30_fixed_summary_cam_softclip_a10_p99_20260326.csv",
            selected_loss_csv=art / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_cam_softclip_a10_p99_20260326.csv",
            fixed_loss_csv=art / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_cam_softclip_a10_p99_20260326.csv",
            loss_col="test_hosp_2_loss_mean",
        ),
    ]


def _load_proxy_tail(effect_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(effect_csv)
    if "proxy_selected_mean" in df.columns and "tail_worst_cvar_mean" in df.columns:
        return df[["regime", "proxy_selected_mean", "tail_worst_cvar_mean"]].rename(
            columns={"proxy_selected_mean": "proxy_mean", "tail_worst_cvar_mean": "tail_mean"}
        )
    if "proxy_worst_loss_clip_mean" in df.columns or "proxy_worst_loss_mean" in df.columns:
        proxy_clip = pd.to_numeric(df.get("proxy_worst_loss_clip_mean"), errors="coerce")
        proxy_raw = pd.to_numeric(df.get("proxy_worst_loss_mean"), errors="coerce")
        out = df[["regime"]].copy()
        out["proxy_mean"] = proxy_clip.fillna(proxy_raw)
        out["tail_mean"] = pd.to_numeric(df["tail_worst_cvar_mean"], errors="coerce")
        return out
    if "proxy_worst_loss" in df.columns or "proxy_worst_loss_clip" in df.columns:
        proxy_clip = pd.to_numeric(df.get("proxy_worst_loss_clip"), errors="coerce")
        proxy_raw = pd.to_numeric(df.get("proxy_worst_loss"), errors="coerce")
        work = df[["regime"]].copy()
        work["proxy_mean"] = proxy_clip.fillna(proxy_raw)
        work["tail_mean"] = pd.to_numeric(df["tail_worst_cvar"], errors="coerce")
        return work.groupby("regime", as_index=False)[["proxy_mean", "tail_mean"]].mean(numeric_only=True)
    raise ValueError(f"Unsupported proxy/tail schema in {effect_csv}")


def _load_loss(loss_csv: Path, loss_col: str) -> pd.DataFrame:
    df = pd.read_csv(loss_csv)
    return df[["regime", loss_col]].rename(columns={loss_col: "loss_mean"})


def _hazard_type(
    d_proxy_sel: float,
    d_loss_sel: float,
    d_tail_sel: float,
    d_loss_fix: float,
    d_tail_fix: float,
) -> str:
    sel_hazard = d_proxy_sel < 0 and d_loss_sel > 0 and d_tail_sel > 0
    fix_hazard = d_loss_fix > 0 and d_tail_fix > 0
    fix_safe = d_loss_fix <= 0 and d_tail_fix <= 0
    if sel_hazard and fix_hazard:
        return "persistent"
    if sel_hazard and fix_safe:
        return "transient"
    if sel_hazard:
        return "mixed"
    return "none"


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def _write_tex(path: Path, df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lllccccc}",
        r"  \toprule",
        r"  Case & Regime & Hazard & $\Delta$Proxy (sel.)$\downarrow$ & $\Delta$Loss (sel.)$\downarrow$ & $\Delta$Tail (sel.)$\downarrow$ & $\Delta$Loss (fix.)$\downarrow$ & $\Delta$Tail (fix.)$\downarrow$ \\",
        r"  \midrule",
    ]
    last_case = None
    for _, row in df.iterrows():
        case = row["case_label"]
        case_cell = case if case != last_case else ""
        last_case = case
        lines.append(
            "  "
            + " & ".join(
                [
                    case_cell,
                    str(row["regime_label"]),
                    str(row["hazard_type"]),
                    _fmt(row["delta_proxy_selected"]),
                    _fmt(row["delta_loss_selected"]),
                    _fmt(row["delta_tail_selected"]),
                    _fmt(row["delta_loss_fixed"]),
                    _fmt(row["delta_tail_fixed"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    root = _repo_root()
    rows = []
    for suite in _default_suites(root):
        sel_pt = _load_proxy_tail(suite.selected_effect_csv).set_index("regime")
        fix_pt = _load_proxy_tail(suite.fixed_effect_csv).set_index("regime")
        sel_loss = _load_loss(suite.selected_loss_csv, suite.loss_col).set_index("regime")
        fix_loss = _load_loss(suite.fixed_loss_csv, suite.loss_col).set_index("regime")

        baseline_regime = sel_pt.index[0]
        if baseline_regime not in fix_pt.index:
            baseline_regime = fix_pt.index[0]

        base_proxy_sel = float(sel_pt.loc[baseline_regime, "proxy_mean"])
        base_tail_sel = float(sel_pt.loc[baseline_regime, "tail_mean"])
        base_loss_sel = float(sel_loss.loc[baseline_regime, "loss_mean"])
        base_proxy_fix = float(fix_pt.loc[baseline_regime, "proxy_mean"])
        base_tail_fix = float(fix_pt.loc[baseline_regime, "tail_mean"])
        base_loss_fix = float(fix_loss.loc[baseline_regime, "loss_mean"])

        for regime in suite.regimes:
            d_proxy_sel = float(sel_pt.loc[regime, "proxy_mean"] - base_proxy_sel)
            d_tail_sel = float(sel_pt.loc[regime, "tail_mean"] - base_tail_sel)
            d_loss_sel = float(sel_loss.loc[regime, "loss_mean"] - base_loss_sel)
            d_proxy_fix = float(fix_pt.loc[regime, "proxy_mean"] - base_proxy_fix)
            d_tail_fix = float(fix_pt.loc[regime, "tail_mean"] - base_tail_fix)
            d_loss_fix = float(fix_loss.loc[regime, "loss_mean"] - base_loss_fix)
            rows.append(
                {
                    "case_label": suite.case_label,
                    "regime": regime,
                    "regime_label": REGIME_LABEL.get(regime, regime),
                    "delta_proxy_selected": d_proxy_sel,
                    "delta_tail_selected": d_tail_sel,
                    "delta_loss_selected": d_loss_sel,
                    "delta_proxy_fixed": d_proxy_fix,
                    "delta_tail_fixed": d_tail_fix,
                    "delta_loss_fixed": d_loss_fix,
                    "hazard_type": _hazard_type(
                        d_proxy_sel=d_proxy_sel,
                        d_loss_sel=d_loss_sel,
                        d_tail_sel=d_tail_sel,
                        d_loss_fix=d_loss_fix,
                        d_tail_fix=d_tail_fix,
                    ),
                }
            )

    out = pd.DataFrame(rows)
    order = {
        "CelebA ERM": 0,
        "Camelyon17 ERM": 1,
        "Camelyon17 Finetune": 2,
        "CelebA Adaptive PseudoGroupDRO": 3,
        "Camelyon17 Adaptive PseudoGroupDRO": 4,
    }
    out["case_order"] = out["case_label"].map(order)
    out["reg_order"] = out["regime_label"].map({"P95": 0}).fillna(9)
    out = out.sort_values(["case_order", "reg_order"]).drop(columns=["case_order", "reg_order"])

    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_tex(out_tex, out)
    print(f"[selection-risk] wrote {out_csv}")
    print(f"[selection-risk] wrote {out_tex}")


if __name__ == "__main__":
    main()
