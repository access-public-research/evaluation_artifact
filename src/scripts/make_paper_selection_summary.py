import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CaseSpec:
    case_label: str
    baseline_regime: str
    target_regime: str
    selected_effect_csv: Path
    fixed_effect_csv: Path
    selected_eval_csv: Path
    fixed_eval_csv: Path
    loss_col: str
    acc_col: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _specs(root: Path) -> list[CaseSpec]:
    art = root / "artifacts" / "metrics"
    figs = root / "figures"
    return [
        CaseSpec(
            case_label="Camelyon17 ERM",
            baseline_regime="erm",
            target_regime="erm_softclip_p95_a10_cam",
            selected_effect_csv=figs / "camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_20260228.csv",
            fixed_effect_csv=figs / "camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_fixed30_20260228.csv",
            selected_eval_csv=art / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv",
            fixed_eval_csv=art / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv",
            loss_col="test_hosp_2_loss_mean",
            acc_col="test_hosp_2_acc_mean",
        ),
        CaseSpec(
            case_label="CelebA ERM",
            baseline_regime="erm",
            target_regime="erm_softclip_p95_a10",
            selected_effect_csv=art / "celeba_effect_size_erm_softclip_celeba_10s_20260325.csv",
            fixed_effect_csv=art / "celeba_effect_size_erm_softclip_celeba_10s_20260325_fixed30.csv",
            selected_eval_csv=art / "celeba_test_wg_selected_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
            fixed_eval_csv=art / "celeba_test_wg_fixed30_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
            loss_col="test_oracle_wg_loss_mean",
            acc_col="test_oracle_wg_acc_mean",
        ),
        CaseSpec(
            case_label="Camelyon17 Finetune",
            baseline_regime="rcgdro_finetune",
            target_regime="rcgdro_softclip_p95_a10_cam_finetune",
            selected_effect_csv=art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_selected.csv",
            fixed_effect_csv=art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_fixed10.csv",
            selected_eval_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv",
            fixed_eval_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv",
            loss_col="test_hosp_2_loss_mean",
            acc_col="test_hosp_2_acc_mean",
        ),
        CaseSpec(
            case_label="Adaptive PseudoGroupDRO",
            baseline_regime="rcgdro",
            target_regime="rcgdro_softclip_p95_a10_cam",
            selected_effect_csv=art / "camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv",
            fixed_effect_csv=art / "camelyon17_selected_vs_epoch30_fixed_summary_cam_softclip_a10_p99_20260326.csv",
            selected_eval_csv=art / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_cam_softclip_a10_p99_20260326.csv",
            fixed_eval_csv=art / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_cam_softclip_a10_p99_20260326.csv",
            loss_col="test_hosp_2_loss_mean",
            acc_col="test_hosp_2_acc_mean",
        ),
    ]


def _load_proxy_tail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"proxy_selected_mean", "tail_worst_cvar_mean"}.issubset(df.columns):
        return df[["regime", "proxy_selected_mean", "tail_worst_cvar_mean"]].rename(
            columns={"proxy_selected_mean": "proxy_mean", "tail_worst_cvar_mean": "tail_mean"}
        )
    if {"proxy_worst_loss_clip_mean", "proxy_worst_loss_mean", "tail_worst_cvar_mean"}.issubset(df.columns):
        out = df[["regime", "proxy_worst_loss_clip_mean", "proxy_worst_loss_mean", "tail_worst_cvar_mean"]].copy()
        out["proxy_mean"] = pd.to_numeric(out["proxy_worst_loss_clip_mean"], errors="coerce").fillna(
            pd.to_numeric(out["proxy_worst_loss_mean"], errors="coerce")
        )
        out["tail_mean"] = pd.to_numeric(out["tail_worst_cvar_mean"], errors="coerce")
        return out[["regime", "proxy_mean", "tail_mean"]]
    if {"proxy_worst_loss", "tail_worst_cvar"}.issubset(df.columns):
        out = df.copy()
        if "proxy_worst_loss_clip" in out.columns:
            out["proxy_mean"] = pd.to_numeric(out["proxy_worst_loss_clip"], errors="coerce").fillna(
                pd.to_numeric(out["proxy_worst_loss"], errors="coerce")
            )
        else:
            out["proxy_mean"] = pd.to_numeric(out["proxy_worst_loss"], errors="coerce")
        out["tail_mean"] = pd.to_numeric(out["tail_worst_cvar"], errors="coerce")
        return out.groupby("regime", as_index=False)[["proxy_mean", "tail_mean"]].mean(numeric_only=True)
    raise ValueError(f"Unsupported proxy/tail schema in {path}")


def _load_eval(path: Path, loss_col: str, acc_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["regime", loss_col, acc_col]
    return df[keep].rename(columns={loss_col: "loss_mean", acc_col: "acc_mean"})


def _hazard_type(d_proxy_sel: float, d_loss_sel: float, d_tail_sel: float, d_loss_fix: float, d_tail_fix: float) -> str:
    selected_hazard = d_proxy_sel < 0 and d_loss_sel > 0 and d_tail_sel > 0
    fixed_hazard = d_loss_fix > 0 and d_tail_fix > 0
    fixed_safe = d_loss_fix <= 0 and d_tail_fix <= 0
    if selected_hazard and fixed_hazard:
        return "persistent"
    if selected_hazard and fixed_safe:
        return "transient"
    if selected_hazard:
        return "mixed"
    return "none"


def _write_tex(path: Path, df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"  \toprule",
        r"  Case & Hazard & $\Delta$Proxy$\downarrow$ & $\Delta$Held-out loss$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $\Delta$Tail CVaR$\downarrow$ \\",
        r"  \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "  {case} & {hazard} & {dproxy:+.3f} & {dloss:+.3f} & {dacc:+.3f} & {dtail:+.3f} \\\\".format(
                case=row["case_label"],
                hazard=row["hazard_type"],
                dproxy=row["delta_proxy_selected"],
                dloss=row["delta_loss_selected"],
                dacc=row["delta_acc_selected"],
                dtail=row["delta_tail_selected"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    rows = []
    for spec in _specs(_repo_root()):
        sel_pt = _load_proxy_tail(spec.selected_effect_csv).set_index("regime")
        fix_pt = _load_proxy_tail(spec.fixed_effect_csv).set_index("regime")
        sel_ev = _load_eval(spec.selected_eval_csv, spec.loss_col, spec.acc_col).set_index("regime")
        fix_ev = _load_eval(spec.fixed_eval_csv, spec.loss_col, spec.acc_col).set_index("regime")

        d_proxy_sel = float(sel_pt.loc[spec.target_regime, "proxy_mean"] - sel_pt.loc[spec.baseline_regime, "proxy_mean"])
        d_loss_sel = float(sel_ev.loc[spec.target_regime, "loss_mean"] - sel_ev.loc[spec.baseline_regime, "loss_mean"])
        d_acc_sel = float(sel_ev.loc[spec.target_regime, "acc_mean"] - sel_ev.loc[spec.baseline_regime, "acc_mean"])
        d_tail_sel = float(sel_pt.loc[spec.target_regime, "tail_mean"] - sel_pt.loc[spec.baseline_regime, "tail_mean"])
        d_loss_fix = float(fix_ev.loc[spec.target_regime, "loss_mean"] - fix_ev.loc[spec.baseline_regime, "loss_mean"])
        d_tail_fix = float(fix_pt.loc[spec.target_regime, "tail_mean"] - fix_pt.loc[spec.baseline_regime, "tail_mean"])
        d_acc_fix = float(fix_ev.loc[spec.target_regime, "acc_mean"] - fix_ev.loc[spec.baseline_regime, "acc_mean"])
        rows.append(
            {
                "case_label": spec.case_label,
                "hazard_type": _hazard_type(d_proxy_sel, d_loss_sel, d_tail_sel, d_loss_fix, d_tail_fix),
                "delta_proxy_selected": d_proxy_sel,
                "delta_loss_selected": d_loss_sel,
                "delta_acc_selected": d_acc_sel,
                "delta_tail_selected": d_tail_sel,
                "delta_loss_fixed": d_loss_fix,
                "delta_acc_fixed": d_acc_fix,
                "delta_tail_fixed": d_tail_fix,
            }
        )

    out = pd.DataFrame(rows)
    order = {"Camelyon17 ERM": 0, "CelebA ERM": 1, "Camelyon17 Finetune": 2, "Adaptive PseudoGroupDRO": 3}
    out["order"] = out["case_label"].map(order)
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_tex(out_tex, out)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
