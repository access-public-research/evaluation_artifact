import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _fmt_signed(v: float, digits: int) -> str:
    return f"{float(v):+.{digits}f}"


def _fmt(v: float, digits: int) -> str:
    return f"{float(v):.{digits}f}"


def _corr_entry(corr_map: Dict[str, float], *candidates: str) -> tuple[str, float]:
    for key in candidates:
        if key in corr_map:
            return key, float(corr_map[key])
    raise KeyError(f"Missing correlation entry. Tried: {candidates}")


def _softclip_proxy_map(df: pd.DataFrame) -> Dict[str, float]:
    regimes = set(df["regime"])
    if "rcgdro" in regimes:
        base_regime = "rcgdro"
        softclip_regimes = ["rcgdro_softclip_p95_a10_cam", "rcgdro_softclip_p97_a10_cam", "rcgdro_softclip_p99_a10_cam"]
    elif "erm" in regimes:
        base_regime = "erm"
        softclip_regimes = ["erm_softclip_p95_a10_cam", "erm_softclip_p97_a10_cam", "erm_softclip_p99_a10_cam"]
    else:
        raise ValueError("softclip effect table must include either rcgdro or erm baseline")
    base = df[df["regime"] == base_regime].iloc[0]
    base_proxy = float(base["proxy_worst_loss_mean"])
    out: Dict[str, float] = {}
    for regime in softclip_regimes:
        row = df[df["regime"] == regime].iloc[0]
        out[regime] = float(row["proxy_worst_loss_clip_mean"] - base_proxy)
    return out


def _domain_delta_map(df: pd.DataFrame, metric: str = "test_hosp_2_acc") -> Dict[str, float]:
    d = df.drop_duplicates(subset=["regime", "seed", "epoch"]).copy()
    means = d.groupby("regime")[metric].mean().to_dict()
    base = float(means["erm"]) if "erm" in means else float(means["rcgdro"])
    return {reg: float(val - base) for reg, val in means.items() if reg != "erm"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orientation_csv", default="replication_rcg/artifacts/metrics/objective_orientation_tail_sign_20260305.csv")
    ap.add_argument("--corr_csv", default="replication_rcg/artifacts/metrics/objective_family_distortion_corr_20260305.csv")
    ap.add_argument("--softclip_effect", default="replication_rcg/artifacts/metrics/camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv")
    ap.add_argument("--softclip_domain", default="replication_rcg/artifacts/metrics/camelyon17_resnet50_domain_acc_cam_softclip_a10_p99_20260207.csv")
    ap.add_argument("--labelsmooth_effect", default="replication_rcg/artifacts/metrics/camelyon17_labelsmooth_effect_size_n10_20260304.csv")
    ap.add_argument("--labelsmooth_domain", default="replication_rcg/artifacts/metrics/camelyon17_resnet50_domain_acc_cam_labelsmooth_n10_20260304.csv")
    ap.add_argument("--focal_effect", default="replication_rcg/artifacts/metrics/camelyon17_focal_effect_size_n10_20260304.csv")
    ap.add_argument("--focal_domain", default="replication_rcg/artifacts/metrics/camelyon17_resnet50_domain_acc_cam_focal_n10_20260304.csv")
    ap.add_argument("--out_controls_tex", default="paper/neurips2026_selection_risk/tables/table_objective_family_controls_camelyon.tex")
    ap.add_argument("--out_predictive_tex", default="paper/neurips2026_selection_risk/tables/table_orientation_predictive_summary.tex")
    ap.add_argument("--out_corr_tex", default="paper/neurips2026_selection_risk/tables/table_objective_family_corr.tex")
    args = ap.parse_args()

    orient = _read(args.orientation_csv)
    corr = _read(args.corr_csv)
    soft = _read(args.softclip_effect)
    soft_dom = _read(args.softclip_domain)
    ls = _read(args.labelsmooth_effect)
    ls_dom = _read(args.labelsmooth_domain)
    fc = _read(args.focal_effect)
    fc_dom = _read(args.focal_domain)
    soft_perf_col = "test_hosp2_acc_mean" if "test_hosp2_acc_mean" in soft_dom.columns else "test_hosp_2_acc"

    proxy_soft = _softclip_proxy_map(soft)
    soft_perf = _domain_delta_map(soft_dom, metric=soft_perf_col)
    ls_proxy = ls.set_index("regime")["delta_proxy_vs_erm"].to_dict()
    ls_perf = _domain_delta_map(ls_dom)
    fc_proxy = fc.set_index("regime")["delta_proxy_vs_erm"].to_dict()
    fc_perf = _domain_delta_map(fc_dom)
    orient_map = orient.set_index("regime")[["delta_tail", "R_w"]].to_dict(orient="index")

    control_rows: List[str] = [
        "\\begin{tabular}{llccccc}",
        "  \\toprule",
        "  Family & Setting & $\\Delta$Proxy$\\downarrow$ & $\\Delta$Tail$\\downarrow$ & $\\Delta$test-hosp2$\\uparrow$ & $R_w$ (stabilized) & Orientation \\\\",
        "  \\midrule",
    ]
    softclip_prefix = "rcgdro_softclip" if "rcgdro_softclip_p95_a10_cam" in orient_map else "erm_softclip"
    spec = [
        ("SoftClip", "P95", f"{softclip_prefix}_p95_a10_cam", proxy_soft[f"{softclip_prefix}_p95_a10_cam"], soft_perf[f"{softclip_prefix}_p95_a10_cam"]),
        ("SoftClip", "P97", f"{softclip_prefix}_p97_a10_cam", proxy_soft[f"{softclip_prefix}_p97_a10_cam"], soft_perf[f"{softclip_prefix}_p97_a10_cam"]),
        ("SoftClip", "P99", f"{softclip_prefix}_p99_a10_cam", proxy_soft[f"{softclip_prefix}_p99_a10_cam"], soft_perf[f"{softclip_prefix}_p99_a10_cam"]),
        ("Label smooth", "e=0.02", "erm_labelsmooth_e02_cam", ls_proxy["erm_labelsmooth_e02_cam"], ls_perf["erm_labelsmooth_e02_cam"]),
        ("Label smooth", "e=0.10", "erm_labelsmooth_e10_cam", ls_proxy["erm_labelsmooth_e10_cam"], ls_perf["erm_labelsmooth_e10_cam"]),
        ("Label smooth", "e=0.20", "erm_labelsmooth_e20_cam", ls_proxy["erm_labelsmooth_e20_cam"], ls_perf["erm_labelsmooth_e20_cam"]),
        ("Focal", "g=1", "erm_focal_g1_cam", fc_proxy["erm_focal_g1_cam"], fc_perf["erm_focal_g1_cam"]),
        ("Focal", "g=2", "erm_focal_g2_cam", fc_proxy["erm_focal_g2_cam"], fc_perf["erm_focal_g2_cam"]),
        ("Focal", "g=4", "erm_focal_g4_cam", fc_proxy["erm_focal_g4_cam"], fc_perf["erm_focal_g4_cam"]),
    ]
    for fam, setting, regime, dproxy, dperf in spec:
        rw = float(orient_map[regime]["R_w"])
        dtail = float(orient_map[regime]["delta_tail"])
        orient_label = "upweight" if rw > 1.0 else "suppress"
        control_rows.append(
            f"  {fam} & {setting} & {_fmt_signed(dproxy,3)} & {_fmt_signed(dtail,2)} & {_fmt_signed(dperf,4)} & {_fmt(rw,3)} & {orient_label} \\\\"
        )
    control_rows.extend(["  \\bottomrule", "\\end{tabular}"])
    Path(args.out_controls_tex).write_text("\n".join(control_rows) + "\n", encoding="utf-8")

    pred_rows = [
        "\\begin{tabular}{lcccc}",
        "  \\toprule",
        "  Family & Setting & $R_w$ & Predicted sign & Observed $\\Delta$Tail \\\\",
        "  \\midrule",
    ]
    for fam, setting, regime in [
        ("SoftClip", "P95", f"{softclip_prefix}_p95_a10_cam"),
        ("Label smoothing", "$\\epsilon=0.10$", "erm_labelsmooth_e10_cam"),
        ("Focal", "$\\gamma=2$", "erm_focal_g2_cam"),
    ]:
        rw = float(orient_map[regime]["R_w"])
        dtail = float(orient_map[regime]["delta_tail"])
        pred = "improve" if rw > 1.0 else "inflate"
        pred_rows.append(f"  {fam} & {setting} & {_fmt(rw,3)} & {pred} & {_fmt_signed(dtail,2)} \\\\")
    pred_rows.extend(["  \\bottomrule", "\\end{tabular}"])
    Path(args.out_predictive_tex).write_text("\n".join(pred_rows) + "\n", encoding="utf-8")

    corr_map = corr.set_index("family")["pearson_r"].to_dict()
    soft_label, soft_corr = _corr_entry(corr_map, "SoftClip (Camelyon ERM)", "SoftClip (pooled head-only)")
    ls_label, ls_corr = _corr_entry(corr_map, "Label smoothing (Camelyon ERM)")
    fc_label, fc_corr = _corr_entry(corr_map, "Focal (Camelyon ERM)")
    corr_rows = [
        "\\begin{tabular}{lcc}",
        "  \\toprule",
        "  Objective family & Pearson $r$(DistMass, $\\Delta$Tail) & Direction \\\\",
        "  \\midrule",
        f"  {soft_label} & {_fmt_signed(soft_corr,3)} & larger distortion $\\rightarrow$ higher tail \\\\",
        f"  {ls_label} & {_fmt_signed(ls_corr,3)} & larger distortion $\\rightarrow$ lower tail \\\\",
        f"  {fc_label} & {_fmt_signed(fc_corr,3)} & larger distortion $\\rightarrow$ lower tail \\\\",
        "  \\bottomrule",
        "\\end{tabular}",
    ]
    Path(args.out_corr_tex).write_text("\n".join(corr_rows) + "\n", encoding="utf-8")
    print(f"[ok] wrote {args.out_controls_tex}")
    print(f"[ok] wrote {args.out_predictive_tex}")
    print(f"[ok] wrote {args.out_corr_tex}")


if __name__ == "__main__":
    main()
