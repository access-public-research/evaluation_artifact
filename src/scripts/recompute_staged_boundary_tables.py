import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REGIME_ORDER = ["P95", "P97", "P99"]


def _load_head_dataset(
    effect_csv: Path,
    perf_col: str,
    regime_map: Dict[str, str],
    perf_csv: Path | None = None,
    perf_col_override: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(effect_csv)
    perf_df = None
    if perf_csv is not None and perf_csv.exists():
        perf_df = pd.read_csv(perf_csv)
    rows = []
    for regime_name, label in regime_map.items():
        r = df[df["regime"] == regime_name]
        if r.empty:
            raise ValueError(f"Missing regime {regime_name} in {effect_csv}")
        rr = r.iloc[0]
        if perf_df is not None:
            pr = perf_df[perf_df["regime"] == regime_name]
            if pr.empty:
                raise ValueError(f"Missing regime {regime_name} in {perf_csv}")
            perf_val = float(pr.iloc[0][perf_col_override or perf_col])
        else:
            perf_val = float(rr[perf_col])
        rows.append(
            {
                "label": label,
                "fracclip": float(rr["frac_clipped_val_mean"]) if pd.notna(rr["frac_clipped_val_mean"]) else 0.0,
                "tail": float(rr["tail_worst_cvar_mean"]),
                "perf": perf_val,
            }
        )
    return pd.DataFrame(rows)


def _load_camelyon_head(
    effect_csv: Path,
    domain_csv: Path,
    regime_map: Dict[str, str],
) -> pd.DataFrame:
    eff = pd.read_csv(effect_csv)
    dom = pd.read_csv(domain_csv)
    dom_mean = dom.groupby("regime", as_index=False)["test_hosp_2_acc"].mean()

    rows = []
    for regime_name, label in regime_map.items():
        er = eff[eff["regime"] == regime_name]
        dr = dom_mean[dom_mean["regime"] == regime_name]
        if er.empty or dr.empty:
            raise ValueError(f"Missing regime {regime_name} in Camelyon inputs")
        rr = er.iloc[0]
        dd = dr.iloc[0]
        rows.append(
            {
                "label": label,
                "fracclip": float(rr["frac_clipped_val_mean"]) if pd.notna(rr["frac_clipped_val_mean"]) else 0.0,
                "tail": float(rr["tail_worst_cvar_mean"]),
                "perf": float(dd["test_hosp_2_acc"]),
            }
        )
    return pd.DataFrame(rows)


def _find_recovery(df: pd.DataFrame, eps_perf: float, eps_tail: float) -> Tuple[str, float, str, float, str]:
    base = df[df["label"] == "rcgdro"].iloc[0]
    sweep = df[df["label"].isin(REGIME_ORDER)].copy()

    perf_threshold = base["perf"] - eps_perf
    tail_threshold = base["tail"] + eps_tail

    perf_row = None
    for label in REGIME_ORDER:
        row = sweep[sweep["label"] == label].iloc[0]
        if row["perf"] >= perf_threshold:
            perf_row = row
            break

    tail_row = None
    for label in REGIME_ORDER:
        row = sweep[sweep["label"] == label].iloc[0]
        if row["tail"] <= tail_threshold:
            tail_row = row
            break

    if perf_row is None and tail_row is None:
        status = "none_reached"
        return "not_reached", float("nan"), "not_reached", float("nan"), status
    if perf_row is not None and tail_row is None:
        status = "staged_directional_tail_unreached"
        return str(perf_row["label"]), float(perf_row["fracclip"]), "not_reached", float("nan"), status
    if perf_row is None and tail_row is not None:
        status = "tail_before_perf"
        return "not_reached", float("nan"), str(tail_row["label"]), float(tail_row["fracclip"]), status

    perf_idx = REGIME_ORDER.index(str(perf_row["label"]))
    tail_idx = REGIME_ORDER.index(str(tail_row["label"]))
    if perf_idx < tail_idx:
        status = "staged_explicit"
    elif perf_idx == tail_idx:
        status = "same_point"
    else:
        status = "tail_before_perf"

    return (
        str(perf_row["label"]),
        float(perf_row["fracclip"]),
        str(tail_row["label"]),
        float(tail_row["fracclip"]),
        status,
    )


def _fmt_u(regime: str, fracclip: float) -> str:
    if regime.startswith("not_reached"):
        return "$>{}$P99 (not reached)"
    return f"{regime} ($u{{=}}{fracclip:.3f}$)"


def _write_threshold_table(
    out_tex: Path,
    rows: List[Dict[str, object]],
) -> None:
    lines = [
        "\\begin{tabular}{llccc}",
        "  \\toprule",
        "  Dataset & Perf Metric & $u_{\\text{perf-rec}}$ & $u_{\\text{tail-rec}}$ & Staged? \\\\",
        "  \\midrule",
    ]
    for r in rows:
        lines.append(
            f"  {r['dataset']} & {r['perf_metric']} & {r['u_perf']} & {r['u_tail']} & {r['staged_note']} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}"]
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sensitivity_table(out_tex: Path, summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "  \\toprule",
        "  Dataset & Explicit Staged Pairs & Directional Staged Pairs & Same-Point Pairs & None Reached & Total Pairs \\\\",
        "  \\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"  {r['dataset']} & {int(r['staged_explicit_pairs'])} & {int(r['staged_directional_pairs'])} & {int(r['same_point_pairs'])} & {int(r['none_reached_pairs'])} & {int(r['total_pairs'])} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}"]
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics")
    ap.add_argument("--tables_dir", default="paper/neurips2026_selection_risk/tables")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    celeba = _load_head_dataset(
        metrics_dir / "celeba_effect_size_v7confclip_p60_p95_p97_p99_10s.csv",
        perf_col="oracle_wg_acc_mean",
        regime_map={
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10": "P95",
            "rcgdro_softclip_p97_a10": "P97",
            "rcgdro_softclip_p99_a10": "P99",
        },
        perf_csv=metrics_dir / "celeba_test_wg_selected_summary_v7confclip_p60_p95_p97_p99_10s_20260308.csv",
        perf_col_override="test_oracle_wg_acc_mean",
    )
    camelyon = _load_camelyon_head(
        metrics_dir / "camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv",
        metrics_dir / "camelyon17_resnet50_domain_acc_cam_softclip_a10_p99_20260207.csv",
        regime_map={
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10_cam": "P95",
            "rcgdro_softclip_p97_a10_cam": "P97",
            "rcgdro_softclip_p99_a10_cam": "P99",
        },
    )

    datasets = {
        "CelebA": (celeba, "Oracle WG acc"),
        "Camelyon17": (camelyon, "Test-hosp2 acc"),
    }

    eps_perf_fixed = 0.02
    eps_tail_fixed = 1.0
    threshold_rows = []
    threshold_csv_rows = []

    for ds, (df, perf_name) in datasets.items():
        u_perf_reg, u_perf_fc, u_tail_reg, u_tail_fc, status = _find_recovery(df, eps_perf_fixed, eps_tail_fixed)
        if status == "staged_explicit":
            staged_note = "Yes"
        elif status == "staged_directional_tail_unreached":
            staged_note = "Yes (tail later)"
        elif status == "same_point":
            staged_note = "Simultaneous"
        else:
            staged_note = "N/A"

        threshold_rows.append(
            {
                "dataset": ds,
                "perf_metric": perf_name,
                "u_perf": _fmt_u(u_perf_reg, u_perf_fc),
                "u_tail": _fmt_u(u_tail_reg, u_tail_fc),
                "staged_note": staged_note,
            }
        )
        threshold_csv_rows.append(
            {
                "dataset": ds,
                "perf_metric": "test_hosp_2_acc" if ds == "Camelyon17" else "test_oracle_wg_acc",
                "eps_perf": eps_perf_fixed,
                "eps_tail": eps_tail_fixed,
                "u_perf_rec_regime": u_perf_reg,
                "u_perf_rec_fracclip": None if pd.isna(u_perf_fc) else round(u_perf_fc, 3),
                "u_tail_rec_regime": u_tail_reg,
                "u_tail_rec_fracclip": None if pd.isna(u_tail_fc) else round(u_tail_fc, 3),
                "staged_recovery_note": (
                    "staged recovery (performance before tail)"
                    if status == "staged_explicit"
                    else (
                        "performance recovers while tail recovery is not reached within sweep"
                        if status == "staged_directional_tail_unreached"
                        else ("simultaneous recovery" if status == "same_point" else "no recovery point reached within sweep")
                    )
                ),
            }
        )

    threshold_csv = metrics_dir / "staged_boundary_thresholds_head_epsperf0p02_epstail1p0_20260227.csv"
    pd.DataFrame(threshold_csv_rows).to_csv(threshold_csv, index=False)
    _write_threshold_table(
        tables_dir / "table_staged_boundary_thresholds.tex",
        threshold_rows,
    )

    eps_perf_grid = [0.0, 0.005, 0.01, 0.02]
    eps_tail_grid = [0.5, 1.0, 1.5]

    sens_rows = []
    for ds, (df, _) in datasets.items():
        for ep in eps_perf_grid:
            for et in eps_tail_grid:
                u_perf_reg, u_perf_fc, u_tail_reg, u_tail_fc, status = _find_recovery(df, ep, et)
                sens_rows.append(
                    {
                        "dataset": ds,
                        "eps_perf": ep,
                        "eps_tail": et,
                        "u_perf_rec_regime": u_perf_reg,
                        "u_perf_rec_fracclip": None if pd.isna(u_perf_fc) else round(u_perf_fc, 3),
                        "u_tail_rec_regime": u_tail_reg,
                        "u_tail_rec_fracclip": None if pd.isna(u_tail_fc) else round(u_tail_fc, 3),
                        "status": status,
                    }
                )

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(metrics_dir / "staged_boundary_threshold_sensitivity_head_20260227.csv", index=False)

    summary_rows = []
    for ds in ["CelebA", "Camelyon17"]:
        sdf = sens_df[sens_df["dataset"] == ds]
        summary_rows.append(
            {
                "dataset": ds,
                "staged_explicit_pairs": int((sdf["status"] == "staged_explicit").sum()),
                "staged_directional_pairs": int((sdf["status"] == "staged_directional_tail_unreached").sum()),
                "same_point_pairs": int((sdf["status"] == "same_point").sum()),
                "none_reached_pairs": int((sdf["status"] == "none_reached").sum()),
                "total_pairs": int(sdf.shape[0]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metrics_dir / "staged_boundary_threshold_sensitivity_summary_head_20260227.csv", index=False)
    _write_sensitivity_table(
        tables_dir / "table_staged_boundary_tolerance_sensitivity.tex",
        summary_df,
    )


if __name__ == "__main__":
    main()
