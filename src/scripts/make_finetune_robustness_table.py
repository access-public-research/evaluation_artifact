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


def _proxy_delta(row: pd.Series, base_proxy: float) -> float:
    clip_proxy = row.get("proxy_worst_loss_clip_mean")
    if pd.notna(clip_proxy):
        return float(clip_proxy) - float(base_proxy)
    return float(row["proxy_worst_loss_mean"]) - float(base_proxy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_tex", default="")
    ap.add_argument("--out_main_tex", default="")
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)

    rows = []

    # CelebA selected finetune with held-out test WG
    ce = pd.read_csv(METRICS / "celeba_effect_size_finetune_scivalid_20260209_n5.csv").copy()
    ce["regime"] = ce["regime"].str.replace("_finetune", "", regex=False)
    ce_test = pd.read_csv(METRICS / "celeba_finetune_test_wg_selected_summary_20260309.csv")
    ce = ce.merge(ce_test[["regime", "test_oracle_wg_acc_mean"]], on="regime", how="left")
    ce_base = ce.loc[ce["regime"] == "rcgdro"].iloc[0]
    for regime in ["rcgdro_softclip_p95_a10", "rcgdro_softclip_p99_a10"]:
        row = ce.loc[ce["regime"] == regime].iloc[0]
        rows.append(
            {
                "dataset": "CelebA",
                "mode": "selected",
                "regime": regime,
                "delta_tail": float(row["tail_worst_cvar_mean"] - ce_base["tail_worst_cvar_mean"]),
                "delta_perf": float(row["test_oracle_wg_acc_mean"] - ce_base["test_oracle_wg_acc_mean"]),
                "perf_label": "test-WG",
            }
        )

    # Camelyon selected finetune
    cam_sel = pd.read_csv(METRICS / "camelyon17_effect_size_finetune_cam_rescue_n5_selected_20260214.csv").copy()
    cam_sel["regime"] = cam_sel["regime"].str.replace("_cam_finetune", "", regex=False).str.replace("_finetune", "", regex=False)
    cam_sel_base = cam_sel.loc[cam_sel["regime"] == "rcgdro"].iloc[0]
    for regime in ["rcgdro_softclip_p95_a10", "rcgdro_softclip_p99_a10"]:
        row = cam_sel.loc[cam_sel["regime"] == regime].iloc[0]
        rows.append(
            {
                "dataset": "Camelyon17",
                "mode": "selected",
                "regime": regime,
                "delta_tail": float(row["tail_worst_cvar_mean"] - cam_sel_base["tail_worst_cvar_mean"]),
                "delta_perf": float(row["test_hosp2_acc_mean"] - cam_sel_base["test_hosp2_acc_mean"]),
                "perf_label": "test-hosp2",
            }
        )

    # Camelyon fixed10 finetune
    cam_fx = pd.read_csv(METRICS / "camelyon17_effect_size_finetune_cam_rescue_n5_fixed10_20260214.csv").copy()
    cam_fx["regime"] = cam_fx["regime"].str.replace("_cam_finetune", "", regex=False).str.replace("_finetune", "", regex=False)
    cam_fx_base = cam_fx.loc[cam_fx["regime"] == "rcgdro"].iloc[0]
    for regime in ["rcgdro_softclip_p95_a10", "rcgdro_softclip_p99_a10"]:
        row = cam_fx.loc[cam_fx["regime"] == regime].iloc[0]
        rows.append(
            {
                "dataset": "Camelyon17",
                "mode": "fixed10",
                "regime": regime,
                "delta_tail": float(row["tail_worst_cvar_mean"] - cam_fx_base["tail_worst_cvar_mean"]),
                "delta_perf": float(row["test_hosp2_acc_mean"] - cam_fx_base["test_hosp2_acc_mean"]),
                "perf_label": "test-hosp2",
            }
        )

    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else METRICS / "finetune_robustness_summary_20260309.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    display = {
        "rcgdro_softclip_p95_a10": "P95",
        "rcgdro_softclip_p99_a10": "P99",
    }
    lines = []
    lines.append("\\begin{tabular}{lllrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Mode & Regime & $\\Delta$Tail & $\\Delta$Perf \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(
            f"{row['dataset']} & {row['mode']} & {display[row['regime']]} & "
            f"{_fmt_signed(row['delta_tail'], 2)} & {_fmt_signed(row['delta_perf'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_tex = Path(args.out_tex) if str(args.out_tex).strip() else TABLES / "table_finetune_robustness_20260309.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="ascii")

    main_rows = df.loc[df["mode"] == "selected", ["dataset", "regime", "delta_tail", "delta_perf"]].copy()
    main_lines = []
    main_lines.append("\\begin{tabular}{llrr}")
    main_lines.append("\\toprule")
    main_lines.append("Dataset & Regime & $\\Delta$Tail & $\\Delta$Perf \\\\")
    main_lines.append("\\midrule")
    for _, row in main_rows.iterrows():
        main_lines.append(
            f"{row['dataset']} & {display[row['regime']]} & "
            f"{_fmt_signed(row['delta_tail'], 2)} & {_fmt_signed(row['delta_perf'])} \\\\"
        )
    main_lines.append("\\bottomrule")
    main_lines.append("\\end{tabular}")
    out_main_tex = Path(args.out_main_tex) if str(args.out_main_tex).strip() else TABLES / "table_finetune_robustness_main_20260309.tex"
    out_main_tex.parent.mkdir(parents=True, exist_ok=True)
    out_main_tex.write_text("\n".join(main_lines) + "\n", encoding="ascii")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")
    print(f"[ok] wrote {out_main_tex}")


if __name__ == "__main__":
    main()
