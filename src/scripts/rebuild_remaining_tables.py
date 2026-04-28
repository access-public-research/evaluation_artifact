import argparse
from pathlib import Path

import pandas as pd


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pm(mean: float, ci: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f} $\\pm$ {ci:.{nd}f}"


def _delta(v: float, nd: int = 3) -> str:
    return f"({v:+.{nd}f})"


def rebuild_cam_p97_alpha(metrics_dir: Path, tables_dir: Path) -> None:
    df = pd.read_csv(metrics_dir / "camelyon17_p97_alpha_effect_size_with_domain_v10cam_p97alpha_5s_20260227.csv")
    order = [
        ("rcgdro", r"rcgdro ($\alpha=1.00$)"),
        ("rcgdro_softclip_p97_a05_cam", r"P97 $\alpha=0.05$"),
        ("rcgdro_softclip_p97_a10_cam", r"P97 $\alpha=0.10$"),
        ("rcgdro_softclip_p97_a20_cam", r"P97 $\alpha=0.20$"),
    ]
    base = df[df["regime"] == "rcgdro"].iloc[0]
    b_proxy = float(base["proxy_worst_loss_mean"])
    b_tail = float(base["tail_worst_cvar_mean"])
    b_test = float(base["test_hosp_2_acc_mean"])

    lines = [
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Regime & FracClip & Proxy$_{\text{unclip}}\downarrow$ ($\Delta$) & Tail CVaR$\downarrow$ ($\Delta$) & Test-H2$\uparrow$ ($\Delta$) \\",
        r"  \midrule",
    ]
    for regime, label in order:
        row = df[df["regime"] == regime].iloc[0]
        if regime == "rcgdro":
            frac = r"0.000 $\pm$ 0.000"
        else:
            frac = _pm(float(row["frac_clipped_val_mean"]), float(row["frac_clipped_val_ci"]), 3)
        p = float(row["proxy_worst_loss_mean"])
        p_ci = float(row["proxy_worst_loss_ci"])
        t = float(row["tail_worst_cvar_mean"])
        t_ci = float(row["tail_worst_cvar_ci"])
        h = float(row["test_hosp_2_acc_mean"])
        h_ci = float(row["test_hosp_2_acc_ci"])
        lines.append(
            f"  {label} & {frac} & {_pm(p, p_ci, 3)} {_delta(p-b_proxy, 3)} & "
            f"{_pm(t, t_ci, 2)} {_delta(t-b_tail, 2)} & {_pm(h, h_ci, 3)} {_delta(h-b_test, 3)} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    _write(tables_dir / "table_cam_p97_alpha_sweep.tex", lines)


def rebuild_mechanism_scalar(metrics_dir: Path, tables_dir: Path) -> None:
    df = pd.read_csv(metrics_dir / "mechanism_scalar_relaxation_p95_to_p99_20260227.csv")
    order = [
        ("CelebA", "all"),
        ("Camelyon17", "id"),
        ("Camelyon17", "ood"),
    ]
    lines = [
        r"\begin{tabular}{llcccccc}",
        r"  \toprule",
        r"  Dataset & Split & Top10@P95 & Top10@P99 & $\Delta$Top10 & Enrich@P95 & Enrich@P99 & $\Delta$Enrich \\",
        r"  \midrule",
    ]
    for dname, split in order:
        row = df[(df["dataset"] == dname) & (df["split"] == split)].iloc[0]
        lines.append(
            f"  {dname} & {split} & {float(row['top10_p95']):.3f} & {float(row['top10_p99']):.3f} & "
            f"{float(row['top10_relax_p95_to_p99']):+.3f} & {float(row['enrich_p95']):.2f} & "
            f"{float(row['enrich_p99']):.2f} & {float(row['enrich_change_p95_to_p99']):+.2f} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    _write(tables_dir / "table_mechanism_scalar_summary.tex", lines)


def rebuild_waterbirds_observability(metrics_dir: Path, tables_dir: Path) -> None:
    df = pd.read_csv(metrics_dir / "waterbirds_resnet50_phase0_val_metrics_v16wb_h256cal_10s_mc10_20260308.csv")
    fam_order = ["teacher_difficulty", "conf_teacher_wpl", "conf_init_wpl"]
    fam_label = {
        "teacher_difficulty": r"teacher\_difficulty",
        "conf_teacher_wpl": r"conf\_teacher\_wpl",
        "conf_init_wpl": r"conf\_init\_wpl",
    }
    lines = [
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Anchor family & Effective cells & P5 cell size & Median cell size & Frac small cells ($<10$) \\",
        r"  \midrule",
    ]
    for fam in fam_order:
        sub = df[df["family"] == fam].copy()
        for col in ["eff_cells", "p5_cell", "median_cell", "frac_small_cells"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        eff = float(sub["eff_cells"].mean())
        p5 = float(sub["p5_cell"].mean())
        med = float(sub["median_cell"].mean())
        frac = float(sub["frac_small_cells"].mean())
        lines.append(f"  {fam_label[fam]} & {eff:.1f} & {p5:.2f} & {med:.2f} & {frac:.3f} \\\\")
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    _write(tables_dir / "table_waterbirds_observability.tex", lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics")
    ap.add_argument("--tables_dir", default="paper/neurips2026_selection_risk/tables")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    rebuild_cam_p97_alpha(metrics_dir, tables_dir)
    rebuild_mechanism_scalar(metrics_dir, tables_dir)
    rebuild_waterbirds_observability(metrics_dir, tables_dir)
    print("[rebuild-remaining-tables] done")


if __name__ == "__main__":
    main()
