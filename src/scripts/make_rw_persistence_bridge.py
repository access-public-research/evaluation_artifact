import argparse
from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


ROOT = Path(__file__).resolve().parents[2]


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(ci95_mean(vals))


def _fmt(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def _trend_label(delta_rw_mean: float) -> str:
    return "relaxes" if delta_rw_mean > 0 else "entrenches"


def _build_seed_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for (dataset, label, seed), sub in rows.groupby(["dataset", "label", "seed"], dropna=False):
        sub = sub.sort_values("epoch").copy()
        early = sub[sub["epoch"].between(1, 5)]
        late = sub[sub["epoch"].between(25, 30)]
        early_gap = early["frac_tail_clipped"] - early["frac_core_clipped"]
        late_gap = late["frac_tail_clipped"] - late["frac_core_clipped"]
        out_rows.append(
            {
                "dataset": dataset,
                "label": label,
                "seed": int(seed),
                "early_rw": float(early["R_w"].mean()),
                "late_rw": float(late["R_w"].mean()),
                "delta_rw": float(late["R_w"].mean() - early["R_w"].mean()),
                "early_gap": float(early_gap.mean()),
                "late_gap": float(late_gap.mean()),
                "delta_gap": float(late_gap.mean() - early_gap.mean()),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)


def _build_suite_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, label), sub in seed_df.groupby(["dataset", "label"], dropna=False):
        early_rw_m, early_rw_ci = _mean_ci(sub["early_rw"].to_numpy())
        late_rw_m, late_rw_ci = _mean_ci(sub["late_rw"].to_numpy())
        delta_rw_m, delta_rw_ci = _mean_ci(sub["delta_rw"].to_numpy())
        early_gap_m, early_gap_ci = _mean_ci(sub["early_gap"].to_numpy())
        late_gap_m, late_gap_ci = _mean_ci(sub["late_gap"].to_numpy())
        delta_gap_m, delta_gap_ci = _mean_ci(sub["delta_gap"].to_numpy())
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "n": int(sub.shape[0]),
                "early_rw_mean": early_rw_m,
                "early_rw_ci": early_rw_ci,
                "late_rw_mean": late_rw_m,
                "late_rw_ci": late_rw_ci,
                "delta_rw_mean": delta_rw_m,
                "delta_rw_ci": delta_rw_ci,
                "early_gap_mean": early_gap_m,
                "early_gap_ci": early_gap_ci,
                "late_gap_mean": late_gap_m,
                "late_gap_ci": late_gap_ci,
                "delta_gap_mean": delta_gap_m,
                "delta_gap_ci": delta_gap_ci,
                "trend": _trend_label(delta_rw_m),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _write_tex(summary_df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"  \toprule",
        r"  Suite & Early $R_w$ & Late $R_w$ & $\Delta R_w$ & $\Delta$clip gap & Trend \\",
        r"  \midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "  {label} & {early_rw} & {late_rw} & {delta_rw} & {delta_gap} & {trend} \\\\".format(
                label=row["label"].replace(" ERM P95", ""),
                early_rw=_fmt(row["early_rw_mean"], row["early_rw_ci"], digits=3),
                late_rw=_fmt(row["late_rw_mean"], row["late_rw_ci"], digits=3),
                delta_rw=_fmt(row["delta_rw_mean"], row["delta_rw_ci"], digits=3),
                delta_gap=_fmt(row["delta_gap_mean"], row["delta_gap_ci"], digits=3),
                trend=row["trend"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_figure(summary_rows: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    colors = {"CelebA ERM P95": "#c06c2b", "Camelyon17 ERM P95": "#0f6cbd"}
    labels = {
        "CelebA ERM P95": "CelebA",
        "Camelyon17 ERM P95": "Camelyon17",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=180)
    x = np.array([0, 1], dtype=float)

    for _, row in summary_rows.iterrows():
        label = row["label"]
        color = colors[label]
        short = labels[label]
        y_rw = np.array([row["early_rw_mean"], row["late_rw_mean"]], dtype=float)
        y_rw_err = np.array([row["early_rw_ci"], row["late_rw_ci"]], dtype=float)
        axes[0].plot(x, y_rw, marker="o", ms=6, lw=2.2, color=color, label=short)
        axes[0].fill_between(x, y_rw - y_rw_err, y_rw + y_rw_err, color=color, alpha=0.18)
        axes[0].annotate(
            f"{row['trend']} ($\\Delta R_w={row['delta_rw_mean']:+.3f}$)",
            (x[-1], y_rw[-1]),
            xytext=(8, -6 if "Camelyon17" in short else 8),
            textcoords="offset points",
            fontsize=8.1,
            color=color,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9),
        )

        early_gap = row["early_gap_mean"]
        late_gap = row["late_gap_mean"]
        early_gap_ci = row["early_gap_ci"]
        late_gap_ci = row["late_gap_ci"]
        y_gap = np.array([early_gap, late_gap], dtype=float)
        y_gap_err = np.array([early_gap_ci, late_gap_ci], dtype=float)
        axes[1].plot(x, y_gap, marker="o", ms=6, lw=2.2, color=color, label=short)
        axes[1].fill_between(x, y_gap - y_gap_err, y_gap + y_gap_err, color=color, alpha=0.18)
        gap_dir = "falls" if float(row["delta_gap_mean"]) < 0 else "rises"
        axes[1].annotate(
            f"clip gap {gap_dir}",
            (x[-1], y_gap[-1]),
            xytext=(8, -6 if "Camelyon17" in short else 8),
            textcoords="offset points",
            fontsize=8.1,
            color=color,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9),
        )

    axes[0].axhline(1.0, color="#6b7280", ls="--", lw=1.1)
    axes[0].set_xticks(x, ["Early\n(1-5)", "Late\n(25-30)"])
    axes[0].set_ylabel(r"$R_w$")
    axes[0].set_title(r"Suppression relaxes on CelebA but entrenches on Camelyon17", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")

    axes[1].axhline(0.0, color="#6b7280", ls="--", lw=1.1)
    axes[1].set_xticks(x, ["Early\n(1-5)", "Late\n(25-30)"])
    axes[1].set_ylabel("Tail-core clip gap")
    axes[1].set_title("Tail-core clip exposure moves in the same direction", fontsize=10.5)

    for ax in axes:
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows_csv",
        default="artifacts/metrics/rw_persistence_timecourse_20260327_rows.csv",
    )
    ap.add_argument(
        "--out_seed_csv",
        default="artifacts/metrics/rw_persistence_bridge_seed_20260329.csv",
    )
    ap.add_argument(
        "--out_summary_csv",
        default="artifacts/metrics/rw_persistence_bridge_summary_20260329.csv",
    )
    ap.add_argument(
        "--out_tex",
        default="paper/neurips2026_selection_risk/tables/table_rw_persistence_bridge.tex",
    )
    ap.add_argument(
        "--out_pdf",
        default="paper/neurips2026_selection_risk/figures/fig_rw_persistence_bridge.pdf",
    )
    ap.add_argument(
        "--out_png",
        default="paper/neurips2026_selection_risk/figures/fig_rw_persistence_bridge.png",
    )
    args = ap.parse_args()

    rows = pd.read_csv(ROOT / args.rows_csv)
    seed_df = _build_seed_summary(rows)
    summary_df = _build_suite_summary(seed_df)

    out_seed_csv = ROOT / args.out_seed_csv
    out_summary_csv = ROOT / args.out_summary_csv
    out_seed_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out_seed_csv, index=False)
    summary_df.to_csv(out_summary_csv, index=False)

    _write_tex(summary_df, ROOT / args.out_tex)
    _make_figure(summary_df, ROOT / args.out_pdf, ROOT / args.out_png)

    print(f"[ok] wrote {out_seed_csv}")
    print(f"[ok] wrote {out_summary_csv}")
    print(f"[ok] wrote {ROOT / args.out_tex}")
    print(f"[ok] wrote {ROOT / args.out_pdf}")
    print(f"[ok] wrote {ROOT / args.out_png}")


if __name__ == "__main__":
    main()
