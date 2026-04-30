from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "neurips2026_selection_risk"


def _load_points() -> pd.DataFrame:
    orient = pd.read_csv(ROOT / "artifacts/metrics/objective_orientation_tail_sign_ermsoftclipfix_20260325.csv")
    soft = pd.read_csv(ROOT / "figures/camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_20260228.csv")
    soft_acc = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv")
    focal = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_focal_effect_size_n10_20260304.csv")
    smooth = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_labelsmooth_effect_size_n10_20260304.csv")
    gce = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_gce_q07_10s_selector_summary_20260430.csv")

    soft = soft[soft["regime"].isin(["erm_softclip_p95_a10_cam", "erm_softclip_p97_a10_cam", "erm_softclip_p99_a10_cam"])]
    soft = (
        soft.groupby("regime", as_index=False)[["proxy_worst_loss_clip", "proxy_worst_loss", "tail_worst_cvar"]]
        .mean(numeric_only=True)
        .copy()
    )
    soft_acc = soft_acc[soft_acc["regime"].isin(["erm", "erm_softclip_p95_a10_cam", "erm_softclip_p97_a10_cam", "erm_softclip_p99_a10_cam"])]
    base_soft = pd.read_csv(ROOT / "figures/camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_20260228.csv")
    erm_row = base_soft[base_soft["regime"] == "erm"].groupby("regime", as_index=False)[["proxy_worst_loss", "tail_worst_cvar"]].mean(numeric_only=True).iloc[0]
    erm_acc = float(soft_acc[soft_acc["regime"] == "erm"]["test_hosp_2_acc_mean"].iloc[0])
    soft = soft.merge(soft_acc[["regime", "test_hosp_2_acc_mean"]], on="regime", how="left")
    soft["delta_proxy"] = pd.to_numeric(soft["proxy_worst_loss_clip"], errors="coerce").fillna(
        pd.to_numeric(soft["proxy_worst_loss"], errors="coerce")
    ) - float(erm_row["proxy_worst_loss"])
    soft["delta_tail"] = pd.to_numeric(soft["tail_worst_cvar"], errors="coerce") - float(erm_row["tail_worst_cvar"])
    soft["delta_acc"] = pd.to_numeric(soft["test_hosp_2_acc_mean"], errors="coerce") - erm_acc
    soft = soft[["regime", "delta_proxy", "delta_tail", "delta_acc"]]
    focal = focal[["regime", "delta_proxy_vs_erm", "delta_tail_vs_erm", "delta_hosp2_vs_erm"]].rename(
        columns={"delta_proxy_vs_erm": "delta_proxy", "delta_tail_vs_erm": "delta_tail", "delta_hosp2_vs_erm": "delta_acc"}
    )
    smooth = smooth[["regime", "delta_proxy_vs_erm", "delta_tail_vs_erm", "delta_hosp2_vs_erm"]].rename(
        columns={"delta_proxy_vs_erm": "delta_proxy", "delta_tail_vs_erm": "delta_tail", "delta_hosp2_vs_erm": "delta_acc"}
    )

    orient = orient[["regime", "family", "R_w"]].copy()
    merged = pd.concat([soft, focal, smooth], ignore_index=True).merge(orient, on="regime", how="inner")
    gce_row = gce[gce["contrast"].eq("GCE proxy-best $-$ ERM baseline-selected")].iloc[0]
    gce_point = pd.DataFrame(
        [
            {
                "regime": "erm_gce_q07_cam",
                "delta_proxy": float(gce_row["delta_gce_proxy_mean"]),
                "delta_tail": float(gce_row["delta_tail_worst_cvar_mean"]),
                "delta_acc": float(gce_row["delta_test_acc_mean"]),
                "family": "GCE",
                "R_w": float(gce_row["mean_rw_a"]),
            }
        ]
    )
    merged = pd.concat([merged, gce_point], ignore_index=True)
    label_map = {
        "erm_softclip_p95_a10_cam": "SoftClip P95",
        "erm_softclip_p97_a10_cam": "SoftClip P97",
        "erm_softclip_p99_a10_cam": "SoftClip P99",
        "erm_labelsmooth_e02_cam": "LS 0.02",
        "erm_labelsmooth_e05_cam": "LS 0.05",
        "erm_labelsmooth_e10_cam": "LS 0.10",
        "erm_labelsmooth_e20_cam": "LS 0.20",
        "erm_focal_g1_cam": "Focal 1",
        "erm_focal_g2_cam": "Focal 2",
        "erm_focal_g4_cam": "Focal 4",
        "erm_gce_q07_cam": "GCE q=0.7",
    }
    merged["label"] = merged["regime"].map(label_map)
    return merged


def _write_tex(path: Path, df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Setting & $\Delta$Proxy$\downarrow$ & $\Delta$Tail CVaR$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $R_w$ \\",
        r"  \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "  {label} & {dproxy:+.3f} & {dtail:+.3f} & {dacc:+.4f} & {rw:.3f} \\\\".format(
                label=row["label"],
                dproxy=row["delta_proxy"],
                dtail=row["delta_tail"],
                dacc=row["delta_acc"],
                rw=row["R_w"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = _load_points()
    colors = {"SoftClip": "#D55E00", "GCE": "#8C6D1F", "LabelSmooth": "#009E73", "Focal": "#0072B2"}
    markers = {"SoftClip": "X", "GCE": "P", "LabelSmooth": "o", "Focal": "D"}
    offsets = {
        "SoftClip P95": (12, 10),
        "SoftClip P99": (12, 8),
        "GCE q=0.7": (10, 10),
        "LS 0.10": (-34, 10),
        "LS 0.20": (-34, -12),
        "Focal 1": (8, 10),
        "Focal 4": (8, 10),
    }
    label_points = {"SoftClip P95", "SoftClip P99", "GCE q=0.7", "LS 0.10", "LS 0.20", "Focal 1", "Focal 4"}

    fig, ax = plt.subplots(figsize=(7.45, 4.5))
    x_vals = np.log10(df["R_w"].to_numpy(dtype=float))
    y_vals = df["delta_tail"].to_numpy(dtype=float)
    x_min = float(x_vals.min() - 0.10)
    x_max = float(x_vals.max() + 0.08)
    y_min = float(y_vals.min() - 0.45)
    y_max = float(y_vals.max() + 0.55)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axvspan(x_min, 0.0, color="#fee2e2", alpha=0.32, zorder=0)
    ax.axvspan(0.0, x_max, color="#dcfce7", alpha=0.22, zorder=0)
    for family, sub in df.groupby("family"):
        x = np.log10(sub["R_w"].to_numpy(dtype=float))
        y = sub["delta_tail"].to_numpy(dtype=float)
        ax.scatter(x, y, s=170, c=colors[family], marker=markers[family], edgecolors="white", linewidths=0.9, alpha=0.95, label=family)
        for _, row in sub.iterrows():
            if row["label"] not in label_points:
                continue
            dx, dy = offsets.get(row["label"], (6, 5))
            ax.annotate(
                row["label"],
                (np.log10(float(row["R_w"])), float(row["delta_tail"])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.2,
                color=colors[family],
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=colors[family], lw=0.4, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=colors[family], shrinkA=2, shrinkB=2),
            )
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.8)
    ax.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$\log_{10} R_w$ (tail/core gradient amplification)", fontsize=10)
    ax.set_ylabel(r"$\Delta$Tail CVaR vs ERM", fontsize=10)
    ax.set_title("Tail direction follows weighting orientation", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper right", fontsize=8.3)
    ax.text(
        x_min + 0.06 * (x_max - x_min),
        y_max - 0.10 * (y_max - y_min),
        "Suppressive side\n" + r"($R_w<1$; tail worsens)",
        fontsize=8.3,
        ha="left",
        va="top",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
    )
    ax.text(
        x_max - 0.38 * (x_max - x_min),
        y_min + 0.08 * (y_max - y_min),
        "Upweighting side\n" + r"($R_w>1$; tail improves)",
        fontsize=8.3,
        ha="left",
        va="bottom",
        color="#166534",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
    )
    fig.tight_layout()

    fig_dir = PAPER / "figures"
    table_dir = PAPER / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / "mechanism_orientation_points.csv", index=False)
    _write_tex(table_dir / "table_mechanism_orientation.tex", df.sort_values(["family", "R_w"]))
    fig.savefig(fig_dir / "fig_mechanism_orientation.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_mechanism_orientation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {fig_dir / 'fig_mechanism_orientation.pdf'}")
    print(f"[ok] wrote {fig_dir / 'fig_mechanism_orientation.png'}")
    print(f"[ok] wrote {table_dir / 'table_mechanism_orientation.tex'}")
    print(f"[ok] wrote {table_dir / 'mechanism_orientation_points.csv'}")


if __name__ == "__main__":
    main()
