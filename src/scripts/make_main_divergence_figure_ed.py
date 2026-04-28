from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "neurips2026_selection_risk"


def load_divergence_points() -> pd.DataFrame:
    rows = []

    selected = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv")
    guard = pd.read_csv(ROOT / "artifacts/metrics/camelyon_erm_p95_selector_summary_trueval_20260329.csv")
    erm_base = selected[selected["regime"] == "erm"].iloc[0]
    erm_guard = guard[guard["selection_policy"] == "guardrail"].iloc[0]
    rows.extend(
        [
            {"suite": "Camelyon17 ERM", "point": "Baseline", "acc": float(erm_base["test_hosp_2_acc_mean"]), "loss": float(erm_base["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 ERM", "point": "Proxy-only P95", "acc": float(selected[selected["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(selected[selected["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_loss_mean"].iloc[0])},
            {"suite": "Camelyon17 ERM", "point": "Guardrail P95", "acc": float(erm_base["test_hosp_2_acc_mean"] + erm_guard["delta_acc"]), "loss": float(erm_base["test_hosp_2_loss_mean"] + erm_guard["delta_loss"])},
            {"suite": "Camelyon17 ERM", "point": "Fixed P95", "acc": float(fixed[fixed["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(fixed[fixed["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_loss_mean"].iloc[0])},
        ]
    )

    selected = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv")
    guard = pd.read_csv(ROOT / "artifacts/metrics/camelyon_finetune_p95_selector_summary_trueval_20260329.csv")
    ft_base = selected[selected["regime"] == "rcgdro_finetune"].iloc[0]
    ft_guard = guard[guard["selection_policy"] == "guardrail"].iloc[0]
    rows.extend(
        [
            {"suite": "Camelyon17 Finetune", "point": "Baseline", "acc": float(ft_base["test_hosp_2_acc_mean"]), "loss": float(ft_base["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 Finetune", "point": "Proxy-only P95", "acc": float(selected[selected["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(selected[selected["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_loss_mean"].iloc[0])},
            {"suite": "Camelyon17 Finetune", "point": "Guardrail P95", "acc": float(ft_base["test_hosp_2_acc_mean"] + ft_guard["delta_acc"]), "loss": float(ft_base["test_hosp_2_loss_mean"] + ft_guard["delta_loss"])},
            {"suite": "Camelyon17 Finetune", "point": "Fixed P95", "acc": float(fixed[fixed["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(fixed[fixed["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_loss_mean"].iloc[0])},
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    divergence = load_divergence_points()
    colors = {
        "Baseline": "#111827",
        "Proxy-only P95": "#D55E00",
        "Guardrail P95": "#009E73",
        "Fixed P95": "#6b7280",
    }
    markers_map = {"Baseline": "o", "Proxy-only P95": "X", "Guardrail P95": "P", "Fixed P95": "D"}
    point_offsets = {
        ("Camelyon17 ERM", "Baseline"): (-36, -16),
        ("Camelyon17 ERM", "Proxy-only P95"): (10, -10),
        ("Camelyon17 ERM", "Guardrail P95"): (10, 8),
        ("Camelyon17 ERM", "Fixed P95"): (-58, 12),
        ("Camelyon17 Finetune", "Baseline"): (-36, -16),
        ("Camelyon17 Finetune", "Proxy-only P95"): (10, -10),
        ("Camelyon17 Finetune", "Guardrail P95"): (-52, 14),
        ("Camelyon17 Finetune", "Fixed P95"): (-58, 12),
    }
    point_sizes = {key: 155 for key in markers_map}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=190)
    for ax, suite in zip(axes, ["Camelyon17 ERM", "Camelyon17 Finetune"]):
        sub = divergence[divergence["suite"] == suite].copy()
        baseline = sub[sub["point"] == "Baseline"].iloc[0]
        sub["delta_acc"] = sub["acc"] - float(baseline["acc"])
        sub["delta_loss"] = sub["loss"] - float(baseline["loss"])
        ax.set_facecolor("#fcfcfc")
        for _, row in sub.iterrows():
            ax.scatter(
                row["delta_acc"],
                row["delta_loss"],
                s=point_sizes[row["point"]],
                c=colors[row["point"]],
                marker=markers_map[row["point"]],
                edgecolors="white",
                linewidths=0.9,
                zorder=4,
            )
            label_map = {
                "Baseline": "Base",
                "Proxy-only P95": "Proxy-selected",
                "Guardrail P95": "Guardrail",
                "Fixed P95": "Matched horizon",
            }
            label = label_map[row["point"]]
            dx, dy = point_offsets[(suite, row["point"])]
            ax.annotate(
                label,
                (row["delta_acc"], row["delta_loss"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.3,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=colors[row["point"]], lw=0.4, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=colors[row["point"]], shrinkA=2, shrinkB=2),
            )
        for target, color, style in [
            ("Proxy-only P95", colors["Proxy-only P95"], "-"),
            ("Guardrail P95", colors["Guardrail P95"], "-"),
            ("Fixed P95", colors["Fixed P95"], "--"),
        ]:
            row = sub[sub["point"] == target].iloc[0]
            ax.annotate(
                "",
                xy=(row["delta_acc"], row["delta_loss"]),
                xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.6, linestyle=style, color=color),
            )
        ax.axhline(0.0, color="#9ca3af", lw=0.8, alpha=0.65, zorder=1)
        ax.axvline(0.0, color="#9ca3af", lw=0.8, alpha=0.65, zorder=1)
        ax.set_title(suite, fontsize=11.2, fontweight="bold")
        ax.set_xlabel("$\\Delta$ held-out accuracy vs. baseline $\\uparrow$", fontsize=10)
        ax.tick_params(labelsize=8.8)
        ax.grid(alpha=0.22)
        ax.text(
            0.98,
            0.03,
            "better = right and down",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.8,
            color="#4b5563",
        )

    axes[0].set_ylabel("$\\Delta$ held-out loss vs. baseline $\\downarrow$", fontsize=10)
    fig.suptitle("Proxy-selected reporting can move right on accuracy and up on loss", fontsize=12.4, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.94), w_pad=1.0)

    out_dir = PAPER / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    divergence.to_csv(out_dir / "fig_main_divergence_ed_points.csv", index=False)
    fig.savefig(out_dir / "fig_main_divergence_ed.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_main_divergence_ed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_dir / 'fig_main_divergence_ed.pdf'}")


if __name__ == "__main__":
    main()
