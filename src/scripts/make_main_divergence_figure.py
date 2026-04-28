from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "neurips2026_selection_risk"
PROXY_FAMILY = "conf_teacher_wpl"


@dataclass(frozen=True)
class TrajectorySpec:
    panel_title: str
    suite_label: str
    phase0_csv: Path
    guardrail_csv: Path
    baseline_regime: str
    target_regime: str


TRAJECTORY_SPECS = [
    TrajectorySpec(
        panel_title="Camelyon17 ERM",
        suite_label="camelyon_erm_p95",
        phase0_csv=ROOT / "artifacts" / "metrics" / "camelyon17_resnet50_phase0_val_metrics_v11erm_softclip_cam_10s_20260228.csv",
        guardrail_csv=ROOT / "artifacts" / "metrics" / "guardrail_merged_rows_camelyon_erm_p95_ratio125_20260326.csv",
        baseline_regime="erm",
        target_regime="erm_softclip_p95_a10_cam",
    ),
    TrajectorySpec(
        panel_title="Camelyon17 Finetune",
        suite_label="camelyon_finetune_p95",
        phase0_csv=ROOT / "artifacts" / "metrics" / "camelyon17_resnet50_phase0_val_metrics_finetune_cam_scivalid10s_20260326.csv",
        guardrail_csv=ROOT / "artifacts" / "metrics" / "guardrail_merged_rows_camelyon_finetune_p95_ratio125_20260326.csv",
        baseline_regime="rcgdro_finetune",
        target_regime="rcgdro_softclip_p95_a10_cam_finetune",
    ),
]


def _load_divergence_points() -> pd.DataFrame:
    rows = []

    selected = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv")
    guard = pd.read_csv(ROOT / "artifacts/metrics/guardrail_summary_camelyon_erm_p95_ratio125_20260326.csv")
    rows.extend(
        [
            {"suite": "Camelyon17 ERM", "point": "Baseline", "acc": float(selected[selected["regime"] == "erm"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(selected[selected["regime"] == "erm"]["test_hosp_2_loss_mean"].iloc[0]), "tail": 5.781140},
            {"suite": "Camelyon17 ERM", "point": "Proxy-only P95", "acc": float(selected[selected["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(selected[selected["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_loss_mean"].iloc[0]), "tail": 11.305657},
            {"suite": "Camelyon17 ERM", "point": "Guardrail P95", "acc": float(guard[guard["selection_policy"] == "guardrail"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(guard[guard["selection_policy"] == "guardrail"]["test_hosp_2_loss_mean"].iloc[0]), "tail": float(guard[guard["selection_policy"] == "guardrail"]["selected_tail_worst_cvar_mean"].iloc[0])},
            {"suite": "Camelyon17 ERM", "point": "Fixed P95", "acc": float(fixed[fixed["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(fixed[fixed["regime"] == "erm_softclip_p95_a10_cam"]["test_hosp_2_loss_mean"].iloc[0]), "tail": 11.215278},
        ]
    )

    selected = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv")
    guard = pd.read_csv(ROOT / "artifacts/metrics/guardrail_summary_camelyon_finetune_p95_ratio125_20260326.csv")
    rows.extend(
        [
            {"suite": "Camelyon17 Finetune", "point": "Baseline", "acc": float(selected[selected["regime"] == "rcgdro_finetune"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(selected[selected["regime"] == "rcgdro_finetune"]["test_hosp_2_loss_mean"].iloc[0]), "tail": 7.105897},
            {"suite": "Camelyon17 Finetune", "point": "Proxy-only P95", "acc": float(selected[selected["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(selected[selected["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_loss_mean"].iloc[0]), "tail": 12.553443},
            {"suite": "Camelyon17 Finetune", "point": "Guardrail P95", "acc": float(guard[guard["selection_policy"] == "guardrail"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(guard[guard["selection_policy"] == "guardrail"]["test_hosp_2_loss_mean"].iloc[0]), "tail": float(guard[guard["selection_policy"] == "guardrail"]["selected_tail_worst_cvar_mean"].iloc[0])},
            {"suite": "Camelyon17 Finetune", "point": "Fixed P95", "acc": float(fixed[fixed["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_acc_mean"].iloc[0]), "loss": float(fixed[fixed["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"]["test_hosp_2_loss_mean"].iloc[0]), "tail": 281.878738},
        ]
    )
    return pd.DataFrame(rows)


def _load_curve(spec: TrajectorySpec, regime: str) -> pd.DataFrame:
    df = pd.read_csv(spec.phase0_csv)
    df = df[(df["regime"] == regime) & (df["family"] == PROXY_FAMILY)].copy()
    curve = (
        df.groupby("epoch", as_index=False)
        .agg(
            {
                "val_overall_loss": "mean",
                "proxy_worst_loss_min": "mean",
                "proxy_worst_loss_clip_min": "mean",
            }
        )
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    if "clip" in regime and pd.to_numeric(curve["proxy_worst_loss_clip_min"], errors="coerce").notna().any():
        curve["proxy_metric"] = pd.to_numeric(curve["proxy_worst_loss_clip_min"], errors="coerce")
    else:
        curve["proxy_metric"] = pd.to_numeric(curve["proxy_worst_loss_min"], errors="coerce")
    curve["panel"] = spec.panel_title
    curve["trajectory_type"] = "baseline" if regime == spec.baseline_regime else "distorted"
    return curve[["panel", "trajectory_type", "epoch", "val_overall_loss", "proxy_metric"]]


def _load_markers(spec: TrajectorySpec) -> pd.DataFrame:
    df = pd.read_csv(spec.guardrail_csv)
    df = df[df["suite"] == spec.suite_label].copy()
    policy_map = {
        "baseline": "Baseline selected",
        "proxy_only": "P95 proxy-selected",
        "guardrail": "P95 guardrail",
    }
    rows = []
    for selection_policy, label in policy_map.items():
        sub = df[df["selection_policy"] == selection_policy].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "panel": spec.panel_title,
                "point": label,
                "val_overall_loss": float(pd.to_numeric(sub["chosen_val_overall_loss"], errors="coerce").mean()),
                "proxy_metric": float(pd.to_numeric(sub["chosen_proxy_metric"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    divergence = _load_divergence_points()
    curves = pd.concat(
        [
            _load_curve(spec, spec.baseline_regime)
            for spec in TRAJECTORY_SPECS
        ]
        + [
            _load_curve(spec, spec.target_regime)
            for spec in TRAJECTORY_SPECS
        ],
        ignore_index=True,
    )
    markers = pd.concat([_load_markers(spec) for spec in TRAJECTORY_SPECS], ignore_index=True)

    colors = {"Baseline": "#111827", "Proxy-only P95": "#4b5563", "Guardrail P95": "#9ca3af", "Fixed P95": "#d1d5db"}
    traj_colors = {
        "baseline_line": "#111827",
        "distorted_line": "#4b5563",
        "Baseline selected": "#111827",
        "P95 proxy-selected": "#4b5563",
        "P95 guardrail": "#9ca3af",
    }
    markers_map = {"Baseline": "o", "Proxy-only P95": "X", "Guardrail P95": "P", "Fixed P95": "D"}
    point_offsets = {
        ("Camelyon17 ERM", "Baseline"): (-26, -12),
        ("Camelyon17 ERM", "Proxy-only P95"): (8, -12),
        ("Camelyon17 ERM", "Guardrail P95"): (8, 8),
        ("Camelyon17 ERM", "Fixed P95"): (-28, 10),
        ("Camelyon17 Finetune", "Baseline"): (-26, -12),
        ("Camelyon17 Finetune", "Proxy-only P95"): (8, -12),
        ("Camelyon17 Finetune", "Guardrail P95"): (-36, 10),
        ("Camelyon17 Finetune", "Fixed P95"): (-28, 10),
    }
    traj_offsets = {
        ("Camelyon17 ERM", "Baseline selected"): (-28, -10),
        ("Camelyon17 ERM", "P95 proxy-selected"): (8, -10),
        ("Camelyon17 ERM", "P95 guardrail"): (8, 8),
        ("Camelyon17 Finetune", "Baseline selected"): (-28, -10),
        ("Camelyon17 Finetune", "P95 proxy-selected"): (8, -10),
        ("Camelyon17 Finetune", "P95 guardrail"): (-34, 8),
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 8.4),
        dpi=190,
        gridspec_kw={"height_ratios": [0.9, 1.3]},
    )

    # Top row: selected checkpoint divergence.
    for ax, suite in zip(axes[0], ["Camelyon17 ERM", "Camelyon17 Finetune"]):
        sub = divergence[divergence["suite"] == suite].copy()
        baseline = sub[sub["point"] == "Baseline"].iloc[0]
        for _, row in sub.iterrows():
            point_size = 85 + 8 * min(float(row["tail"]), 20.0)
            if row["point"] == "Fixed P95" and suite == "Camelyon17 Finetune":
                point_size = 220
            ax.scatter(
                row["acc"],
                row["loss"],
                s=point_size,
                c=colors[row["point"]],
                marker=markers_map[row["point"]],
                edgecolors="white",
                linewidths=0.9,
                zorder=4,
            )
            label = row["point"].replace("Proxy-only ", "Proxy ").replace("Guardrail ", "Guard ").replace("Baseline", "Base")
            dx, dy = point_offsets[(suite, row["point"])]
            ax.annotate(
                label,
                (row["acc"], row["loss"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=colors[row["point"]], shrinkA=2, shrinkB=2),
            )
        for target, color, style in [("Proxy-only P95", "#4b5563", "-"), ("Guardrail P95", "#9ca3af", "-"), ("Fixed P95", "#6b7280", "--")]:
            row = sub[sub["point"] == target].iloc[0]
            ax.annotate(
                "",
                xy=(row["acc"], row["loss"]),
                xytext=(baseline["acc"], baseline["loss"]),
                arrowprops=dict(arrowstyle="->", lw=1.6, linestyle=style, color=color),
            )
        ax.set_title(suite, fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Held-out accuracy", fontsize=10)
        ax.tick_params(labelsize=8.8)
        ax.grid(alpha=0.22)

    axes[0, 0].set_ylabel("Held-out loss", fontsize=10)

    # Bottom row: proxy versus standard-loss trajectories.
    for ax, spec in zip(axes[1], TRAJECTORY_SPECS):
        panel_curves = curves[curves["panel"] == spec.panel_title].copy()
        base = panel_curves[panel_curves["trajectory_type"] == "baseline"].sort_values("epoch")
        dist = panel_curves[panel_curves["trajectory_type"] == "distorted"].sort_values("epoch")
        panel_markers = markers[markers["panel"] == spec.panel_title].copy()

        ax.plot(base["val_overall_loss"], base["proxy_metric"], color=traj_colors["baseline_line"], linewidth=1.6, linestyle="--", alpha=0.9)
        ax.scatter(base["val_overall_loss"], base["proxy_metric"], c="#d1d5db", s=18, zorder=2)

        ax.plot(dist["val_overall_loss"], dist["proxy_metric"], color=traj_colors["distorted_line"], linewidth=2.0, alpha=0.85, zorder=3)
        ax.scatter(dist["val_overall_loss"], dist["proxy_metric"], c="#9ca3af", s=22, edgecolors="white", linewidths=0.35, zorder=4)

        if len(dist) >= 6:
            arrow_idx = [max(1, len(dist) // 3), max(2, (2 * len(dist)) // 3)]
            for idx in arrow_idx:
                start = dist.iloc[idx - 1]
                end = dist.iloc[idx]
                ax.annotate(
                    "",
                    xy=(end["val_overall_loss"], end["proxy_metric"]),
                    xytext=(start["val_overall_loss"], start["proxy_metric"]),
                    arrowprops=dict(arrowstyle="-|>", color=traj_colors["distorted_line"], lw=1.3, shrinkA=0, shrinkB=0),
                    zorder=5,
                )

        for _, row in panel_markers.iterrows():
            marker = "o" if row["point"] == "Baseline selected" else ("X" if row["point"] == "P95 proxy-selected" else "P")
            size = 85 if row["point"] == "Baseline selected" else (170 if row["point"] == "P95 proxy-selected" else 125)
            ax.scatter(
                row["val_overall_loss"],
                row["proxy_metric"],
                marker=marker,
                s=size,
                c=traj_colors[row["point"]],
                edgecolors="#111827" if row["point"] == "P95 proxy-selected" else "white",
                linewidths=1.2 if row["point"] == "P95 proxy-selected" else 0.8,
                zorder=7 if row["point"] == "P95 proxy-selected" else 6,
            )
            dx, dy = traj_offsets[(spec.panel_title, row["point"])]
            ax.annotate(
                row["point"].replace(" selected", "").replace("proxy-", ""),
                (row["val_overall_loss"], row["proxy_metric"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.1,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=traj_colors[row["point"]], shrinkA=2, shrinkB=2),
            )

        proxy_sel = panel_markers[panel_markers["point"] == "P95 proxy-selected"]
        if not proxy_sel.empty:
            row = proxy_sel.iloc[0]
            ax.annotate(
                "selected by proxy",
                (row["val_overall_loss"], row["proxy_metric"]),
                xytext=(8, 14),
                textcoords="offset points",
                fontsize=8.0,
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#111827", alpha=0.92),
            )

        ax.set_xlabel("Standard validation loss", fontsize=10)
        ax.tick_params(labelsize=8.8)
        ax.grid(alpha=0.22, zorder=0)

    axes[1, 0].set_ylabel("Selection proxy", fontsize=10)
    fig.text(0.015, 0.76, "Selected checkpoints", rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    fig.text(0.015, 0.255, "Training trajectories", rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle("Proxy-only selection improves its own metric while pulling reliability the wrong way", fontsize=13.0, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.04, 0.01, 1.0, 0.97), h_pad=1.2, w_pad=1.0)

    out_dir = PAPER / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    divergence.to_csv(out_dir / "fig_main_divergence_points.csv", index=False)
    curves.to_csv(out_dir / "fig_main_divergence_curves.csv", index=False)
    markers.to_csv(out_dir / "fig_main_divergence_markers.csv", index=False)
    fig.savefig(out_dir / "fig_main_divergence.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_main_divergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_dir / 'fig_main_divergence.pdf'}")
    print(f"[ok] wrote {out_dir / 'fig_main_divergence.png'}")


if __name__ == "__main__":
    main()
