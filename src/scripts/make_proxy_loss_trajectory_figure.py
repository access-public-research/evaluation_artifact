from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper" / "neurips2026_selection_risk"
PROXY_FAMILY = "conf_teacher_wpl"


@dataclass(frozen=True)
class SuiteSpec:
    panel_title: str
    suite_label: str
    phase0_csv: Path
    guardrail_csv: Path
    baseline_regime: str
    target_regime: str


SPECS = [
    SuiteSpec(
        panel_title="Camelyon17 ERM",
        suite_label="camelyon_erm_p95",
        phase0_csv=ROOT / "artifacts" / "metrics" / "camelyon17_resnet50_phase0_val_metrics_v11erm_softclip_cam_10s_20260228.csv",
        guardrail_csv=ROOT / "artifacts" / "metrics" / "guardrail_merged_rows_camelyon_erm_p95_ratio125_20260326.csv",
        baseline_regime="erm",
        target_regime="erm_softclip_p95_a10_cam",
    ),
    SuiteSpec(
        panel_title="Camelyon17 Finetune",
        suite_label="camelyon_finetune_p95",
        phase0_csv=ROOT / "artifacts" / "metrics" / "camelyon17_resnet50_phase0_val_metrics_finetune_cam_scivalid10s_20260326.csv",
        guardrail_csv=ROOT / "artifacts" / "metrics" / "guardrail_merged_rows_camelyon_finetune_p95_ratio125_20260326.csv",
        baseline_regime="rcgdro_finetune",
        target_regime="rcgdro_softclip_p95_a10_cam_finetune",
    ),
]


def _load_curve(spec: SuiteSpec, regime: str) -> pd.DataFrame:
    df = pd.read_csv(spec.phase0_csv)
    df = df[(df["regime"] == regime) & (df["family"] == PROXY_FAMILY)].copy()
    if df.empty:
        raise FileNotFoundError(f"No phase0 rows for regime={regime} in {spec.phase0_csv}")
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
    clip_col = pd.to_numeric(curve["proxy_worst_loss_clip_min"], errors="coerce")
    if "clip" in regime and clip_col.notna().any():
        curve["proxy_metric"] = clip_col
    else:
        curve["proxy_metric"] = pd.to_numeric(curve["proxy_worst_loss_min"], errors="coerce")
    curve["panel"] = spec.panel_title
    curve["regime"] = regime
    curve["trajectory_type"] = "baseline" if regime == spec.baseline_regime else "distorted"
    return curve[["panel", "regime", "trajectory_type", "epoch", "val_overall_loss", "proxy_metric"]]


def _load_markers(spec: SuiteSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.guardrail_csv)
    df = df[df["suite"] == spec.suite_label].copy()
    if df.empty:
        raise FileNotFoundError(f"No guardrail rows for suite={spec.suite_label} in {spec.guardrail_csv}")
    policy_map = {
        "baseline": "Baseline selected",
        "proxy_only": "P95 proxy-selected",
        "guardrail": "P95 guardrail",
    }
    markers = []
    for selection_policy, label in policy_map.items():
        sub = df[df["selection_policy"] == selection_policy].copy()
        if sub.empty:
            continue
        markers.append(
            {
                "panel": spec.panel_title,
                "point": label,
                "val_overall_loss": float(pd.to_numeric(sub["chosen_val_overall_loss"], errors="coerce").mean()),
                "proxy_metric": float(pd.to_numeric(sub["chosen_proxy_metric"], errors="coerce").mean()),
                "epoch_mean": float(pd.to_numeric(sub["epoch"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(markers)


def main() -> None:
    curve_frames = []
    marker_frames = []
    for spec in SPECS:
        curve_frames.append(_load_curve(spec, spec.baseline_regime))
        curve_frames.append(_load_curve(spec, spec.target_regime))
        marker_frames.append(_load_markers(spec))

    curves = pd.concat(curve_frames, ignore_index=True)
    markers = pd.concat(marker_frames, ignore_index=True)

    colors = {
        "baseline_line": "#111827",
        "distorted_line": "#4b5563",
        "Baseline selected": "#111827",
        "P95 proxy-selected": "#4b5563",
        "P95 guardrail": "#9ca3af",
    }
    marker_offsets = {
        ("Camelyon17 ERM", "Baseline selected"): (-32, -10),
        ("Camelyon17 ERM", "P95 proxy-selected"): (8, -10),
        ("Camelyon17 ERM", "P95 guardrail"): (8, 8),
        ("Camelyon17 Finetune", "Baseline selected"): (-32, -10),
        ("Camelyon17 Finetune", "P95 proxy-selected"): (8, -10),
        ("Camelyon17 Finetune", "P95 guardrail"): (-34, 8),
    }
    panel_order = [spec.panel_title for spec in SPECS]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=180)
    shared_scatter = None
    for ax, panel in zip(axes, panel_order):
        panel_curves = curves[curves["panel"] == panel].copy()
        base = panel_curves[panel_curves["trajectory_type"] == "baseline"].sort_values("epoch")
        dist = panel_curves[panel_curves["trajectory_type"] == "distorted"].sort_values("epoch")
        panel_markers = markers[markers["panel"] == panel].copy()

        ax.plot(
            base["val_overall_loss"],
            base["proxy_metric"],
            color=colors["baseline_line"],
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label="Baseline trajectory",
        )
        ax.scatter(
            base["val_overall_loss"],
            base["proxy_metric"],
            c="#d1d5db",
            s=26,
            zorder=2,
        )

        ax.plot(
            dist["val_overall_loss"],
            dist["proxy_metric"],
            color=colors["distorted_line"],
            linewidth=2.2,
            alpha=0.85,
            label="P95 trajectory",
            zorder=3,
        )
        shared_scatter = ax.scatter(
            dist["val_overall_loss"],
            dist["proxy_metric"],
            c=dist["epoch"],
            cmap="Greys",
            s=42,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )

        # Add small directional arrows so the time flow into the hazard region is visible on a skim.
        if len(dist) >= 6:
            arrow_idx = [max(0, len(dist) // 3), max(1, (2 * len(dist)) // 3)]
            for idx in arrow_idx:
                start = dist.iloc[idx - 1]
                end = dist.iloc[idx]
                ax.annotate(
                    "",
                    xy=(end["val_overall_loss"], end["proxy_metric"]),
                    xytext=(start["val_overall_loss"], start["proxy_metric"]),
                    arrowprops=dict(arrowstyle="-|>", color=colors["distorted_line"], lw=1.4, shrinkA=0, shrinkB=0),
                    zorder=5,
                )

        if not panel_markers.empty and "Baseline selected" in set(panel_markers["point"]):
            base_marker = panel_markers[panel_markers["point"] == "Baseline selected"].iloc[0]
            ax.axvline(base_marker["val_overall_loss"], color="#cbd5e1", linewidth=1.0, linestyle=":")
            ax.axhline(base_marker["proxy_metric"], color="#cbd5e1", linewidth=1.0, linestyle=":")

        for _, row in panel_markers.iterrows():
            marker = "o" if row["point"] == "Baseline selected" else ("X" if row["point"] == "P95 proxy-selected" else "P")
            size = 110 if row["point"] == "Baseline selected" else (210 if row["point"] == "P95 proxy-selected" else 150)
            ax.scatter(
                row["val_overall_loss"],
                row["proxy_metric"],
                marker=marker,
                s=size,
                c=colors[row["point"]],
                edgecolors="#111827" if row["point"] == "P95 proxy-selected" else "white",
                linewidths=1.4 if row["point"] == "P95 proxy-selected" else 0.8,
                zorder=7 if row["point"] == "P95 proxy-selected" else 6,
            )
            dx, dy = marker_offsets[(panel, row["point"])]
            ax.annotate(
                row["point"].replace(" selected", "").replace("proxy-", ""),
                (row["val_overall_loss"], row["proxy_metric"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=colors[row["point"]], shrinkA=2, shrinkB=2),
            )

        proxy_sel = panel_markers[panel_markers["point"] == "P95 proxy-selected"]
        if not proxy_sel.empty:
            row = proxy_sel.iloc[0]
            ax.annotate(
                "selected by proxy",
                (row["val_overall_loss"], row["proxy_metric"]),
                xytext=(10, 16),
                textcoords="offset points",
                fontsize=8.1,
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#111827", alpha=0.92),
            )

        ax.set_title(panel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Standard validation loss", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.22, zorder=0)

    axes[0].set_ylabel("Selection proxy", fontsize=10)
    cbar = fig.colorbar(shared_scatter, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("Epoch", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle("Proxy improvement can detach from standard validation loss", fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()

    out_dir = PAPER_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_csv = out_dir / "proxy_loss_trajectory_curves.csv"
    markers_csv = out_dir / "proxy_loss_trajectory_markers.csv"
    pdf = out_dir / "proxy_loss_trajectory.pdf"
    png = out_dir / "proxy_loss_trajectory.png"
    curves.to_csv(curves_csv, index=False)
    markers.to_csv(markers_csv, index=False)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"[ok] wrote {curves_csv}")
    print(f"[ok] wrote {markers_csv}")
    print(f"[ok] wrote {pdf}")
    print(f"[ok] wrote {png}")


if __name__ == "__main__":
    main()
