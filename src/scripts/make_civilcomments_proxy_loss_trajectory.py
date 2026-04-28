from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper" / "neurips2026_selection_risk"
PHASE0_CSV = ROOT / "artifacts" / "metrics" / "civilcomments_distilbert-base-uncased_phase0_val_metrics_civilcomments_erm_softclip_10s_20260328.csv"
GUARDRAIL_CSV = ROOT / "artifacts" / "metrics" / "guardrail_merged_rows_civilcomments_erm_p95_ratio125_20260328.csv"
PROXY_FAMILY = "global_hash"
BASELINE_REGIME = "erm"
TARGET_REGIME = "erm_softclip_p95_a10"


def _load_curve(regime: str) -> pd.DataFrame:
    df = pd.read_csv(PHASE0_CSV)
    df = df[(df["regime"] == regime) & (df["family"] == PROXY_FAMILY)].copy()
    if df.empty:
        raise FileNotFoundError(f"No phase0 rows for regime={regime} in {PHASE0_CSV}")
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
    curve["trajectory_type"] = "baseline" if regime == BASELINE_REGIME else "distorted"
    curve["regime"] = regime
    return curve[["regime", "trajectory_type", "epoch", "val_overall_loss", "proxy_metric"]]


def _load_markers() -> pd.DataFrame:
    df = pd.read_csv(GUARDRAIL_CSV)
    df = df[df["suite"] == "civilcomments_erm_p95"].copy()
    if df.empty:
        raise FileNotFoundError(f"No guardrail rows found in {GUARDRAIL_CSV}")
    policy_map = {
        "baseline": "Baseline selected",
        "proxy_only": "P95 proxy-selected",
        "guardrail": "P95 guardrail",
    }
    rows = []
    for policy, label in policy_map.items():
        sub = df[df["selection_policy"] == policy].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "point": label,
                "val_overall_loss": float(pd.to_numeric(sub["chosen_val_overall_loss"], errors="coerce").mean()),
                "proxy_metric": float(pd.to_numeric(sub["chosen_proxy_metric"], errors="coerce").mean()),
                "epoch_mean": float(pd.to_numeric(sub["epoch"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    baseline = _load_curve(BASELINE_REGIME)
    distorted = _load_curve(TARGET_REGIME)
    markers = _load_markers()

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.0), dpi=180)
    ax.plot(
        baseline["val_overall_loss"],
        baseline["proxy_metric"],
        color="#6b7280",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
        label="Baseline trajectory",
    )
    ax.scatter(baseline["val_overall_loss"], baseline["proxy_metric"], c="#9ca3af", s=28, zorder=2)

    sc = ax.scatter(
        distorted["val_overall_loss"],
        distorted["proxy_metric"],
        c=distorted["epoch"],
        cmap="viridis",
        s=42,
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
        label="P95 trajectory",
    )
    ax.plot(
        distorted["val_overall_loss"],
        distorted["proxy_metric"],
        color="#dc2626",
        linewidth=2.1,
        alpha=0.85,
        zorder=3,
    )

    colors = {
        "Baseline selected": "#111827",
        "P95 proxy-selected": "#dc2626",
        "P95 guardrail": "#059669",
    }
    for _, row in markers.iterrows():
        marker = "o" if row["point"] == "Baseline selected" else ("X" if row["point"] == "P95 proxy-selected" else "P")
        size = 140 if row["point"] != "Baseline selected" else 110
        ax.scatter(
            row["val_overall_loss"],
            row["proxy_metric"],
            marker=marker,
            s=size,
            c=colors[row["point"]],
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
        ax.annotate(
            row["point"].replace(" selected", "").replace("proxy-", ""),
            (row["val_overall_loss"], row["proxy_metric"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("CivilComments: proxy detaches from standard loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Standard validation loss")
    ax.set_ylabel("Selection proxy")
    ax.grid(alpha=0.22, zorder=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Epoch")
    fig.tight_layout()

    out_dir = PAPER_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_csv = out_dir / "civilcomments_proxy_loss_trajectory_curves.csv"
    markers_csv = out_dir / "civilcomments_proxy_loss_trajectory_markers.csv"
    pdf = out_dir / "civilcomments_proxy_loss_trajectory.pdf"
    png = out_dir / "civilcomments_proxy_loss_trajectory.png"
    pd.concat([baseline, distorted], ignore_index=True).to_csv(curves_csv, index=False)
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
