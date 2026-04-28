from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper" / "neurips2026_selection_risk"


def _load_celeba_points() -> pd.DataFrame:
    selected = pd.read_csv(ROOT / "artifacts/metrics/celeba_test_wg_selected_with_loss_summary_erm_softclip_celeba_10s_20260325.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/celeba_test_wg_fixed30_with_loss_summary_erm_softclip_celeba_10s_20260325.csv")

    rows = []
    sel_base = selected[selected["regime"] == "erm"].iloc[0]
    sel_p95 = selected[selected["regime"] == "erm_softclip_p95_a10"].iloc[0]
    fix_p95 = fixed[fixed["regime"] == "erm_softclip_p95_a10"].iloc[0]
    fix_base = fixed[fixed["regime"] == "erm"].iloc[0]
    rows.extend(
        [
            {"suite": "CelebA ERM", "point": "Baseline selected", "acc": float(sel_base["test_oracle_wg_acc_mean"]), "loss": float(sel_base["test_oracle_wg_loss_mean"])},
            {"suite": "CelebA ERM", "point": "P95 proxy-selected", "acc": float(sel_p95["test_oracle_wg_acc_mean"]), "loss": float(sel_p95["test_oracle_wg_loss_mean"])},
            {"suite": "CelebA ERM", "point": "P95 fixed30", "acc": float(fix_p95["test_oracle_wg_acc_mean"]), "loss": float(fix_p95["test_oracle_wg_loss_mean"])},
            {"suite": "CelebA ERM", "point": "Baseline fixed30", "acc": float(fix_base["test_oracle_wg_acc_mean"]), "loss": float(fix_base["test_oracle_wg_loss_mean"])},
        ]
    )
    guard = ROOT / "artifacts/metrics/guardrail_summary_celeba_erm_p95_ratio125_20260326.csv"
    if guard.exists():
        gdf = pd.read_csv(guard)
        grow = gdf[gdf["selection_policy"] == "guardrail"].iloc[0]
        rows.append(
            {
                "suite": "CelebA ERM",
                "point": "P95 guardrail",
                "acc": float(grow["test_oracle_wg_acc_mean"]),
                "loss": float(grow["test_oracle_wg_loss_mean"]),
            }
        )
    return pd.DataFrame(rows)


def _load_cam_erm_points() -> pd.DataFrame:
    selected = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv")
    rows = []
    sel_base = selected[selected["regime"] == "erm"].iloc[0]
    sel_p95 = selected[selected["regime"] == "erm_softclip_p95_a10_cam"].iloc[0]
    fix_p95 = fixed[fixed["regime"] == "erm_softclip_p95_a10_cam"].iloc[0]
    fix_base = fixed[fixed["regime"] == "erm"].iloc[0]
    rows.extend(
        [
            {"suite": "Camelyon17 ERM", "point": "Baseline selected", "acc": float(sel_base["test_hosp_2_acc_mean"]), "loss": float(sel_base["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 ERM", "point": "P95 proxy-selected", "acc": float(sel_p95["test_hosp_2_acc_mean"]), "loss": float(sel_p95["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 ERM", "point": "P95 fixed30", "acc": float(fix_p95["test_hosp_2_acc_mean"]), "loss": float(fix_p95["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 ERM", "point": "Baseline fixed30", "acc": float(fix_base["test_hosp_2_acc_mean"]), "loss": float(fix_base["test_hosp_2_loss_mean"])},
        ]
    )
    guard = ROOT / "artifacts/metrics/guardrail_summary_camelyon_erm_p95_ratio125_20260326.csv"
    if guard.exists():
        gdf = pd.read_csv(guard)
        grow = gdf[gdf["selection_policy"] == "guardrail"].iloc[0]
        rows.append(
            {
                "suite": "Camelyon17 ERM",
                "point": "P95 guardrail",
                "acc": float(grow["test_hosp_2_acc_mean"]),
                "loss": float(grow["test_hosp_2_loss_mean"]),
            }
        )
    return pd.DataFrame(rows)


def _load_cam_finetune_points() -> pd.DataFrame:
    selected = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv")
    fixed = pd.read_csv(ROOT / "artifacts/metrics/camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv")
    rows = []
    sel_base = selected[selected["regime"] == "rcgdro_finetune"].iloc[0]
    sel_p95 = selected[selected["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"].iloc[0]
    fix_p95 = fixed[fixed["regime"] == "rcgdro_softclip_p95_a10_cam_finetune"].iloc[0]
    fix_base = fixed[fixed["regime"] == "rcgdro_finetune"].iloc[0]
    rows.extend(
        [
            {"suite": "Camelyon17 Finetune", "point": "Baseline selected", "acc": float(sel_base["test_hosp_2_acc_mean"]), "loss": float(sel_base["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 Finetune", "point": "P95 proxy-selected", "acc": float(sel_p95["test_hosp_2_acc_mean"]), "loss": float(sel_p95["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 Finetune", "point": "P95 fixed10", "acc": float(fix_p95["test_hosp_2_acc_mean"]), "loss": float(fix_p95["test_hosp_2_loss_mean"])},
            {"suite": "Camelyon17 Finetune", "point": "Baseline fixed10", "acc": float(fix_base["test_hosp_2_acc_mean"]), "loss": float(fix_base["test_hosp_2_loss_mean"])},
        ]
    )
    guard = ROOT / "artifacts/metrics/guardrail_summary_camelyon_finetune_p95_ratio125_20260326.csv"
    if guard.exists():
        gdf = pd.read_csv(guard)
        grow = gdf[gdf["selection_policy"] == "guardrail"].iloc[0]
        rows.append(
            {
                "suite": "Camelyon17 Finetune",
                "point": "P95 guardrail",
                "acc": float(grow["test_hosp_2_acc_mean"]),
                "loss": float(grow["test_hosp_2_loss_mean"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frames = [_load_celeba_points(), _load_cam_erm_points(), _load_cam_finetune_points()]
    df = pd.concat(frames, ignore_index=True)

    colors = {
        "Baseline selected": "#1f2937",
        "Baseline fixed30": "#6b7280",
        "Baseline fixed10": "#6b7280",
        "P95 proxy-selected": "#dc2626",
        "P95 fixed30": "#2563eb",
        "P95 fixed10": "#2563eb",
        "P95 guardrail": "#059669",
    }
    markers = {
        "Baseline selected": "o",
        "Baseline fixed30": "o",
        "Baseline fixed10": "o",
        "P95 proxy-selected": "X",
        "P95 fixed30": "D",
        "P95 fixed10": "D",
        "P95 guardrail": "P",
    }

    titles = ["CelebA ERM", "Camelyon17 ERM", "Camelyon17 Finetune"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, title in zip(axes, titles):
        sub = df[df["suite"] == title].copy()
        for _, row in sub.iterrows():
            ax.scatter(
                row["acc"],
                row["loss"],
                s=130,
                c=colors[row["point"]],
                marker=markers[row["point"]],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            label = row["point"].replace("Baseline ", "Base ").replace("proxy-selected", "selected")
            ax.annotate(label, (row["acc"], row["loss"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

        # Emphasize the divergence trajectory.
        if "Baseline selected" in set(sub["point"]) and "P95 proxy-selected" in set(sub["point"]):
            a = sub[sub["point"] == "Baseline selected"].iloc[0]
            b = sub[sub["point"] == "P95 proxy-selected"].iloc[0]
            ax.annotate("", xy=(b["acc"], b["loss"]), xytext=(a["acc"], a["loss"]), arrowprops=dict(arrowstyle="->", lw=1.8, color="#dc2626"))
        fixed_name = "P95 fixed30" if "P95 fixed30" in set(sub["point"]) else "P95 fixed10"
        base_fixed_name = "Baseline fixed30" if "Baseline fixed30" in set(sub["point"]) else "Baseline fixed10"
        if fixed_name in set(sub["point"]) and base_fixed_name in set(sub["point"]):
            a = sub[sub["point"] == base_fixed_name].iloc[0]
            b = sub[sub["point"] == fixed_name].iloc[0]
            ax.annotate("", xy=(b["acc"], b["loss"]), xytext=(a["acc"], a["loss"]), arrowprops=dict(arrowstyle="->", lw=1.4, linestyle="--", color="#2563eb"))
        if "P95 guardrail" in set(sub["point"]) and "Baseline selected" in set(sub["point"]):
            a = sub[sub["point"] == "Baseline selected"].iloc[0]
            b = sub[sub["point"] == "P95 guardrail"].iloc[0]
            ax.annotate("", xy=(b["acc"], b["loss"]), xytext=(a["acc"], a["loss"]), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Held-out accuracy")
        ax.grid(alpha=0.2, zorder=0)
    axes[0].set_ylabel("Held-out loss")
    fig.suptitle("Accuracy-loss divergence under proxy-only selection", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_dir = PAPER_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "accuracy_loss_divergence.png"
    pdf = out_dir / "accuracy_loss_divergence.pdf"
    csv = out_dir / "accuracy_loss_divergence_points.csv"
    df.to_csv(csv, index=False)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {csv}")
    print(f"[ok] wrote {png}")
    print(f"[ok] wrote {pdf}")


if __name__ == "__main__":
    main()
