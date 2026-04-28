import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_pdf", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    plt.style.use("default")

    regime_map = {
        "rcgdro": ("rcgdro", "#1f77b4"),
        "rcgdro_softclip_p95_a10_cam": ("p95", "#d62728"),
        "rcgdro_softclip_p97_a10_cam": ("p97", "#ff7f0e"),
        "rcgdro_softclip_p99_a10_cam": ("p99", "#2ca02c"),
    }
    split_order = ["id", "ood"]

    sub = df[df["regime"].isin(regime_map.keys()) & df["split"].isin(split_order)].copy()
    if sub.empty:
        raise ValueError("No matching rows found for expected Camelyon regimes/splits.")

    # Ensure numeric ordering for deciles.
    sub["decile"] = pd.to_numeric(sub["decile"], errors="coerce")
    sub = sub.dropna(subset=["decile"]).copy()
    sub["decile"] = sub["decile"].astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.8), sharex=True, dpi=180)
    fig.patch.set_facecolor("white")
    handles = []
    labels = []

    for ax, split in zip(axes, split_order):
        split_df = sub[sub["split"] == split]
        for regime, (label, color) in regime_map.items():
            r = split_df[split_df["regime"] == regime].sort_values("decile")
            if r.empty:
                continue
            h = ax.errorbar(
                r["decile"],
                r["clip_rate_mean"],
                yerr=r["clip_rate_ci95"],
                marker="o",
                markersize=3.8,
                linewidth=1.8,
                capsize=2.5,
                color=color,
                label=label,
            )
            if label not in labels:
                handles.append(h)
                labels.append(label)

        ax.set_ylim(-0.01, 1.02)
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.set_ylabel("Clip rate")
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_title("ID hospitals" if split == "id" else "OOD hospital", fontsize=10)

    axes[-1].set_xlabel("Difficulty decile (teacher anchor)")
    axes[0].legend(handles, labels, loc="upper left", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.18)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
