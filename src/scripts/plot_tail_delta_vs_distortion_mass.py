import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_csvs", required=True, help="Comma-separated rows CSVs.")
    ap.add_argument("--out_name", default="tail_delta_vs_distortion_mass_head_20260227")
    args = ap.parse_args()

    paths: List[Path] = [Path(p.strip()) for p in str(args.rows_csvs).split(",") if p.strip()]
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    x = df["distortion_mass_selected"].to_numpy(dtype=float)
    y = df["tail_delta_vs_baseline"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    xg = x[m]
    yg = y[m]

    colors = {"celeba": "#1f77b4", "waterbirds": "#ff7f0e", "camelyon17": "#2ca02c"}
    labels = {"celeba": "CelebA", "waterbirds": "Waterbirds", "camelyon17": "Camelyon17"}

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for ds, grp in df.groupby("dataset"):
        gx = grp["distortion_mass_selected"].to_numpy(dtype=float)
        gy = grp["tail_delta_vs_baseline"].to_numpy(dtype=float)
        mm = np.isfinite(gx) & np.isfinite(gy)
        ax.scatter(
            gx[mm],
            gy[mm],
            s=22,
            alpha=0.75,
            color=colors.get(str(ds), "#666666"),
            label=labels.get(str(ds), str(ds)),
            edgecolors="none",
        )

    if xg.size >= 3:
        b1, b0 = np.polyfit(xg, yg, 1)
        xs = np.linspace(float(np.min(xg)), float(np.max(xg)), 200)
        ys = b1 * xs + b0
        ax.plot(xs, ys, color="#222222", linewidth=1.6, label="Pooled linear fit")
        r = _corr(xg, yg)
        ax.text(
            0.03,
            0.97,
            f"Pooled Pearson r = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#bbbbbb"),
        )

    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Distortion Mass = (1-alpha)E[(L-tau)+]")
    ax.set_ylabel("Tail delta vs proper baseline")
    ax.set_title("Tail Inflation vs Distortion Mass (Head-only)")
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()

    out_dir = Path("figures")
    ensure_dir(out_dir)
    stem = str(args.out_name).strip()
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[distortion-plot] wrote {png}")
    print(f"[distortion-plot] wrote {pdf}")


if __name__ == "__main__":
    main()

