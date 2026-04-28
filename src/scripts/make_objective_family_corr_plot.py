import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def _panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    marker: str = "o",
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    r = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 else float("nan")

    ax.scatter(x, y, s=24, alpha=0.78, color=color, edgecolor="none", marker=marker)
    if x.size >= 2:
        m, b = _fit_line(x, y)
        xx = np.linspace(float(x.min()), float(x.max()), 120)
        ax.plot(xx, m * xx + b, color="black", lw=1.2)
    ax.axhline(0.0, color="gray", ls="--", lw=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(
        0.03,
        0.96,
        f"r = {r:+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    return r


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--softclip_rows",
        nargs="+",
        default=[
            "replication_rcg/artifacts/metrics/celeba_tail_distortion_rows_v7confclip_head_20260227.csv",
            "replication_rcg/artifacts/metrics/camelyon17_tail_distortion_rows_cam_softclip_a10_p99_20260227.csv",
        ],
        help="One or more softclip rows CSVs to pool.",
    )
    ap.add_argument(
        "--labelsmooth_rows",
        default="replication_rcg/artifacts/metrics/camelyon17_cam_labelsmooth_n10_20260304_distortion_rows_20260304.csv",
    )
    ap.add_argument(
        "--focal_rows",
        default="replication_rcg/artifacts/metrics/camelyon17_cam_focal_n10_20260304_distortion_rows_20260304.csv",
    )
    ap.add_argument(
        "--softclip_label",
        default="SoftClip (pooled head-only)",
    )
    ap.add_argument(
        "--out_png",
        default="replication_rcg/artifacts/metrics/objective_family_distortion_corr_20260305.png",
    )
    ap.add_argument(
        "--out_csv",
        default="replication_rcg/artifacts/metrics/objective_family_distortion_corr_20260305.csv",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    soft_frames: List[pd.DataFrame] = []
    for p in args.softclip_rows:
        d = _read(p).copy()
        d["x"] = d["distortion_mass_selected"]
        d["y"] = d["tail_delta_vs_baseline"]
        d["family"] = args.softclip_label
        soft_frames.append(d[["x", "y", "family"]])
    soft = pd.concat(soft_frames, ignore_index=True)

    ls = _read(args.labelsmooth_rows).copy()
    ls = ls.drop_duplicates(subset=["regime", "seed", "epoch"]).reset_index(drop=True)
    ls["x"] = ls["distortion_mass"]
    ls["y"] = ls["tail_delta_vs_erm"]
    ls["family"] = "Label smoothing (Camelyon ERM)"
    ls = ls[["x", "y", "family"]]

    fc = _read(args.focal_rows).copy()
    fc = fc.drop_duplicates(subset=["regime", "seed", "epoch"]).reset_index(drop=True)
    fc["x"] = fc["distortion_mass"]
    fc["y"] = fc["tail_delta_vs_erm"]
    fc["family"] = "Focal (Camelyon ERM)"
    fc = fc[["x", "y", "family"]]

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.9), constrained_layout=True)
    r_soft = _panel(
        axs[0],
        soft["x"].to_numpy(),
        soft["y"].to_numpy(),
        "SoftClip",
        "DistortionMass",
        "Tail delta",
        color="tab:red",
    )
    r_ls = _panel(
        axs[1],
        ls["x"].to_numpy(),
        ls["y"].to_numpy(),
        "Label Smoothing",
        "DistortionMass",
        "Tail delta",
        color="tab:green",
    )
    r_fc = _panel(
        axs[2],
        fc["x"].to_numpy(),
        fc["y"].to_numpy(),
        "Focal",
        "DistortionMass",
        "Tail delta",
        color="tab:blue",
    )
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {"family": args.softclip_label, "pearson_r": r_soft, "n": int(soft.shape[0])},
            {"family": "Label smoothing (Camelyon ERM)", "pearson_r": r_ls, "n": int(ls.shape[0])},
            {"family": "Focal (Camelyon ERM)", "pearson_r": r_fc, "n": int(fc.shape[0])},
        ]
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print("[objective-family-corr] wrote:")
    print(f" - {out_png}")
    print(f" - {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
