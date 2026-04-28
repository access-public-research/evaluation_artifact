import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


REGIME_ORDER = [
    ("rcgdro", "rcgdro"),
    ("rcgdro_softclip_p95_a10_cam", "P95"),
    ("rcgdro_softclip_p97_a10_cam", "P97"),
    ("rcgdro_softclip_p99_a10_cam", "P99"),
]


def _mean_ci(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(ci95_mean(arr))


def _scatter_plot(
    x: np.ndarray,
    xerr: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    labels: list[str],
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(4.6, 3.5), dpi=180)
    for xi, xe, yi, ye, label in zip(x, xerr, y, yerr, labels):
        plt.errorbar(xi, yi, xerr=xe, yerr=ye, fmt="o", capsize=3, label=label)
    plt.xlabel("Fraction clipped (val)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--effect_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv",
    )
    ap.add_argument(
        "--domain_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_resnet50_domain_acc_cam_softclip_a10_p99_20260207.csv",
    )
    ap.add_argument(
        "--out_proxy_png",
        default="replication_rcg/figures/camelyon17_properness_proxy_cam_softclip_a10_p99_20260207.png",
    )
    ap.add_argument(
        "--out_tail_png",
        default="replication_rcg/figures/camelyon17_properness_tail_cam_softclip_a10_p99_20260207.png",
    )
    ap.add_argument(
        "--out_perf_png",
        default="replication_rcg/figures/camelyon17_properness_test_hosp2_cam_softclip_a10_p99_20260217.png",
    )
    args = ap.parse_args()

    eff = pd.read_csv(args.effect_csv)
    dom = pd.read_csv(args.domain_csv)

    frac_mean = []
    frac_ci = []
    proxy_mean = []
    proxy_ci = []
    tail_mean = []
    tail_ci = []
    perf_mean = []
    perf_ci = []
    labels = []

    for regime, label in REGIME_ORDER:
        row = eff[eff["regime"] == regime].iloc[0]
        frac_mean.append(0.0 if regime == "rcgdro" else float(row["frac_clipped_val_mean"]))
        frac_ci.append(0.0 if regime == "rcgdro" else float(row["frac_clipped_val_ci"]))
        if regime == "rcgdro":
            proxy_mean.append(float(row["proxy_worst_loss_mean"]))
            proxy_ci.append(float(row["proxy_worst_loss_ci"]))
        else:
            proxy_mean.append(float(row["proxy_worst_loss_clip_mean"]))
            proxy_ci.append(float(row["proxy_worst_loss_clip_ci"]))
        tail_mean.append(float(row["tail_worst_cvar_mean"]))
        tail_ci.append(float(row["tail_worst_cvar_ci"]))
        dsub = pd.to_numeric(dom.loc[dom["regime"] == regime, "test_hosp_2_acc"], errors="coerce").to_numpy()
        dmean, dci = _mean_ci(dsub)
        perf_mean.append(dmean)
        perf_ci.append(dci)
        labels.append(label)

    out_proxy = Path(args.out_proxy_png)
    out_tail = Path(args.out_tail_png)
    out_perf = Path(args.out_perf_png)
    out_proxy.parent.mkdir(parents=True, exist_ok=True)
    out_tail.parent.mkdir(parents=True, exist_ok=True)
    out_perf.parent.mkdir(parents=True, exist_ok=True)

    _scatter_plot(
        np.asarray(frac_mean),
        np.asarray(frac_ci),
        np.asarray(proxy_mean),
        np.asarray(proxy_ci),
        labels,
        "Proxy worst loss (selected)",
        out_proxy,
    )
    _scatter_plot(
        np.asarray(frac_mean),
        np.asarray(frac_ci),
        np.asarray(tail_mean),
        np.asarray(tail_ci),
        labels,
        "Tail worst-cell CVaR",
        out_tail,
    )
    _scatter_plot(
        np.asarray(frac_mean),
        np.asarray(frac_ci),
        np.asarray(perf_mean),
        np.asarray(perf_ci),
        labels,
        "test-hosp2 accuracy",
        out_perf,
    )

    print(f"[ok] wrote {out_proxy}")
    print(f"[ok] wrote {out_tail}")
    print(f"[ok] wrote {out_perf}")


if __name__ == "__main__":
    main()
