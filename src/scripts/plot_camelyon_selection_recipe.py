import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _proxy(row: pd.Series, mode: str) -> float:
    mode = str(mode).strip().lower()
    if mode == "clip_aware":
        v = row.get("proxy_worst_loss_clip_mean")
        if pd.notna(v):
            return float(v)
        return float(row["proxy_worst_loss_mean"])
    if mode == "stationary_unclipped":
        return float(row["proxy_worst_loss_mean"])
    raise ValueError(f"Unknown selection_metric_mode={mode}")


def _test_mean(dom: pd.DataFrame, regime: str) -> float:
    x = pd.to_numeric(dom[dom["regime"] == regime]["test_hosp_2_acc"], errors="coerce")
    return float(x.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effect_csv", required=True)
    ap.add_argument("--domain_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument(
        "--selection_metric_mode",
        default="clip_aware",
        choices=["clip_aware", "stationary_unclipped"],
        help="Proxy metric shown on the x-axis. clip_aware is the within-family worked-example view.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.effect_csv)
    dom = pd.read_csv(args.domain_csv)
    order = [
        ("rcgdro", "rcgdro"),
        ("rcgdro_softclip_p95_a10_cam", "P95"),
        ("rcgdro_softclip_p97_a10_cam", "P97"),
        ("rcgdro_softclip_p99_a10_cam", "P99"),
    ]

    baseline = df[df["regime"] == "rcgdro"].iloc[0]
    proxy_b = _proxy(baseline, args.selection_metric_mode)
    tail_b = float(baseline["tail_worst_cvar_mean"])
    # Match manuscript arithmetic from displayed table values.
    proxy_thr = round(proxy_b, 3) * 1.05
    tail_thr = tail_b + 1.0

    x_vals, y_vals, labels, perf_txt = [], [], [], []
    x_err, y_err = [], []
    for regime, lbl in order:
        row = df[df["regime"] == regime].iloc[0]
        x_vals.append(_proxy(row, args.selection_metric_mode))
        y_vals.append(float(row["tail_worst_cvar_mean"]))
        labels.append(lbl)
        perf_txt.append(_test_mean(dom, regime))
        # Use proxy_worst_loss_clip_ci when present; baseline falls back to proxy_worst_loss_ci.
        pci = row.get("proxy_worst_loss_clip_ci")
        if pd.isna(pci):
            pci = row["proxy_worst_loss_ci"]
        x_err.append(float(pci))
        y_err.append(float(row["tail_worst_cvar_ci"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    # Admissible rectangle: proxy <= threshold AND tail <= threshold.
    ax.axvspan(min(x_vals) - 0.02, proxy_thr, ymin=0.0, ymax=1.0, color="#d9f2d9", alpha=0.25, lw=0)
    ax.axhspan(min(y_vals) - 0.5, tail_thr, xmin=0.0, xmax=1.0, color="#d9f2d9", alpha=0.25, lw=0)
    ax.axvline(proxy_thr, color="#2f6f2f", ls="--", lw=1.2, label=f"Proxy budget = {proxy_thr:.3f}")
    ax.axhline(tail_thr, color="#2f6f2f", ls=":", lw=1.2, label=f"Tail budget = {tail_thr:.2f}")

    colors = {
        "rcgdro": "#1f77b4",
        "P95": "#d62728",
        "P97": "#ff7f0e",
        "P99": "#2ca02c",
    }
    for x, y, xe, ye, lbl, t2 in zip(x_vals, y_vals, x_err, y_err, labels, perf_txt):
        ax.errorbar(
            x, y, xerr=xe, yerr=ye, fmt="o", ms=7, capsize=2.5,
            color=colors[lbl], ecolor=colors[lbl], elinewidth=1.0
        )
        ax.annotate(f"{lbl}\nH2={t2:.3f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    xlabel = "Proxy (clip-aware worst-cell loss, lower is better)"
    if args.selection_metric_mode == "stationary_unclipped":
        xlabel = "Proxy (unclipped worst-cell loss, lower is better)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tail worst-cell CVaR (lower is better)")
    ax.set_title("Camelyon17 Worked Example: Constrained Selection Region")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_pdf = Path(args.out_pdf)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("[selection-recipe-plot] wrote", out_png)
    print("[selection-recipe-plot] wrote", out_pdf)


if __name__ == "__main__":
    main()
