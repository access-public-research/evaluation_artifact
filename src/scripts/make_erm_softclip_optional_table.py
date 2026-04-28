import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_pm(mean: float, ci: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    ci_val = 0.0 if not np.isfinite(ci) else ci
    return f"{mean:.{digits}f} $\\pm$ {ci_val:.{digits}f}"


def _metric_with_delta(mean: float, ci: float, delta: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    sign = "+" if np.isfinite(delta) and delta >= 0 else ""
    return f"{_fmt_pm(mean, ci, digits)} ({sign}{delta:.{digits}f})"


def _merge_cam_perf(effect_cam: pd.DataFrame, cam_domain_csv: Path) -> pd.DataFrame:
    domain = pd.read_csv(cam_domain_csv)
    rows = []
    for regime, sub in domain.groupby("regime"):
        vals = sub["test_hosp_2_acc"].astype(float).to_numpy()
        mean = float(np.mean(vals))
        ci = 0.0
        if vals.size > 1:
            ci = ci95_mean(vals)
        rows.append({"regime": regime, "test_hosp_2_acc_mean": mean, "test_hosp_2_acc_ci": ci})
    dom = pd.DataFrame(rows)
    return effect_cam.merge(dom, on="regime", how="left")


def _dataset_block(name: str, df: pd.DataFrame, perf_col: str, perf_ci_col: str) -> list[str]:
    label_map = {
        "erm": "ERM",
        "erm_softclip_p95_a10": "P95",
        "erm_softclip_p97_a10": "P97",
        "erm_softclip_p99_a10": "P99",
        "erm_softclip_p95_a10_wb": "P95",
        "erm_softclip_p97_a10_wb": "P97",
        "erm_softclip_p99_a10_wb": "P99",
        "erm_softclip_p95_a10_cam": "P95",
        "erm_softclip_p97_a10_cam": "P97",
        "erm_softclip_p99_a10_cam": "P99",
    }
    order = [r for r in ["erm", *sorted([x for x in df["regime"].tolist() if x != "erm"])] if r in df["regime"].values]
    base = df[df["regime"] == "erm"].iloc[0]
    out = []
    first = True
    for regime in order:
        row = df[df["regime"] == regime].iloc[0]
        frac = 0.0 if regime == "erm" else float(row["frac_clipped_val_mean"])
        frac_ci = 0.0 if regime == "erm" else float(row["frac_clipped_val_ci"])
        if regime == "erm":
            proxy_mean = float(row["proxy_worst_loss_mean"])
            proxy_ci = float(row["proxy_worst_loss_ci"])
            proxy_delta = 0.0
        else:
            proxy_mean = float(row["proxy_worst_loss_clip_mean"])
            proxy_ci = float(row["proxy_worst_loss_clip_ci"])
            if not np.isfinite(proxy_mean):
                proxy_mean = float(row["proxy_worst_loss_mean"])
                proxy_ci = float(row["proxy_worst_loss_ci"])
            proxy_delta = float(proxy_mean - base["proxy_worst_loss_mean"])
        tail_delta = float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"])
        perf_delta = float(row[perf_col] - base[perf_col])

        prefix = name if first else ""
        first = False
        out.append(
            "  "
            + " & ".join(
                [
                    prefix,
                    label_map.get(regime, regime),
                    _fmt_pm(frac, frac_ci, 3),
                    _metric_with_delta(proxy_mean, proxy_ci, proxy_delta, 3),
                    _metric_with_delta(float(row["tail_worst_cvar_mean"]), float(row["tail_worst_cvar_ci"]), tail_delta, 2),
                    _metric_with_delta(float(row[perf_col]), float(row[perf_ci_col]), perf_delta, 3),
                ]
            )
            + " \\\\"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba_csv", required=True)
    ap.add_argument("--waterbirds_csv", required=True)
    ap.add_argument("--camelyon_csv", required=True)
    ap.add_argument("--camelyon_domain_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    celeba = pd.read_csv(args.celeba_csv).copy()
    waterbirds = pd.read_csv(args.waterbirds_csv).copy()
    cam = pd.read_csv(args.camelyon_csv).copy()
    cam = _merge_cam_perf(cam, Path(args.camelyon_domain_csv))

    lines: list[str] = []
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("  \\toprule")
    lines.append("  Dataset & Regime & FracClip & Proxy$\\downarrow$ ($\\Delta$) & Tail CVaR$\\downarrow$ ($\\Delta$) & Perf$\\uparrow$ ($\\Delta$) \\\\")
    lines.append("  \\midrule")
    lines.extend(_dataset_block("CelebA", celeba, "val_overall_acc_mean", "val_overall_acc_ci"))
    lines.append("  \\midrule")
    lines.extend(_dataset_block("Waterbirds", waterbirds, "val_overall_acc_mean", "val_overall_acc_ci"))
    lines.append("  \\midrule")
    lines.extend(_dataset_block("Camelyon17", cam, "test_hosp_2_acc_mean", "test_hosp_2_acc_ci"))
    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[erm-softclip-table] wrote {out_path}")


if __name__ == "__main__":
    main()
