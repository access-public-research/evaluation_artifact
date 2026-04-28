import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _mean_ci(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1:
        return mean, 0.0
    ci = ci95_mean(vals)
    return mean, ci


def _fmt_pm(mean: float, ci: float, d: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    ci = 0.0 if not np.isfinite(ci) else ci
    return f"{mean:.{d}f} $\\pm$ {ci:.{d}f}"


def _fmt_metric(mean: float, ci: float, delta: float, d: int = 3) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{_fmt_pm(mean, ci, d)} ({sign}{delta:.{d}f})"


def _merge_perf(effect: pd.DataFrame, domain_csv: Path) -> pd.DataFrame:
    dom = pd.read_csv(domain_csv)
    rows = []
    for regime, sub in dom.groupby("regime"):
        m, c = _mean_ci(sub["test_hosp_2_acc"].to_numpy())
        rows.append({"regime": regime, "test_hosp_2_acc_mean": m, "test_hosp_2_acc_ci": c})
    perf = pd.DataFrame(rows)
    return effect.merge(perf, on="regime", how="left")


def _block_lines(split_name: str, df: pd.DataFrame) -> list[str]:
    label = {
        "erm": "ERM",
        "erm_softclip_p95_a10_cam": "P95",
        "erm_softclip_p97_a10_cam": "P97",
        "erm_softclip_p99_a10_cam": "P99",
    }
    order = ["erm", "erm_softclip_p95_a10_cam", "erm_softclip_p97_a10_cam", "erm_softclip_p99_a10_cam"]
    base = df[df["regime"] == "erm"].iloc[0]
    out = []
    first = True
    for reg in order:
        row = df[df["regime"] == reg].iloc[0]
        is_base = reg == "erm"
        frac_mean = 0.0 if is_base else float(row["frac_clipped_val_mean"])
        frac_ci = 0.0 if is_base else float(row["frac_clipped_val_ci"])
        proxy_mean = float(row["proxy_worst_loss_mean"]) if is_base else float(row["proxy_worst_loss_clip_mean"])
        proxy_ci = float(row["proxy_worst_loss_ci"]) if is_base else float(row["proxy_worst_loss_clip_ci"])
        proxy_delta = proxy_mean - float(base["proxy_worst_loss_mean"])
        tail_delta = float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"])
        perf_delta = float(row["test_hosp_2_acc_mean"] - base["test_hosp_2_acc_mean"])
        prefix = split_name if first else ""
        first = False
        out.append(
            "  "
            + " & ".join(
                [
                    prefix,
                    label[reg],
                    _fmt_pm(frac_mean, frac_ci, 3),
                    _fmt_metric(proxy_mean, proxy_ci, proxy_delta, 3),
                    _fmt_metric(float(row["tail_worst_cvar_mean"]), float(row["tail_worst_cvar_ci"]), tail_delta, 2),
                    _fmt_metric(float(row["test_hosp_2_acc_mean"]), float(row["test_hosp_2_acc_ci"]), perf_delta, 3),
                ]
            )
            + " \\\\"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_effect_csv", required=True)
    ap.add_argument("--selected_domain_csv", required=True)
    ap.add_argument("--fixed_effect_csv", required=True)
    ap.add_argument("--fixed_domain_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    selected = _merge_perf(pd.read_csv(args.selected_effect_csv), Path(args.selected_domain_csv))
    fixed = _merge_perf(pd.read_csv(args.fixed_effect_csv), Path(args.fixed_domain_csv))

    lines = []
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("  \\toprule")
    lines.append("  Split & Regime & FracClip & Proxy$\\downarrow$ ($\\Delta$) & Tail CVaR$\\downarrow$ ($\\Delta$) & Test Hosp2$\\uparrow$ ($\\Delta$) \\\\")
    lines.append("  \\midrule")
    lines.extend(_block_lines("Selected", selected))
    lines.append("  \\midrule")
    lines.extend(_block_lines("Fixed e30", fixed))
    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[camelyon-erm-table] wrote {out}")


if __name__ == "__main__":
    main()
