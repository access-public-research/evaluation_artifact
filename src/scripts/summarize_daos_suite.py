import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    if x.size == 1:
        return m, 0.0
    ci = ci95_mean(x)
    return m, ci


def _pick_perf_metric(df_rows: pd.DataFrame, requested: str) -> str:
    if requested and requested in df_rows.columns:
        return requested
    for cand in ("test_hosp_2_acc", "oracle_wg_acc", "val_overall_acc"):
        if cand in df_rows.columns:
            return cand
    raise ValueError("Could not determine performance metric from rows/domain inputs.")


def _build_domain_map(domain_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    req = {"regime", "seed", metric}
    miss = req.difference(domain_df.columns)
    if miss:
        raise ValueError(f"domain_csv missing required columns: {sorted(miss)}")
    d = domain_df[["regime", "seed", metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rows_csv", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--distortion_rows_csv", default="")
    ap.add_argument("--domain_csv", default="")
    ap.add_argument("--domain_metric", default="test_hosp_2_acc")
    ap.add_argument("--selection_mode", default="selected_best_proxy")
    ap.add_argument("--baseline_regime", default="rcgdro")
    ap.add_argument("--tail_budget_delta", type=float, default=1.0)
    ap.add_argument("--perf_budget_delta", type=float, default=-0.005)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    df_rows = pd.read_csv(args.rows_csv)
    df_summary = pd.read_csv(args.summary_csv)

    if "selection_mode" in df_rows.columns:
        df_rows = df_rows[df_rows["selection_mode"] == args.selection_mode].copy()
    if "selection_mode" in df_summary.columns:
        df_summary = df_summary[df_summary["selection_mode"] == args.selection_mode].copy()

    if df_rows.empty:
        raise ValueError(f"No rows remaining for selection_mode={args.selection_mode}")

    # Attach domain metric by (regime, seed) when provided.
    perf_metric = _pick_perf_metric(df_rows, "")
    if args.domain_csv.strip():
        dmap = _build_domain_map(pd.read_csv(args.domain_csv), metric=args.domain_metric)
        df_rows = df_rows.merge(dmap, on=["regime", "seed"], how="left", suffixes=("", "_domain"))
        perf_metric = args.domain_metric
    else:
        perf_metric = _pick_perf_metric(df_rows, args.domain_metric)

    # Optional distortion diagnostics by selected rows (regime/seed/epoch).
    if args.distortion_rows_csv.strip():
        dist = pd.read_csv(args.distortion_rows_csv)
        keep_cols = [
            "regime",
            "seed",
            "epoch_selected",
            "distortion_mass_selected",
            "mean_excess_selected",
            "frac_clipped_selected",
            "clip_alpha",
            "tail_delta_vs_baseline",
        ]
        keep_cols = [c for c in keep_cols if c in dist.columns]
        if keep_cols:
            dist = dist[keep_cols].copy()
            if "epoch_selected" in dist.columns:
                dist = dist.rename(columns={"epoch_selected": "epoch"})
                df_rows = df_rows.merge(dist, on=["regime", "seed", "epoch"], how="left")
            else:
                df_rows = df_rows.merge(dist, on=["regime", "seed"], how="left")

    metric_cols = {
        "proxy_selected": "proxy_selected",
        "tail_worst_cvar": "tail_worst_cvar",
        "val_overall_acc": "val_overall_acc",
        "frac_clipped_val": "frac_clipped_val",
        "clip_alpha_active": "clip_alpha_active",
        "distortion_mass_selected": "distortion_mass_selected",
        "mean_excess_selected": "mean_excess_selected",
        perf_metric: perf_metric,
    }

    rows = []
    for regime, g in df_rows.groupby("regime", dropna=False):
        row: Dict[str, float | str | int] = {
            "dataset": args.dataset,
            "selection_mode": args.selection_mode,
            "regime": str(regime),
            "n": int(g["seed"].nunique()),
        }
        for out_name, col in metric_cols.items():
            if col not in g.columns:
                continue
            m, ci = _mean_ci(g[col])
            row[f"{out_name}_mean"] = m
            row[f"{out_name}_ci95"] = ci
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
    if out.empty:
        raise ValueError("No regime rows aggregated.")
    if args.baseline_regime not in set(out["regime"].astype(str)):
        raise ValueError(f"Baseline regime '{args.baseline_regime}' missing from summary.")

    base = out[out["regime"] == args.baseline_regime].iloc[0]
    for col, sign in [("proxy_selected_mean", -1.0), ("tail_worst_cvar_mean", +1.0), (f"{perf_metric}_mean", +1.0)]:
        if col not in out.columns:
            continue
        delta = out[col] - float(base[col])
        out[f"delta_vs_{args.baseline_regime}_{col}"] = delta
        # normalized direction (+ better)
        out[f"improvement_vs_{args.baseline_regime}_{col}"] = -sign * delta

    tail_delta_col = f"delta_vs_{args.baseline_regime}_tail_worst_cvar_mean"
    perf_delta_col = f"delta_vs_{args.baseline_regime}_{perf_metric}_mean"
    proxy_delta_col = f"delta_vs_{args.baseline_regime}_proxy_selected_mean"
    if tail_delta_col in out.columns:
        out["tail_within_budget"] = out[tail_delta_col] <= float(args.tail_budget_delta)
    if perf_delta_col in out.columns:
        out["perf_within_budget"] = out[perf_delta_col] >= float(args.perf_budget_delta)
    if proxy_delta_col in out.columns:
        out["proxy_improved"] = out[proxy_delta_col] < 0.0
    if {"proxy_improved", "tail_within_budget", "perf_within_budget"}.issubset(set(out.columns)):
        out["passes_all"] = out["proxy_improved"] & out["tail_within_budget"] & out["perf_within_budget"]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    lines = [
        f"# DAOS Summary ({args.dataset})",
        "",
        f"- selection mode: `{args.selection_mode}`",
        f"- baseline: `{args.baseline_regime}`",
        f"- performance metric: `{perf_metric}`",
        f"- tail budget: delta <= {args.tail_budget_delta}",
        f"- perf budget: delta >= {args.perf_budget_delta}",
        "",
        "## Regime Table",
        "",
    ]
    md_cols = [
        "regime",
        "n",
        "proxy_selected_mean",
        "tail_worst_cvar_mean",
        f"{perf_metric}_mean",
        "frac_clipped_val_mean",
        "clip_alpha_active_mean",
        "distortion_mass_selected_mean",
        "passes_all",
    ]
    md_cols = [c for c in md_cols if c in out.columns]
    try:
        lines.append(out[md_cols].to_markdown(index=False))
    except Exception:
        lines.append(out[md_cols].to_csv(index=False))
    lines.append("")

    if "passes_all" in out.columns:
        passed = out[out["passes_all"] == True]["regime"].astype(str).tolist()  # noqa: E712
        lines.append("## Pass Set")
        lines.append("")
        if passed:
            for r in passed:
                lines.append(f"- `{r}`")
        else:
            lines.append("- none")
        lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summarize-daos] wrote {out_csv}")
    print(f"[summarize-daos] wrote {out_md}")


if __name__ == "__main__":
    main()
