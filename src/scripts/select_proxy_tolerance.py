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
    return m, ci95_mean(x)


def _is_clipped_regime(regime: str) -> bool:
    return "clip" in str(regime).lower()


def _selection_proxy_column(df: pd.DataFrame, regime: str, mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "stationary_unclipped":
        return "proxy_worst_loss"
    if mode == "clip_aware":
        if "proxy_worst_loss_clip" in df.columns and pd.to_numeric(df["proxy_worst_loss_clip"], errors="coerce").notna().any():
            return "proxy_worst_loss_clip"
        return "proxy_worst_loss"
    if mode == "auto":
        if _is_clipped_regime(regime) and "proxy_worst_loss_clip" in df.columns:
            if pd.to_numeric(df["proxy_worst_loss_clip"], errors="coerce").notna().any():
                return "proxy_worst_loss_clip"
        return "proxy_worst_loss"
    raise ValueError(f"Unknown selection_metric_mode={mode}")


def _aggregate_phase0(df0: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "proxy_worst_loss_min",
        "proxy_worst_loss_clip_min",
        "proxy_worst_acc_min",
        "val_overall_acc",
        "val_overall_loss",
        "frac_clipped_train",
        "frac_clipped_val",
        "clip_alpha_active",
    ]
    cols = [c for c in cols if c in df0.columns]
    return (
        df0.groupby(["regime", "seed", "tag", "family", "epoch"], as_index=False)[cols]
        .mean()
        .copy()
    )


def _aggregate_phase1(df1: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "oracle_wg_acc",
        "within_cvar_mean",
        "worst_cell_cvar",
        "worst_cell_mean_loss",
    ]
    cols = [c for c in cols if c in df1.columns]
    return (
        df1.groupby(["regime", "seed", "tag", "family", "epoch"], as_index=False)[cols]
        .mean()
        .copy()
    )


def _pick_row(group_df: pd.DataFrame, threshold: float, selection_metric_mode: str) -> Dict:
    g = group_df.sort_values("epoch").copy()
    proxy_col = _selection_proxy_column(g, str(g["regime"].iloc[0]), selection_metric_mode)
    feasible = g[pd.to_numeric(g[proxy_col], errors="coerce") <= float(threshold)].copy()
    if not feasible.empty:
        # Among proxy-feasible epochs, choose safest tail first.
        feasible = feasible.sort_values(
            ["tail_worst_cvar", "oracle_wg_acc", "epoch"],
            ascending=[True, False, True],
        )
        row = feasible.iloc[0]
        status = "feasible_tail_min"
        feasible_count = int(feasible.shape[0])
    else:
        # Fallback when no epoch meets proxy tolerance.
        g2 = g.sort_values([proxy_col, "epoch"], ascending=[True, True])
        row = g2.iloc[0]
        status = "fallback_proxy_min"
        feasible_count = 0
    out = row.to_dict()
    out["selection_status"] = status
    out["feasible_epoch_count"] = feasible_count
    out["selection_proxy_col"] = proxy_col
    return out


def _summarize(df_sel: pd.DataFrame, selection_mode: str) -> pd.DataFrame:
    metrics = [
        "proxy_worst_loss",
        "proxy_worst_loss_clip",
        "proxy_worst_acc",
        "tail_worst_cvar",
        "oracle_wg_acc",
        "val_overall_acc",
        "val_overall_loss",
        "frac_clipped_train",
        "frac_clipped_val",
        "clip_alpha_active",
    ]
    rows = []
    for regime, g in df_sel.groupby("regime", dropna=False):
        row = {
            "selection_mode": selection_mode,
            "regime": str(regime),
            "n": int(g["seed"].nunique()),
            "feasible_seed_frac": float(np.mean(g["selection_status"] == "feasible_tail_min")),
            "fallback_seed_frac": float(np.mean(g["selection_status"] == "fallback_proxy_min")),
        }
        for m in metrics:
            if m not in g.columns:
                continue
            mean, ci = _mean_ci(g[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0_csv", required=True)
    ap.add_argument("--phase1_csv", required=True)
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--tail_family", default="teacher_difficulty")
    ap.add_argument("--baseline_regime", default="rcgdro")
    ap.add_argument("--proxy_tol", type=float, default=0.05)
    ap.add_argument("--selection_mode", default="selected_proxy_tol")
    ap.add_argument(
        "--selection_metric_mode",
        default="stationary_unclipped",
        choices=["stationary_unclipped", "clip_aware", "auto"],
        help="Proxy column used for feasibility/selection. stationary_unclipped matches the paper's DAOS gate.",
    )
    ap.add_argument("--out_selected_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    args = ap.parse_args()

    df0 = pd.read_csv(args.phase0_csv)
    df1 = pd.read_csv(args.phase1_csv)

    if "split" in df1.columns:
        df1 = df1[df1["split"] == "val"].copy()

    a0 = _aggregate_phase0(df0)
    a1 = _aggregate_phase1(df1)

    p0 = a0[a0["family"] == args.proxy_family].copy()
    p1 = a1[a1["family"] == args.tail_family].copy()
    if p0.empty:
        raise ValueError(f"No phase0 rows for proxy_family={args.proxy_family}")
    if p1.empty:
        raise ValueError(f"No phase1 rows for tail_family={args.tail_family}")

    merged = p0.merge(
        p1[["regime", "seed", "tag", "epoch", "oracle_wg_acc", "within_cvar_mean", "worst_cell_cvar", "worst_cell_mean_loss"]],
        on=["regime", "seed", "tag", "epoch"],
        how="inner",
    ).copy()
    if merged.empty:
        raise ValueError("No merged phase0/phase1 rows found for requested families.")

    merged = merged.rename(
        columns={
            "proxy_worst_loss_min": "proxy_worst_loss",
            "proxy_worst_loss_clip_min": "proxy_worst_loss_clip",
            "proxy_worst_acc_min": "proxy_worst_acc",
            "worst_cell_cvar": "tail_worst_cvar",
            "within_cvar_mean": "tail_within_cvar",
            "worst_cell_mean_loss": "tail_worst_loss",
        }
    )

    # Per-seed baseline best proxy value (stationary selector base).
    b = merged[merged["regime"] == args.baseline_regime].copy()
    if b.empty:
        raise ValueError(f"baseline_regime={args.baseline_regime} missing from merged data")
    baseline_proxy_rows = []
    for seed, g in b.groupby("seed", dropna=False):
        proxy_col = _selection_proxy_column(g, str(args.baseline_regime), args.selection_metric_mode)
        vals = pd.to_numeric(g[proxy_col], errors="coerce")
        if vals.notna().any():
            baseline_proxy_rows.append((int(seed), float(vals.min())))
    baseline_proxy_by_seed: Dict[int, float] = dict(baseline_proxy_rows)

    selected_rows = []
    for (regime, seed, tag), g in merged.groupby(["regime", "seed", "tag"], dropna=False):
        if int(seed) not in baseline_proxy_by_seed:
            continue
        base_proxy = float(baseline_proxy_by_seed[int(seed)])
        threshold = base_proxy * (1.0 + float(args.proxy_tol))
        row = _pick_row(g, threshold=threshold, selection_metric_mode=args.selection_metric_mode)
        row["baseline_proxy_best"] = base_proxy
        row["proxy_tolerance_threshold"] = threshold
        row["selection_mode"] = args.selection_mode
        row["selection_metric_mode"] = args.selection_metric_mode
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError("No rows selected under proxy tolerance rule.")

    df_sel = pd.DataFrame(selected_rows)
    # Keep compatibility with downstream summarizers expecting proxy_selected.
    if "proxy_selected" not in df_sel.columns and "proxy_worst_loss" in df_sel.columns:
        df_sel["proxy_selected"] = df_sel["proxy_worst_loss"]
    ordered_cols = [
        "selection_mode",
        "selection_status",
        "regime",
        "seed",
        "epoch",
        "tag",
        "baseline_proxy_best",
        "proxy_tolerance_threshold",
        "feasible_epoch_count",
        "proxy_worst_loss",
        "proxy_worst_loss_clip",
        "proxy_worst_acc",
        "proxy_selected",
        "tail_worst_cvar",
        "tail_within_cvar",
        "tail_worst_loss",
        "oracle_wg_acc",
        "val_overall_acc",
        "val_overall_loss",
        "frac_clipped_train",
        "frac_clipped_val",
        "clip_alpha_active",
    ]
    ordered_cols = [c for c in ordered_cols if c in df_sel.columns]
    df_sel = df_sel[ordered_cols].copy()

    out_selected = Path(args.out_selected_csv)
    out_summary = Path(args.out_summary_csv)
    out_selected.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    df_sel.to_csv(out_selected, index=False)
    _summarize(df_sel, selection_mode=args.selection_mode).to_csv(out_summary, index=False)

    print(f"[proxy-tol-select] wrote {out_selected}")
    print(f"[proxy-tol-select] wrote {out_summary}")


if __name__ == "__main__":
    main()
