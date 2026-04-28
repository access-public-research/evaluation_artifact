import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _aggregate_phase0(metrics_csv: Path, proxy_family: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    req = {"regime", "seed", "tag", "epoch", "family", "proxy_worst_loss_min", "proxy_worst_loss_clip_min", "val_overall_loss"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{metrics_csv} missing required columns: {sorted(miss)}")
    df = df[df["family"] == proxy_family].copy()
    if df.empty:
        raise ValueError(f"{metrics_csv} has no rows for family={proxy_family}")
    return (
        df.groupby(["regime", "seed", "tag", "epoch"], as_index=False)
        .agg(
            {
                "proxy_worst_loss_min": "mean",
                "proxy_worst_loss_clip_min": "mean",
                "val_overall_loss": "mean",
                "val_overall_acc": "mean",
            }
        )
        .sort_values(["regime", "seed", "tag", "epoch"])
        .reset_index(drop=True)
    )


def _metric_col(df: pd.DataFrame, regime: str, selection_metric_mode: str) -> str:
    mode = str(selection_metric_mode).strip().lower()
    if mode == "proxy_unclipped":
        return "proxy_worst_loss_min"
    if mode == "proxy_clip":
        return "proxy_worst_loss_clip_min" if df["proxy_worst_loss_clip_min"].notna().any() else "proxy_worst_loss_min"
    if "clip" in str(regime).lower() and df["proxy_worst_loss_clip_min"].notna().any():
        return "proxy_worst_loss_clip_min"
    return "proxy_worst_loss_min"


def _select_best(df: pd.DataFrame, regime: str, selection_metric_mode: str) -> pd.Series:
    metric_col = _metric_col(df, regime=regime, selection_metric_mode=selection_metric_mode)
    idx = int(df[metric_col].idxmin())
    row = df.loc[idx].copy()
    row["selected_proxy_metric"] = float(row[metric_col])
    row["selected_proxy_metric_col"] = metric_col
    return row


def _budget_threshold(baseline_loss: float, mode: str, value: float) -> float:
    mode = str(mode).strip().lower()
    if mode == "ratio":
        return float(baseline_loss) * float(value)
    if mode == "abs":
        return float(baseline_loss) + float(value)
    raise ValueError(f"Unknown budget_mode={mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--baseline_regime", required=True)
    ap.add_argument("--candidate_regimes", required=True, help="Comma-separated regimes to evaluate with proxy-only vs guardrail.")
    ap.add_argument("--suite_label", required=True)
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--selection_metric_mode", default="auto", choices=["auto", "proxy_unclipped", "proxy_clip"])
    ap.add_argument("--budget_mode", default="ratio", choices=["ratio", "abs"])
    ap.add_argument("--budget_value", type=float, default=1.5)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    phase0 = _aggregate_phase0(Path(args.metrics_csv), proxy_family=args.proxy_family)
    candidate_regimes: List[str] = [r.strip() for r in str(args.candidate_regimes).split(",") if r.strip()]
    if not candidate_regimes:
        raise ValueError("No candidate_regimes provided.")

    rows = []
    for candidate_regime in candidate_regimes:
        seeds = sorted(
            set(phase0.loc[phase0["regime"] == args.baseline_regime, "seed"].astype(int))
            & set(phase0.loc[phase0["regime"] == candidate_regime, "seed"].astype(int))
        )
        for seed in seeds:
            base_df = phase0[(phase0["regime"] == args.baseline_regime) & (phase0["seed"] == seed)].copy()
            cand_df = phase0[(phase0["regime"] == candidate_regime) & (phase0["seed"] == seed)].copy()
            if base_df.empty or cand_df.empty:
                continue

            baseline_pick = _select_best(base_df, regime=args.baseline_regime, selection_metric_mode="proxy_unclipped")
            proxy_pick = _select_best(cand_df, regime=candidate_regime, selection_metric_mode=args.selection_metric_mode)

            budget_thr = _budget_threshold(
                baseline_loss=float(baseline_pick["val_overall_loss"]),
                mode=args.budget_mode,
                value=float(args.budget_value),
            )
            feasible = cand_df[cand_df["val_overall_loss"] <= budget_thr].copy()
            if feasible.empty:
                guard_pick = baseline_pick.copy()
                fallback_to_baseline = True
                feasible_target_epochs = 0
            else:
                guard_pick = _select_best(feasible, regime=candidate_regime, selection_metric_mode=args.selection_metric_mode)
                fallback_to_baseline = False
                feasible_target_epochs = int(feasible.shape[0])

            common = {
                "suite": args.suite_label,
                "target_regime": candidate_regime,
                "baseline_regime": args.baseline_regime,
                "seed": int(seed),
                "baseline_epoch": int(baseline_pick["epoch"]),
                "baseline_tag": str(baseline_pick["tag"]),
                "baseline_val_overall_loss": float(baseline_pick["val_overall_loss"]),
                "baseline_proxy_metric": float(baseline_pick["selected_proxy_metric"]),
                "budget_mode": str(args.budget_mode),
                "budget_value": float(args.budget_value),
                "budget_threshold": float(budget_thr),
            }
            rows.append(
                {
                    **common,
                    "selection_policy": "baseline",
                    "regime": args.baseline_regime,
                    "tag": str(baseline_pick["tag"]),
                    "epoch": int(baseline_pick["epoch"]),
                    "chosen_val_overall_loss": float(baseline_pick["val_overall_loss"]),
                    "chosen_val_overall_acc": float(baseline_pick["val_overall_acc"]),
                    "chosen_proxy_metric": float(baseline_pick["selected_proxy_metric"]),
                    "chosen_proxy_metric_col": str(baseline_pick["selected_proxy_metric_col"]),
                    "fallback_to_baseline": False,
                    "feasible_target_epochs": int(cand_df.shape[0]),
                }
            )
            rows.append(
                {
                    **common,
                    "selection_policy": "proxy_only",
                    "regime": candidate_regime,
                    "tag": str(proxy_pick["tag"]),
                    "epoch": int(proxy_pick["epoch"]),
                    "chosen_val_overall_loss": float(proxy_pick["val_overall_loss"]),
                    "chosen_val_overall_acc": float(proxy_pick["val_overall_acc"]),
                    "chosen_proxy_metric": float(proxy_pick["selected_proxy_metric"]),
                    "chosen_proxy_metric_col": str(proxy_pick["selected_proxy_metric_col"]),
                    "fallback_to_baseline": False,
                    "feasible_target_epochs": int(cand_df.shape[0]),
                }
            )
            rows.append(
                {
                    **common,
                    "selection_policy": "guardrail",
                    "regime": str(guard_pick["regime"]),
                    "tag": str(guard_pick["tag"]),
                    "epoch": int(guard_pick["epoch"]),
                    "chosen_val_overall_loss": float(guard_pick["val_overall_loss"]),
                    "chosen_val_overall_acc": float(guard_pick["val_overall_acc"]),
                    "chosen_proxy_metric": float(guard_pick["selected_proxy_metric"]),
                    "chosen_proxy_metric_col": str(guard_pick["selected_proxy_metric_col"]),
                    "fallback_to_baseline": bool(fallback_to_baseline),
                    "feasible_target_epochs": int(feasible_target_epochs),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No guardrail selection rows produced.")
    out = out.sort_values(["suite", "target_regime", "seed", "selection_policy"]).reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
