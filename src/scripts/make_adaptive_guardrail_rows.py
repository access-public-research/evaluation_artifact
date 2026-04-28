import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .make_guardrail_selection_rows import _aggregate_phase0, _select_best


def _convergence_threshold(base_df: pd.DataFrame, baseline_window: int, sigma_mult: float) -> tuple[float, float, float]:
    max_epoch = int(pd.to_numeric(base_df["epoch"], errors="coerce").max())
    start_epoch = max(1, max_epoch - int(baseline_window) + 1)
    window_df = base_df[pd.to_numeric(base_df["epoch"], errors="coerce") >= start_epoch].copy()
    vals = pd.to_numeric(window_df["val_overall_loss"], errors="coerce").dropna()
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0
    return mu + float(sigma_mult) * sigma, mu, sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--baseline_regime", required=True)
    ap.add_argument("--candidate_regimes", required=True)
    ap.add_argument("--suite_label", required=True)
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--selection_metric_mode", default="auto", choices=["auto", "proxy_unclipped", "proxy_clip"])
    ap.add_argument("--baseline_window", type=int, default=5)
    ap.add_argument("--sigma_mult", type=float, default=2.0)
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
            budget_thr, budget_mu, budget_sigma = _convergence_threshold(
                base_df,
                baseline_window=int(args.baseline_window),
                sigma_mult=float(args.sigma_mult),
            )
            feasible = cand_df[pd.to_numeric(cand_df["val_overall_loss"], errors="coerce") <= budget_thr].copy()
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
                "budget_mode": "mu_plus_sigma_window",
                "budget_value": float(args.sigma_mult),
                "budget_threshold": float(budget_thr),
                "baseline_window": int(args.baseline_window),
                "baseline_window_mean": float(budget_mu),
                "baseline_window_sd": float(budget_sigma),
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
        raise ValueError("No adaptive guardrail rows produced.")
    out = out.sort_values(["suite", "target_regime", "seed", "selection_policy"]).reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
