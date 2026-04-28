import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .run_camelyon_bootstrap_pilot import _merge_eval_bundle, _summarize_against_baseline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_eval_cfg(config_path: Path, out_path: Path) -> None:
    cfg = _load_yaml(config_path)
    cfg.setdefault("compute", {})
    cfg["compute"]["device"] = "cpu"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _policy_rank(policy: str) -> int:
    return {
        "baseline": 0,
        "proxy_only": 1,
        "val_loss_only": 2,
        "guardrail": 3,
        "oracle_loss": 4,
    }.get(str(policy), 99)


def _build_guardrail_rows(
    *,
    selected_rows: pd.DataFrame,
    detail_rows: pd.DataFrame,
    rho: float,
) -> pd.DataFrame:
    selected = selected_rows.copy()
    detail = detail_rows.copy().sort_values(["seed", "epoch"]).reset_index(drop=True)

    base = (
        selected[
            (selected["selection_mode"] == "selected_best_proxy")
            & (selected["selection_policy"] == "baseline")
        ]
        .drop_duplicates(subset=["seed"])
        .set_index("seed")
    )

    rows: List[Dict[str, object]] = []
    for seed, sub in detail.groupby("seed", sort=True):
        seed = int(seed)
        if seed not in base.index:
            continue
        base_row = base.loc[seed]
        base_std_val = float(base_row["val_overall_loss"])
        threshold = float(rho) * base_std_val
        early_rw_mean = float(sub[sub["epoch"].astype(int).between(1, 5)]["bootstrap_rw"].astype(float).mean())
        feasible = sub[sub["val_overall_loss"].astype(float) <= threshold].copy()
        if feasible.empty:
            rows.append(
                {
                    "selection_mode": "selected_best_proxy",
                    "selection_policy": "guardrail",
                    "regime": str(base_row["regime"]),
                    "seed": seed,
                    "tag": str(base_row["tag"]),
                    "epoch": int(base_row["epoch"]),
                    "val_overall_loss": float(base_row["val_overall_loss"]),
                    "val_overall_acc": float(base_row["val_overall_acc"]),
                    "proxy_metric": float(base_row["proxy_metric"]),
                    "proxy_metric_name": str(base_row["proxy_metric_name"]),
                    "bootstrap_rw": np.nan,
                    "bootstrap_rw_early_mean": early_rw_mean,
                    "fallback_to_baseline": 1,
                    "guardrail_budget_rho": float(rho),
                    "guardrail_threshold": threshold,
                }
            )
            continue

        pick = feasible.sort_values(
            ["bootstrap_proxy_worst_loss", "val_overall_loss", "epoch"],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append(
            {
                "selection_mode": "selected_best_proxy",
                "selection_policy": "guardrail",
                "regime": str(pick["regime"]),
                "seed": seed,
                "tag": str(pick["tag"]),
                "epoch": int(pick["epoch"]),
                "val_overall_loss": float(pick["val_overall_loss"]),
                "val_overall_acc": float(pick["val_overall_acc"]),
                "proxy_metric": float(pick["bootstrap_proxy_worst_loss"]),
                "proxy_metric_name": "bootstrap_proxy_worst_loss",
                "bootstrap_rw": float(pick["bootstrap_rw"]),
                "bootstrap_rw_early_mean": early_rw_mean,
                "fallback_to_baseline": 0,
                "guardrail_budget_rho": float(rho),
                "guardrail_threshold": threshold,
            }
        )

    return pd.DataFrame(rows).sort_values(["seed", "epoch"]).reset_index(drop=True)


def _build_all_epoch_rows(detail_rows: pd.DataFrame) -> pd.DataFrame:
    keep = detail_rows[["regime", "seed", "tag", "epoch"]].drop_duplicates().copy()
    keep["selection_mode"] = "all_epochs"
    keep["selection_policy"] = "oracle_candidate"
    return keep.sort_values(["seed", "epoch"]).reset_index(drop=True)


def _select_oracle_rows(
    *,
    detail_rows: pd.DataFrame,
    all_epoch_bundle: pd.DataFrame,
) -> pd.DataFrame:
    detail = detail_rows.copy()
    detail["seed"] = detail["seed"].astype(int)
    detail["epoch"] = detail["epoch"].astype(int)
    merged = all_epoch_bundle.merge(
        detail[
            [
                "regime",
                "seed",
                "tag",
                "epoch",
                "val_overall_loss",
                "val_overall_acc",
                "bootstrap_proxy_worst_loss",
                "bootstrap_rw",
            ]
        ],
        on=["regime", "seed", "tag", "epoch"],
        how="left",
    )

    rows: List[Dict[str, object]] = []
    for seed, sub in merged.groupby("seed", sort=True):
        pick = sub.sort_values(
            ["test_hosp_2_loss", "test_ece", "epoch"],
            ascending=[True, True, True],
        ).iloc[0]
        early_rw = float(detail[detail["seed"] == int(seed)].query("1 <= epoch <= 5")["bootstrap_rw"].astype(float).mean())
        rows.append(
            {
                "selection_mode": "selected_best_proxy",
                "selection_policy": "oracle_loss",
                "regime": str(pick["regime"]),
                "seed": int(seed),
                "tag": str(pick["tag"]),
                "epoch": int(pick["epoch"]),
                "val_overall_loss": float(pick["val_overall_loss"]),
                "val_overall_acc": float(pick["val_overall_acc"]),
                "proxy_metric": float(pick["bootstrap_proxy_worst_loss"]),
                "proxy_metric_name": "bootstrap_proxy_worst_loss",
                "bootstrap_rw": float(pick["bootstrap_rw"]),
                "bootstrap_rw_early_mean": early_rw,
                "oracle_target_metric": "test_hosp_2_loss",
            }
        )

    return pd.DataFrame(rows).sort_values(["seed", "epoch"]).reset_index(drop=True)


def _build_oracle_gap_summary(bundle: pd.DataFrame) -> pd.DataFrame:
    sub = bundle[bundle["selection_policy"].isin(["proxy_only", "oracle_loss"])].copy()
    proxy = sub[sub["selection_policy"] == "proxy_only"].drop_duplicates(subset=["seed"]).set_index("seed")
    oracle = sub[sub["selection_policy"] == "oracle_loss"].drop_duplicates(subset=["seed"]).set_index("seed")
    seeds = sorted(set(proxy.index) & set(oracle.index))
    if not seeds:
        return pd.DataFrame()
    proxy = proxy.loc[seeds]
    oracle = oracle.loc[seeds]
    row = {
        "n": len(seeds),
        "proxy_minus_oracle_test_hosp_2_loss": float((proxy["test_hosp_2_loss"] - oracle["test_hosp_2_loss"]).mean()),
        "proxy_minus_oracle_test_hosp_2_acc": float((proxy["test_hosp_2_acc"] - oracle["test_hosp_2_acc"]).mean()),
        "proxy_minus_oracle_tail_worst_cvar": float((proxy["tail_worst_cvar_selected"] - oracle["tail_worst_cvar_selected"]).mean()),
        "proxy_minus_oracle_test_ece": float((proxy["test_ece"] - oracle["test_ece"]).mean()),
        "proxy_minus_oracle_epoch": float((proxy["epoch"] - oracle["epoch"]).mean()),
        "proxy_minus_oracle_val_overall_loss": float((proxy["val_overall_loss"] - oracle["val_overall_loss"]).mean()),
    }
    return pd.DataFrame([row])


def _write_selector_table_tex(summary_df: pd.DataFrame, out_tex: Path) -> None:
    labels = {
        "baseline": "Baseline",
        "proxy_only": "Proxy-only",
        "val_loss_only": "Val-loss-only",
        "guardrail": "1.25x guardrail",
        "oracle_loss": "Oracle loss",
    }
    lines = [
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Selector & $\Delta$Held-out loss$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $\Delta$Tail CVaR$\downarrow$ & $\Delta$ECE$\downarrow$ \\",
        r"  \midrule",
    ]
    order = ["baseline", "proxy_only", "val_loss_only", "guardrail", "oracle_loss"]
    keyed = summary_df.set_index("selection_policy")
    for policy in order:
        if policy not in keyed.index:
            continue
        row = keyed.loc[policy]
        lines.append(
            "  {label} & {dloss:+.3f} & {dacc:+.3f} & {dtail:+.3f} & {dece:+.3f} \\\\".format(
                label=labels.get(policy, policy),
                dloss=float(row.get("delta_test_hosp_2_loss_vs_baseline", np.nan)),
                dacc=float(row.get("delta_test_hosp_2_acc_vs_baseline", np.nan)),
                dtail=float(row.get("delta_tail_worst_cvar_selected_vs_baseline", np.nan)),
                dece=float(row.get("delta_test_ece_vs_baseline", np.nan)),
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_v30_erm_bootstrap_camelyon_b60_10seeds.yaml")
    ap.add_argument("--dataset", default="camelyon17")
    ap.add_argument("--suite_suffix", default="camelyon_bootstrap_h60_pilot10s_20260331")
    ap.add_argument("--baseline_regime", default="erm")
    ap.add_argument("--target_regime", default="erm_bootstrap_h60_cam")
    ap.add_argument("--baseline_tag_filter", default="v11ermsoftclipfix_cam_10s")
    ap.add_argument("--guardrail_rho", type=float, default=1.25)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--stamp", default="20260331")
    args = ap.parse_args()

    repo_root = _repo_root()
    py = str(args.python)
    dataset = str(args.dataset)
    suite_suffix = str(args.suite_suffix)
    target_regime = str(args.target_regime)
    baseline_regime = str(args.baseline_regime)

    config_path = repo_root / str(args.config)
    cfg = _load_yaml(config_path)
    target_tag_filter = str(cfg.get("training", {}).get("tag_suffix", "")).strip()
    if not target_tag_filter:
        raise ValueError("training.tag_suffix must be set in the bootstrap config.")
    tag_filter = f"{args.baseline_tag_filter},{target_tag_filter}"
    regimes_csv = f"{baseline_regime},{target_regime}"

    metrics_dir = repo_root / "artifacts" / "metrics"
    tables_dir = repo_root / "paper" / "neurips2026_selection_risk" / "tables"

    eval_cfg = metrics_dir / f"{dataset}_bootstrap_selector_evalcfg_{args.stamp}.yaml"
    existing_selected = metrics_dir / f"{dataset}_bootstrap_selected_rows_{suite_suffix}.csv"
    existing_domain = metrics_dir / f"{dataset}_bootstrap_domain_selected_rows_{suite_suffix}.csv"
    existing_tail = metrics_dir / f"{dataset}_tail_distortion_rows_{suite_suffix}_selected_best_proxy.csv"
    existing_cal = metrics_dir / f"{dataset}_bootstrap_calibration_selected_rows_{suite_suffix}.csv"
    detail_csv = metrics_dir / f"{dataset}_bootstrap_bootstrap_proxy_detail_{suite_suffix}.csv"
    pockets_csv = metrics_dir / f"{dataset}_resnet50_phase1_pockets_{suite_suffix}.csv"

    out_rows = metrics_dir / f"{dataset}_bootstrap_selector_rows_extended_{args.stamp}.csv"
    out_domain = metrics_dir / f"{dataset}_bootstrap_selector_domain_rows_extended_{args.stamp}.csv"
    out_domain_summary = metrics_dir / f"{dataset}_bootstrap_selector_domain_summary_extended_{args.stamp}.csv"
    out_tail_rows = metrics_dir / f"{dataset}_tail_distortion_rows_bootstrap_selector_extended_{args.stamp}.csv"
    out_tail_summary = metrics_dir / f"{dataset}_tail_distortion_summary_bootstrap_selector_extended_{args.stamp}.csv"
    out_cal = metrics_dir / f"{dataset}_bootstrap_selector_calibration_rows_extended_{args.stamp}.csv"
    out_cal_summary = metrics_dir / f"{dataset}_bootstrap_selector_calibration_summary_extended_{args.stamp}.csv"
    out_summary = metrics_dir / f"{dataset}_bootstrap_selector_summary_extended_{args.stamp}.csv"

    all_epoch_rows = metrics_dir / f"{dataset}_bootstrap_all_epoch_rows_{args.stamp}.csv"
    all_epoch_domain = metrics_dir / f"{dataset}_bootstrap_all_epoch_domain_rows_{args.stamp}.csv"
    all_epoch_domain_summary = metrics_dir / f"{dataset}_bootstrap_all_epoch_domain_summary_{args.stamp}.csv"
    all_epoch_tail_rows = metrics_dir / f"{dataset}_tail_distortion_rows_bootstrap_all_epochs_{args.stamp}.csv"
    all_epoch_tail_summary = metrics_dir / f"{dataset}_tail_distortion_summary_bootstrap_all_epochs_{args.stamp}.csv"
    all_epoch_cal = metrics_dir / f"{dataset}_bootstrap_all_epoch_calibration_rows_{args.stamp}.csv"
    all_epoch_cal_summary = metrics_dir / f"{dataset}_bootstrap_all_epoch_calibration_summary_{args.stamp}.csv"
    oracle_gap_csv = metrics_dir / f"{dataset}_bootstrap_oracle_gap_summary_{args.stamp}.csv"
    out_tex = tables_dir / "table_bootstrap_selector_comparison_extended.tex"

    _write_eval_cfg(config_path, eval_cfg)

    selected_df = pd.read_csv(existing_selected)
    detail_df = pd.read_csv(detail_csv)
    base_keep = selected_df[
        (selected_df["selection_mode"] == "selected_best_proxy")
        & (selected_df["selection_policy"].isin(["baseline", "proxy_only", "val_loss_only"]))
    ].copy()
    guardrail_df = _build_guardrail_rows(
        selected_rows=base_keep,
        detail_rows=detail_df,
        rho=float(args.guardrail_rho),
    )
    combined_rows = pd.concat([base_keep, guardrail_df], ignore_index=True)
    combined_rows["policy_rank"] = combined_rows["selection_policy"].map(_policy_rank)
    combined_rows = combined_rows.sort_values(["seed", "policy_rank", "epoch"]).drop(columns=["policy_rank"]).reset_index(drop=True)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    combined_rows.to_csv(out_rows, index=False)

    _run(
        [
            py,
            "-m",
            "src.scripts.camelyon_domain_eval",
            "--config",
            str(eval_cfg),
            "--dataset",
            dataset,
            "--summary_csv",
            str(out_rows),
            "--tag_filter",
            tag_filter,
            "--out_csv",
            str(out_domain),
            "--out_summary",
            str(out_domain_summary),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.compute_tail_distortion_diagnostics",
            "--config",
            str(eval_cfg),
            "--dataset",
            dataset,
            "--selected_rows_csv",
            str(out_rows),
            "--selection_mode",
            "selected_best_proxy",
            "--pockets_csv",
            str(pockets_csv),
            "--families",
            "teacher_difficulty",
            "--banks",
            "A,B",
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--split",
            "val_skew",
            "--out_suffix",
            f"bootstrap_selector_extended_{args.stamp}",
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.evaluate_selected_camelyon_calibration",
            "--config",
            str(eval_cfg),
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--proxy_family",
            "conf_teacher_wpl",
            "--selection_metric_mode",
            "auto",
            "--tag_filter",
            tag_filter,
            "--selected_rows_csv",
            str(out_rows),
            "--out_rows",
            str(out_cal),
            "--out_summary",
            str(out_cal_summary),
        ],
        cwd=repo_root,
    )

    combined_bundle = _merge_eval_bundle(
        rows_csv=out_rows,
        domain_csv=out_domain,
        tail_csv=out_tail_rows,
        calibration_csv=out_cal,
    )

    all_rows = _build_all_epoch_rows(detail_df)
    all_rows.to_csv(all_epoch_rows, index=False)
    _run(
        [
            py,
            "-m",
            "src.scripts.camelyon_domain_eval",
            "--config",
            str(eval_cfg),
            "--dataset",
            dataset,
            "--summary_csv",
            str(all_epoch_rows),
            "--tag_filter",
            tag_filter,
            "--out_csv",
            str(all_epoch_domain),
            "--out_summary",
            str(all_epoch_domain_summary),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.compute_tail_distortion_diagnostics",
            "--config",
            str(eval_cfg),
            "--dataset",
            dataset,
            "--selected_rows_csv",
            str(all_epoch_rows),
            "--selection_mode",
            "all_epochs",
            "--pockets_csv",
            str(pockets_csv),
            "--families",
            "teacher_difficulty",
            "--banks",
            "A,B",
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--split",
            "val_skew",
            "--out_suffix",
            f"bootstrap_all_epochs_{args.stamp}",
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.evaluate_selected_camelyon_calibration",
            "--config",
            str(eval_cfg),
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--proxy_family",
            "conf_teacher_wpl",
            "--selection_metric_mode",
            "auto",
            "--tag_filter",
            tag_filter,
            "--selected_rows_csv",
            str(all_epoch_rows),
            "--out_rows",
            str(all_epoch_cal),
            "--out_summary",
            str(all_epoch_cal_summary),
        ],
        cwd=repo_root,
    )

    all_epoch_bundle = _merge_eval_bundle(
        rows_csv=all_epoch_rows,
        domain_csv=all_epoch_domain,
        tail_csv=all_epoch_tail_rows,
        calibration_csv=all_epoch_cal,
    )
    oracle_rows = _select_oracle_rows(
        detail_rows=detail_df,
        all_epoch_bundle=all_epoch_bundle,
    )
    full_rows = pd.concat([combined_rows, oracle_rows], ignore_index=True)
    full_rows["policy_rank"] = full_rows["selection_policy"].map(_policy_rank)
    full_rows = full_rows.sort_values(["seed", "policy_rank", "epoch"]).drop(columns=["policy_rank"]).reset_index(drop=True)
    full_rows.to_csv(out_rows, index=False)

    # Reuse the already-scored rows and append oracle rows from the all-epoch bundle.
    oracle_bundle = all_epoch_bundle.drop(columns=["selection_policy", "selection_mode"], errors="ignore").merge(
        oracle_rows[["regime", "seed", "tag", "epoch", "selection_policy", "selection_mode"]],
        on=["regime", "seed", "tag", "epoch"],
        how="inner",
    )
    final_bundle = pd.concat([combined_bundle, oracle_bundle], ignore_index=True)
    final_bundle = final_bundle.sort_values(["seed", "selection_policy", "epoch"]).reset_index(drop=True)
    summary_df = _summarize_against_baseline(final_bundle)
    summary_df.to_csv(out_summary, index=False)
    _write_selector_table_tex(summary_df, out_tex)

    oracle_gap_df = _build_oracle_gap_summary(final_bundle)
    oracle_gap_df.to_csv(oracle_gap_csv, index=False)

    manifest = {
        "config": str(config_path),
        "dataset": dataset,
        "suite_suffix": suite_suffix,
        "guardrail_rho": float(args.guardrail_rho),
        "baseline_regime": baseline_regime,
        "target_regime": target_regime,
        "baseline_tag_filter": str(args.baseline_tag_filter),
        "target_tag_filter": target_tag_filter,
        "outputs": {
            "selector_rows": str(out_rows),
            "selector_summary": str(out_summary),
            "oracle_gap_csv": str(oracle_gap_csv),
            "selector_table_tex": str(out_tex),
        },
    }
    manifest_path = metrics_dir / f"{dataset}_bootstrap_selector_manifest_extended_{args.stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {oracle_gap_csv}")
    print(f"[ok] wrote {out_tex}")
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
