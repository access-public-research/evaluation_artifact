import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from ..config import load_config
from .make_properness_plots import _discover_runs, _select_epoch
from .phase0_eval import _load_families, _load_partitions, _proxy_metrics


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _bootstrap_hard_losses_from_logits(logits: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    pseudo = (z >= 0.0).astype(np.float64)
    targets = float(beta) * yy + (1.0 - float(beta)) * pseudo
    return np.maximum(z, 0.0) - z * targets + np.log1p(np.exp(-np.abs(z)))


def _bootstrap_hard_rw(logits: np.ndarray, y: np.ndarray, tail_mask: np.ndarray, beta: float) -> float:
    z = np.asarray(logits, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-z))
    ce_grad = np.abs(probs - yy)
    pseudo = (z >= 0.0).astype(np.float64)
    targets = float(beta) * yy + (1.0 - float(beta)) * pseudo
    obj_grad = np.abs(probs - targets)
    core_mask = ~tail_mask
    eps = 1e-8
    mean_ce_tail = float(np.mean(ce_grad[tail_mask]))
    mean_ce_core = float(np.mean(ce_grad[core_mask]))
    mean_obj_tail = float(np.mean(obj_grad[tail_mask]))
    mean_obj_core = float(np.mean(obj_grad[core_mask]))
    return float((mean_obj_tail / max(mean_ce_tail, eps)) / max((mean_obj_core / max(mean_ce_core, eps)), eps))


def _aggregate_phase0_proxy(phase0_csv: Path, proxy_family: str) -> pd.DataFrame:
    df = pd.read_csv(phase0_csv)
    keep = {
        "regime",
        "seed",
        "tag",
        "epoch",
        "family",
        "proxy_worst_loss_min",
        "proxy_worst_loss_clip_min",
        "val_overall_loss",
        "val_overall_acc",
    }
    miss = keep.difference(df.columns)
    if miss:
        raise ValueError(f"{phase0_csv} missing required columns: {sorted(miss)}")
    df = df[df["family"] == proxy_family].copy()
    if df.empty:
        raise ValueError(f"No phase0 rows for proxy family {proxy_family} in {phase0_csv}")
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


def _resolve_proxy_parts(eval_root: Path, proxy_family: str) -> tuple[List[np.ndarray], int]:
    fams = _load_families(eval_root, [proxy_family])
    if len(fams) != 1:
        raise ValueError(f"Expected exactly one proxy family for {proxy_family}, got {fams}")
    family, prefix, K, M = fams[0]
    parts: List[np.ndarray] = []
    for bank in ("A", "B"):
        parts.extend(
            _load_partitions(
                eval_root=eval_root,
                family=family,
                bank=bank,
                split="val_skew",
                prefix=prefix,
                num_parts=M,
            )
        )
    return parts, int(K)


def _resolve_tail_mask(eval_root: Path, tail_family: str) -> np.ndarray:
    fams = _load_families(eval_root, [tail_family])
    if len(fams) != 1:
        raise ValueError(f"Expected exactly one tail family for {tail_family}, got {fams}")
    family, prefix, K, _M = fams[0]
    bank_a = eval_root / family / "bankA" / "val_skew"
    matches = sorted(bank_a.glob(f"{prefix}_m00_K*.npy"))
    if not matches:
        raise FileNotFoundError(f"Missing tail partition under {bank_a}")
    bins = np.load(matches[0]).astype(np.int64)
    tail_start = int(np.floor(0.9 * int(K)))
    return bins >= tail_start


def _build_selection_rows(
    *,
    repo_root: Path,
    config_path: Path,
    phase0_csv: Path,
    proxy_family: str,
    tail_family: str,
    baseline_regime: str,
    target_regime: str,
    baseline_tag_filter: str,
    target_tag_filter: str,
    beta: float,
    out_selected_csv: Path,
    out_fixed_csv: Path,
) -> dict:
    cfg = load_config(str(config_path), dataset_path="configs/datasets/camelyon17.yaml")
    dataset = str(cfg["dataset"]["name"])
    backbone = str(cfg["embeddings"]["backbone"])
    artifacts_dir = Path(str(cfg["project"]["artifacts_dir"]))
    runs_root = Path(str(cfg["project"]["runs_dir"]))
    feat_dir = artifacts_dir / "embeds" / f"{dataset}_{backbone}"
    eval_root = artifacts_dir / "partitions_eval" / f"{dataset}_{backbone}"
    y_val = np.load(feat_dir / "y_val_skew.npy").astype(np.int64)
    proxy_parts, proxy_K = _resolve_proxy_parts(eval_root=eval_root, proxy_family=proxy_family)
    tail_mask = _resolve_tail_mask(eval_root=eval_root, tail_family=tail_family)

    phase0 = _aggregate_phase0_proxy(phase0_csv=phase0_csv, proxy_family=proxy_family)
    runs = _discover_runs(runs_root, dataset, [baseline_regime, target_regime])
    baseline_runs = [r for r in runs if r.regime == baseline_regime and baseline_tag_filter in r.tag]
    target_runs = [r for r in runs if r.regime == target_regime and target_tag_filter in r.tag]
    if not baseline_runs:
        raise FileNotFoundError(f"No baseline runs found for regime={baseline_regime} tag~{baseline_tag_filter}")
    if not target_runs:
        raise FileNotFoundError(f"No target runs found for regime={target_regime} tag~{target_tag_filter}")

    selected_rows: List[Dict[str, object]] = []
    fixed_rows: List[Dict[str, object]] = []
    bootstrap_detail_rows: List[Dict[str, object]] = []

    # Baseline rows from the existing proxy selector.
    for run in sorted(baseline_runs, key=lambda r: (r.seed, r.tag)):
        sub = phase0[(phase0["regime"] == run.regime) & (phase0["seed"] == run.seed) & (phase0["tag"] == run.tag)].copy()
        if sub.empty:
            continue
        ep_sel = _select_epoch(sub, run.regime, mode="auto")
        sel_row = sub[sub["epoch"] == ep_sel].iloc[0]
        selected_rows.append(
            {
                "selection_mode": "selected_best_proxy",
                "selection_policy": "baseline",
                "regime": run.regime,
                "seed": int(run.seed),
                "tag": run.tag,
                "epoch": int(ep_sel),
                "val_overall_loss": float(sel_row["val_overall_loss"]),
                "val_overall_acc": float(sel_row["val_overall_acc"]),
                "proxy_metric": float(sel_row["proxy_worst_loss_min"]),
                "proxy_metric_name": "proxy_worst_loss_min",
            }
        )
        fixed_rows.append(
            {
                "selection_mode": "fixed_epoch_30",
                "selection_policy": "baseline",
                "regime": run.regime,
                "seed": int(run.seed),
                "tag": run.tag,
                "epoch": 30,
            }
        )

    # Target rows with bootstrap-aware selector and val-loss comparator.
    for run in sorted(target_runs, key=lambda r: (r.seed, r.tag)):
        sub = phase0[(phase0["regime"] == run.regime) & (phase0["seed"] == run.seed) & (phase0["tag"] == run.tag)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("epoch").reset_index(drop=True)
        logits_by_epoch = np.load(run.run_dir / "val_logits_by_epoch.npy", mmap_mode="r")
        epochs = sub["epoch"].astype(int).tolist()
        bootstrap_proxy = []
        bootstrap_rw = []
        for ep in epochs:
            logits_ep = np.asarray(logits_by_epoch[int(ep) - 1], dtype=np.float64)
            losses_ep = _bootstrap_hard_losses_from_logits(logits_ep, y_val, beta=beta)
            correct_ep = ((logits_ep >= 0.0).astype(np.int64) == y_val).astype(np.float64)
            proxy_ep = _proxy_metrics(
                losses=losses_ep,
                correct=correct_ep,
                parts=proxy_parts,
                num_cells=proxy_K,
                min_cell=20,
                cvar_q=0.1,
            )["worst_loss"]
            rw_ep = _bootstrap_hard_rw(logits_ep, y_val, tail_mask=tail_mask, beta=beta)
            bootstrap_proxy.append(float(proxy_ep))
            bootstrap_rw.append(float(rw_ep))

        sub["bootstrap_proxy_worst_loss"] = np.asarray(bootstrap_proxy, dtype=np.float64)
        sub["bootstrap_rw"] = np.asarray(bootstrap_rw, dtype=np.float64)
        bootstrap_detail_rows.extend(sub.assign(tag=run.tag).to_dict(orient="records"))

        proxy_pick = sub.sort_values(
            ["bootstrap_proxy_worst_loss", "val_overall_loss", "epoch"],
            ascending=[True, True, True],
        ).iloc[0]
        val_pick = sub.sort_values(
            ["val_overall_loss", "bootstrap_proxy_worst_loss", "epoch"],
            ascending=[True, True, True],
        ).iloc[0]
        fixed_pick = sub[sub["epoch"] == 30]
        if fixed_pick.empty:
            raise ValueError(f"Missing epoch 30 for {run.run_dir}")
        fixed_pick = fixed_pick.iloc[0]

        early_rw_mean = float(sub[sub["epoch"].between(1, 5)]["bootstrap_rw"].mean())
        for policy, pick in (("proxy_only", proxy_pick), ("val_loss_only", val_pick)):
            selected_rows.append(
                {
                    "selection_mode": "selected_best_proxy",
                    "selection_policy": policy,
                    "regime": run.regime,
                    "seed": int(run.seed),
                    "tag": run.tag,
                    "epoch": int(pick["epoch"]),
                    "val_overall_loss": float(pick["val_overall_loss"]),
                    "val_overall_acc": float(pick["val_overall_acc"]),
                    "proxy_metric": float(pick["bootstrap_proxy_worst_loss"]),
                    "proxy_metric_name": "bootstrap_proxy_worst_loss",
                    "bootstrap_rw": float(pick["bootstrap_rw"]),
                    "bootstrap_rw_early_mean": early_rw_mean,
                }
            )
        fixed_rows.append(
            {
                "selection_mode": "fixed_epoch_30",
                "selection_policy": "proxy_only",
                "regime": run.regime,
                "seed": int(run.seed),
                "tag": run.tag,
                "epoch": 30,
                "proxy_metric": float(fixed_pick["bootstrap_proxy_worst_loss"]),
                "proxy_metric_name": "bootstrap_proxy_worst_loss",
                "bootstrap_rw": float(fixed_pick["bootstrap_rw"]),
                "bootstrap_rw_early_mean": early_rw_mean,
            }
        )

    df_sel = pd.DataFrame(selected_rows).sort_values(["selection_mode", "seed", "selection_policy"]).reset_index(drop=True)
    df_fix = pd.DataFrame(fixed_rows).sort_values(["seed", "selection_policy"]).reset_index(drop=True)
    df_detail = pd.DataFrame(bootstrap_detail_rows).sort_values(["seed", "epoch"]).reset_index(drop=True)

    out_selected_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(out_selected_csv, index=False)
    df_fix.to_csv(out_fixed_csv, index=False)
    detail_csv = out_selected_csv.with_name(out_selected_csv.stem.replace("_selected_rows_", "_bootstrap_proxy_detail_") + ".csv")
    df_detail.to_csv(detail_csv, index=False)
    return {
        "selected_rows_csv": str(out_selected_csv),
        "fixed_rows_csv": str(out_fixed_csv),
        "bootstrap_detail_csv": str(detail_csv),
    }


def _merge_eval_bundle(
    *,
    rows_csv: Path,
    domain_csv: Path,
    tail_csv: Path,
    calibration_csv: Path,
) -> pd.DataFrame:
    keys = ["regime", "seed", "epoch", "tag"]
    base = pd.read_csv(rows_csv)
    domain = pd.read_csv(domain_csv)
    tail = pd.read_csv(tail_csv)
    calib = pd.read_csv(calibration_csv)
    out = base.merge(domain, on=keys, how="left")
    out = out.merge(
        tail[
            [
                "regime",
                "seed",
                "tag",
                "epoch_selected",
                "tail_worst_cvar_selected",
                "tail_delta_vs_baseline",
                "distortion_mass_selected",
                "frac_clipped_selected",
            ]
        ].rename(columns={"epoch_selected": "epoch"}),
        on=keys,
        how="left",
    )
    out = out.merge(
        calib[
            [
                "regime",
                "seed",
                "tag",
                "epoch",
                "test_overall_acc",
                "test_overall_loss",
                "test_brier",
                "test_ece",
                "test_worst_domain_brier",
                "test_worst_domain_ece",
            ]
        ],
        on=keys,
        how="left",
    )
    return out


def _summarize_against_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for selection_mode, sub_mode in df.groupby("selection_mode", dropna=False):
        base = sub_mode[sub_mode["selection_policy"] == "baseline"].drop_duplicates(subset=["seed"]).set_index("seed")
        if base.empty:
            continue
        for policy, sub in sub_mode.groupby("selection_policy", dropna=False):
            sub = sub.drop_duplicates(subset=["seed"]).set_index("seed")
            seeds = sorted(set(base.index) & set(sub.index))
            if not seeds:
                continue
            base_sub = base.loc[seeds]
            sub = sub.loc[seeds]
            rec: Dict[str, object] = {
                "selection_mode": selection_mode,
                "selection_policy": policy,
                "n": len(seeds),
                "epoch_mean": float(sub["epoch"].mean()),
            }
            for col in [
                "test_hosp_2_acc",
                "test_hosp_2_loss",
                "tail_worst_cvar_selected",
                "test_ece",
                "test_worst_domain_ece",
                "proxy_metric",
            ]:
                if col in sub.columns and col in base_sub.columns:
                    rec[f"delta_{col}_vs_baseline"] = float((sub[col] - base_sub[col]).mean())
            if selection_mode == "selected_best_proxy" and policy == "proxy_only" and "bootstrap_rw" in sub.columns:
                rec["bootstrap_rw_selected_mean"] = float(sub["bootstrap_rw"].mean())
                rec["bootstrap_rw_early_mean"] = float(sub["bootstrap_rw_early_mean"].mean())
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["selection_mode", "selection_policy"]).reset_index(drop=True)
    if out.empty:
        return out

    # Add within-target proxy-only vs fixed30 delta on the bootstrap proxy.
    sel = out[(out["selection_mode"] == "selected_best_proxy") & (out["selection_policy"] == "proxy_only")]
    fix = out[(out["selection_mode"] == "fixed_epoch_30") & (out["selection_policy"] == "proxy_only")]
    if not sel.empty and not fix.empty:
        out["delta_proxy_metric_vs_fixed30"] = np.nan
        out.loc[sel.index, "delta_proxy_metric_vs_fixed30"] = float(
            sel.iloc[0]["delta_proxy_metric_vs_baseline"] - fix.iloc[0]["delta_proxy_metric_vs_baseline"]
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_v30_erm_bootstrap_camelyon_3seeds.yaml")
    ap.add_argument("--dataset", default="camelyon17")
    ap.add_argument("--bootstrap_regime", default="erm_bootstrap_h80_cam")
    ap.add_argument("--baseline_regime", default="erm")
    ap.add_argument("--baseline_tag_filter", default="v11ermsoftclipfix_cam_10s")
    ap.add_argument("--suite_suffix", default="camelyon_bootstrap_h80_pilot3s_20260331")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_train", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / str(args.config)).resolve()
    cfg = _load_yaml(config_path)
    target_tag_filter = str(cfg.get("training", {}).get("tag_suffix", "")).strip()
    if not target_tag_filter:
        raise ValueError("training.tag_suffix must be set in the bootstrap config.")
    beta = float(_load_yaml(repo_root / "configs" / "regimes" / f"{args.bootstrap_regime}.yaml").get("bootstrap_hard_beta", 0.0))
    if beta <= 0.0:
        raise ValueError("bootstrap_hard_beta must be positive.")

    py = str(args.python)
    dataset = str(args.dataset)
    suite_suffix = str(args.suite_suffix)
    regimes_csv = f"{args.baseline_regime},{args.bootstrap_regime}"
    tag_filter = f"{args.baseline_tag_filter},{target_tag_filter}"
    metrics_dir = repo_root / "artifacts" / "metrics"
    eval_config_path = metrics_dir / f"{dataset}_bootstrap_evalcfg_{suite_suffix}.yaml"
    phase0_csv = metrics_dir / f"{dataset}_resnet50_phase0_val_metrics_{suite_suffix}.csv"
    phase1_csv = metrics_dir / f"{dataset}_resnet50_phase1_pockets_{suite_suffix}.csv"
    selected_rows_csv = metrics_dir / f"{dataset}_bootstrap_selected_rows_{suite_suffix}.csv"
    fixed_rows_csv = metrics_dir / f"{dataset}_bootstrap_fixed30_rows_{suite_suffix}.csv"
    selected_domain_rows = metrics_dir / f"{dataset}_bootstrap_domain_selected_rows_{suite_suffix}.csv"
    selected_domain_summary = metrics_dir / f"{dataset}_bootstrap_domain_selected_summary_{suite_suffix}.csv"
    fixed_domain_rows = metrics_dir / f"{dataset}_bootstrap_domain_fixed30_rows_{suite_suffix}.csv"
    fixed_domain_summary = metrics_dir / f"{dataset}_bootstrap_domain_fixed30_summary_{suite_suffix}.csv"
    selected_cal_rows = metrics_dir / f"{dataset}_bootstrap_calibration_selected_rows_{suite_suffix}.csv"
    selected_cal_summary = metrics_dir / f"{dataset}_bootstrap_calibration_selected_summary_{suite_suffix}.csv"
    fixed_cal_rows = metrics_dir / f"{dataset}_bootstrap_calibration_fixed30_rows_{suite_suffix}.csv"
    fixed_cal_summary = metrics_dir / f"{dataset}_bootstrap_calibration_fixed30_summary_{suite_suffix}.csv"
    summary_csv = metrics_dir / f"{dataset}_bootstrap_pilot_summary_{suite_suffix}.csv"
    manifest_path = metrics_dir / f"{dataset}_bootstrap_manifest_{suite_suffix}.json"

    eval_cfg = _load_yaml(config_path)
    eval_cfg.setdefault("compute", {})
    eval_cfg["compute"]["device"] = "cpu"
    eval_config_path.parent.mkdir(parents=True, exist_ok=True)
    eval_config_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")

    if not int(args.skip_train):
        _run(
            [
                py,
                "-m",
                "src.scripts.train_bootstrap",
                "--config",
                str(args.config),
                "--dataset",
                dataset,
                "--regime",
                str(args.bootstrap_regime),
            ],
            cwd=repo_root,
        )

    _run(
        [
                py,
                "-m",
                "src.scripts.phase0_eval",
                "--config",
                str(eval_config_path),
                "--dataset",
                dataset,
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--tag_filter",
            tag_filter,
            "--out_suffix",
            suite_suffix,
            "--overwrite",
            "1",
        ],
        cwd=repo_root,
    )
    _run(
        [
                py,
                "-m",
                "src.scripts.phase1_eval",
                "--config",
                str(eval_config_path),
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--tag_filter",
            tag_filter,
            "--out_suffix",
            suite_suffix,
            "--overwrite",
            "1",
        ],
        cwd=repo_root,
    )

    selection_manifest = _build_selection_rows(
        repo_root=repo_root,
        config_path=config_path,
        phase0_csv=phase0_csv,
        proxy_family="conf_teacher_wpl",
        tail_family="teacher_difficulty",
        baseline_regime=str(args.baseline_regime),
        target_regime=str(args.bootstrap_regime),
        baseline_tag_filter=str(args.baseline_tag_filter),
        target_tag_filter=target_tag_filter,
        beta=beta,
        out_selected_csv=selected_rows_csv,
        out_fixed_csv=fixed_rows_csv,
    )

    for rows_csv, sel_mode, domain_rows, domain_summary, cal_rows, cal_summary in [
        (selected_rows_csv, "selected_best_proxy", selected_domain_rows, selected_domain_summary, selected_cal_rows, selected_cal_summary),
        (fixed_rows_csv, "fixed_epoch_30", fixed_domain_rows, fixed_domain_summary, fixed_cal_rows, fixed_cal_summary),
    ]:
        _run(
            [
                py,
                "-m",
                "src.scripts.camelyon_domain_eval",
                "--config",
                str(eval_config_path),
                "--dataset",
                dataset,
                "--summary_csv",
                str(rows_csv),
                "--tag_filter",
                tag_filter,
                "--out_csv",
                str(domain_rows),
                "--out_summary",
                str(domain_summary),
            ],
            cwd=repo_root,
        )
        _run(
            [
                py,
                "-m",
                "src.scripts.compute_tail_distortion_diagnostics",
                "--config",
                str(eval_config_path),
                "--dataset",
                dataset,
                "--selected_rows_csv",
                str(rows_csv),
                "--selection_mode",
                sel_mode,
                "--pockets_csv",
                str(phase1_csv),
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
                f"{suite_suffix}_{sel_mode}",
            ],
            cwd=repo_root,
        )
        _run(
            [
                py,
                "-m",
                "src.scripts.evaluate_selected_camelyon_calibration",
                "--config",
                str(eval_config_path),
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
                str(rows_csv),
                "--out_rows",
                str(cal_rows),
                "--out_summary",
                str(cal_summary),
            ],
            cwd=repo_root,
        )

    selected_tail_rows = metrics_dir / f"{dataset}_tail_distortion_rows_{suite_suffix}_selected_best_proxy.csv"
    fixed_tail_rows = metrics_dir / f"{dataset}_tail_distortion_rows_{suite_suffix}_fixed_epoch_30.csv"
    selected_bundle = _merge_eval_bundle(
        rows_csv=selected_rows_csv,
        domain_csv=selected_domain_rows,
        tail_csv=selected_tail_rows,
        calibration_csv=selected_cal_rows,
    )
    fixed_bundle = _merge_eval_bundle(
        rows_csv=fixed_rows_csv,
        domain_csv=fixed_domain_rows,
        tail_csv=fixed_tail_rows,
        calibration_csv=fixed_cal_rows,
    )
    summary = _summarize_against_baseline(pd.concat([selected_bundle, fixed_bundle], ignore_index=True))
    summary.to_csv(summary_csv, index=False)

    manifest = {
        "config": str(args.config),
        "dataset": dataset,
        "baseline_regime": str(args.baseline_regime),
        "bootstrap_regime": str(args.bootstrap_regime),
        "baseline_tag_filter": str(args.baseline_tag_filter),
        "target_tag_filter": target_tag_filter,
        "suite_suffix": suite_suffix,
        "beta": beta,
        "eval_config": str(eval_config_path),
        "selection_outputs": selection_manifest,
        "summary_csv": str(summary_csv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {summary_csv}")
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
