import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from ..utils.stats import ci95_mean
from .make_properness_plots import _discover_runs, _select_epoch
from .train import build_head, eval_logits_loss_correct


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(ci95_mean(x))


def _oracle_worst_group_acc(correct: np.ndarray, groups: np.ndarray) -> float:
    worst = 1.0
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        worst = min(worst, float(np.mean(correct[mask])))
    return float(worst)


def _oracle_worst_group_loss(losses: np.ndarray, groups: np.ndarray) -> float:
    worst = -np.inf
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        worst = max(worst, float(np.mean(losses[mask])))
    return float(worst)


def _load_ckpt_epoch(run_dir: Path, epoch: int) -> Dict:
    ckpt_path = run_dir / f"ckpt_epoch{int(epoch):03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--metrics_suffix", required=True)
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--selection_metric_mode", default="auto", choices=["auto", "proxy_unclipped", "proxy_clip"])
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--selected_rows_csv", default="")
    ap.add_argument("--fixed_epoch", type=int, default=-1)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_root = Path(cfg["project"]["runs_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset}_{backbone}"
    selected_rows_csv = str(args.selected_rows_csv).strip()
    if selected_rows_csv:
        selected_df = pd.read_csv(selected_rows_csv)
        for col in ("regime", "seed", "epoch"):
            if col not in selected_df.columns:
                raise ValueError(f"selected_rows_csv missing required column: {col}")
        if "tag" not in selected_df.columns:
            selected_df["tag"] = ""
    else:
        phase0 = pd.read_csv(artifacts_dir / "metrics" / f"{dataset}_{backbone}_phase0_val_metrics_{args.metrics_suffix}.csv")
        tag_filters = [t.strip() for t in str(args.tag_filter).split(",") if t.strip()]

        regimes = [r.strip() for r in str(args.regimes).split(",") if r.strip()]
        runs = _discover_runs(runs_root, dataset, regimes)
        if tag_filters:
            runs = [r for r in runs if any(t in r.tag for t in tag_filters)]
        if not runs:
            raise FileNotFoundError("No matching runs found.")
        selected_rows = []
        for run in runs:
            proxy_df = phase0[
                (phase0["regime"] == run.regime)
                & (phase0["seed"] == run.seed)
                & (phase0["tag"] == run.tag)
                & (phase0["family"] == args.proxy_family)
            ].copy()
            if proxy_df.empty:
                continue
            if int(args.fixed_epoch) > 0:
                epoch = int(args.fixed_epoch)
                if epoch not in set(proxy_df["epoch"].astype(int)):
                    continue
            else:
                epoch = _select_epoch(proxy_df, run.regime, mode=args.selection_metric_mode)
            selected_rows.append(
                {
                    "regime": run.regime,
                    "seed": int(run.seed),
                    "tag": run.tag,
                    "epoch": int(epoch),
                }
            )
        selected_df = pd.DataFrame(selected_rows)
        if selected_df.empty:
            raise FileNotFoundError("No selected checkpoints found after applying filters.")

    X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(feat_dir / "y_test.npy")
    g_test = np.load(feat_dir / "g_test.npy")

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rows: List[Dict[str, object]] = []
    for _, sel in selected_df.iterrows():
        regime = str(sel["regime"])
        seed = int(sel["seed"])
        epoch = int(sel["epoch"])
        tag = str(sel.get("tag", "")).strip()
        run_dir = runs_root / dataset / regime / f"seed{seed}"
        if tag:
            run_dir = run_dir / tag
        ckpt = _load_ckpt_epoch(run_dir, epoch)
        run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model = build_head(
            d_in=int(run_cfg["d_in"]),
            hidden_dim=int(run_cfg.get("training", {}).get("hidden_dim", 0)),
            dropout=float(run_cfg.get("training", {}).get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        _logits, losses, corr = eval_logits_loss_correct(
            model,
            X_test,
            y_test,
            batch_size=int(cfg["training"].get("eval_batch_size", 2048)),
            device=device,
        )
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "tag": tag,
                "epoch": int(epoch),
                "test_oracle_wg_acc": _oracle_worst_group_acc(corr, g_test),
                "test_oracle_wg_loss": _oracle_worst_group_loss(losses, g_test),
                "test_overall_acc": float(np.mean(corr)),
                "test_overall_loss": float(np.mean(losses)),
            }
        )

    df = pd.DataFrame(rows)
    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_rows, index=False)

    summary_rows: List[Dict[str, object]] = []
    for regime, sub in df.groupby("regime", dropna=False):
        wg_m, wg_ci = _mean_ci(sub["test_oracle_wg_acc"].to_numpy())
        wgl_m, wgl_ci = _mean_ci(sub["test_oracle_wg_loss"].to_numpy())
        acc_m, acc_ci = _mean_ci(sub["test_overall_acc"].to_numpy())
        loss_m, loss_ci = _mean_ci(sub["test_overall_loss"].to_numpy())
        summary_rows.append(
            {
                "regime": regime,
                "n": int(sub.shape[0]),
                "test_oracle_wg_acc_mean": wg_m,
                "test_oracle_wg_acc_ci": wg_ci,
                "test_oracle_wg_loss_mean": wgl_m,
                "test_oracle_wg_loss_ci": wgl_ci,
                "test_overall_acc_mean": acc_m,
                "test_overall_acc_ci": acc_ci,
                "test_overall_loss_mean": loss_m,
                "test_overall_loss_ci": loss_ci,
            }
        )
    out_summary = Path(args.out_summary)
    pd.DataFrame(summary_rows).sort_values("regime").to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
