import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from ..utils.stats import ci95_mean
from .evaluate_selected_civilcomments_test import CIVILCOMMENTS_IDENTITY_FIELDS
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


def _load_ckpt_epoch(run_dir: Path, epoch: int) -> Dict:
    ckpt_path = run_dir / f"ckpt_epoch{int(epoch):03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def _ece_binary(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    conf = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(np.int64)
    correct = (preds == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, int(num_bins) + 1)
    ece = 0.0
    n = max(1, conf.shape[0])
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += float(mask.sum() / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _worst_group_metric(
    probs: np.ndarray,
    labels: np.ndarray,
    meta: np.ndarray,
    metric: str,
    num_bins: int,
) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    meta = np.asarray(meta)
    worst = -np.inf
    for identity_idx in range(len(CIVILCOMMENTS_IDENTITY_FIELDS)):
        identity = meta[:, identity_idx].astype(np.int64)
        for label in (0, 1):
            mask = (identity == 1) & (labels == label)
            if not np.any(mask):
                continue
            if metric == "brier":
                val = float(np.mean(np.square(probs[mask] - labels[mask].astype(np.float64))))
            elif metric == "ece":
                val = _ece_binary(probs[mask], labels[mask], num_bins=num_bins)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            worst = max(worst, val)
    return float(worst) if np.isfinite(worst) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--metrics_suffix", required=True)
    ap.add_argument("--proxy_family", default="global_hash")
    ap.add_argument("--selection_metric_mode", default="auto", choices=["auto", "proxy_unclipped", "proxy_clip"])
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--selected_rows_csv", default="")
    ap.add_argument("--num_bins", type=int, default=15)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    if dataset != "civilcomments":
        raise ValueError("This evaluator is only for civilcomments.")

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
            epoch = _select_epoch(proxy_df, run.regime, mode=args.selection_metric_mode)
            selected_rows.append(
                {"regime": run.regime, "seed": int(run.seed), "tag": run.tag, "epoch": int(epoch)}
            )
        selected_df = pd.DataFrame(selected_rows)
        if selected_df.empty:
            raise FileNotFoundError("No selected checkpoints found.")

    X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(feat_dir / "y_test.npy")
    meta_test = np.load(feat_dir / "meta_test.npy")

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
        logits, losses, corr = eval_logits_loss_correct(
            model,
            X_test,
            y_test,
            batch_size=int(cfg["training"].get("eval_batch_size", 2048)),
            device=device,
        )
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        brier = np.square(probs - y_test.astype(np.float64))
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "tag": tag,
                "epoch": epoch,
                "test_overall_acc": float(np.mean(corr)),
                "test_overall_loss": float(np.mean(losses)),
                "test_brier": float(np.mean(brier)),
                "test_ece": _ece_binary(probs, y_test, num_bins=int(args.num_bins)),
                "test_wilds_wg_brier": _worst_group_metric(probs, y_test, meta_test, metric="brier", num_bins=int(args.num_bins)),
                "test_wilds_wg_ece": _worst_group_metric(probs, y_test, meta_test, metric="ece", num_bins=int(args.num_bins)),
            }
        )

    df = pd.DataFrame(rows)
    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_rows, index=False)

    summary_rows: List[Dict[str, object]] = []
    for regime, sub in df.groupby("regime", dropna=False):
        record = {"regime": regime, "n": int(sub.shape[0])}
        for col in [
            "test_overall_acc",
            "test_overall_loss",
            "test_brier",
            "test_ece",
            "test_wilds_wg_brier",
            "test_wilds_wg_ece",
        ]:
            mean, ci = _mean_ci(pd.to_numeric(sub[col], errors="coerce").to_numpy())
            record[f"{col}_mean"] = mean
            record[f"{col}_ci"] = ci
        summary_rows.append(record)

    out_summary = Path(args.out_summary)
    pd.DataFrame(summary_rows).sort_values("regime").to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
