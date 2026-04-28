import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.stats import ci95_mean
from torch import nn

from .finetune import build_model
from .make_properness_plots import _discover_runs, _select_epoch


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


@torch.no_grad()
def _eval_full_model(model, loader: DataLoader, device: str, use_amp: bool):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    logits_all = []
    loss_all = []
    corr_all = []
    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb).squeeze(1)
        else:
            logits = model(xb).squeeze(1)
        losses = bce(logits, yb.float())
        preds = (logits >= 0).long()
        corr = (preds == yb).long()
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        loss_all.append(losses.detach().cpu().numpy().astype(np.float32))
        corr_all.append(corr.detach().cpu().numpy().astype(np.uint8))
    return np.concatenate(logits_all), np.concatenate(loss_all), np.concatenate(corr_all)


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
    metrics_dir = artifacts_dir / "metrics"

    g_test = np.load(feat_dir / "g_test.npy")
    y_test = np.load(feat_dir / "y_test.npy")

    ds = load_wilds_dataset(cfg["dataset"]["wilds_dataset"], cfg["dataset"]["data_dir"], download=False)
    model_template, tfm = build_model(backbone)
    del model_template
    test_subset = ds.get_subset("test", frac=1.0, transform=tfm)
    if len(test_subset) != len(g_test) or len(test_subset) != len(y_test):
        raise ValueError(
            f"Test subset length mismatch: dataset={len(test_subset)}, g_test={len(g_test)}, y_test={len(y_test)}"
        )

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    eval_batch_size = int(cfg.get("finetune", {}).get("eval_batch_size", 256))
    num_workers = 0
    use_amp = bool(cfg["compute"].get("amp", True))
    test_loader = DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    phase0 = pd.read_csv(metrics_dir / f"{dataset}_{backbone}_phase0_val_metrics_{args.metrics_suffix}.csv")
    tag_filters = [t.strip() for t in str(args.tag_filter).split(",") if t.strip()]

    regimes = [r.strip() for r in str(args.regimes).split(",") if r.strip()]
    runs = _discover_runs(runs_root, dataset, [f"{r}_finetune" for r in regimes])
    if tag_filters:
        runs = [r for r in runs if any(t in r.tag for t in tag_filters)]
    if not runs:
        raise FileNotFoundError("No matching finetune runs found.")

    rows: List[Dict[str, object]] = []
    for run in runs:
        base_regime = str(run.regime).removesuffix("_finetune")
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
        ckpt = _load_ckpt_epoch(run.run_dir, epoch)
        model, _ = build_model(backbone)
        model.to(device)
        model.load_state_dict(ckpt["model_state"])
        logits, _loss, corr = _eval_full_model(model, test_loader, device=device, use_amp=use_amp)
        preds = (logits >= 0).astype(np.int64)
        if preds.shape[0] != y_test.shape[0]:
            raise ValueError("Prediction length mismatch on test set.")
        rows.append(
            {
                "dataset": dataset,
                "regime": base_regime,
                "run_regime": run.regime,
                "seed": int(run.seed),
                "tag": run.tag,
                "epoch": int(epoch),
                "test_oracle_wg_acc": _oracle_worst_group_acc(corr, g_test),
                "test_overall_acc": float(np.mean(corr)),
            }
        )

    df = pd.DataFrame(rows)
    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_rows, index=False)

    summary_rows: List[Dict[str, object]] = []
    for regime, sub in df.groupby("regime", dropna=False):
        wg_m, wg_ci = _mean_ci(sub["test_oracle_wg_acc"].to_numpy())
        acc_m, acc_ci = _mean_ci(sub["test_overall_acc"].to_numpy())
        summary_rows.append(
            {
                "regime": regime,
                "n": int(sub.shape[0]),
                "test_oracle_wg_acc_mean": wg_m,
                "test_oracle_wg_acc_ci": wg_ci,
                "test_overall_acc_mean": acc_m,
                "test_overall_acc_ci": acc_ci,
            }
        )
    out_summary = Path(args.out_summary)
    pd.DataFrame(summary_rows).sort_values("regime").to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
