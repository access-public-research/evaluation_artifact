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
from .camelyon_domain_eval import (
    WildsWithDomain,
    build_backbone,
    build_head,
    eval_logits,
    eval_logits_full,
)
from .make_properness_plots import _discover_runs, _select_epoch


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(ci95_mean(x))


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


def _worst_domain_metric(
    probs: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    metric: str,
    num_bins: int,
) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    domains = np.asarray(domains, dtype=np.int64)
    worst = -np.inf
    for d in np.unique(domains):
        mask = domains == d
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


def _load_selected_rows(
    *,
    runs_root: Path,
    phase0_path: Path,
    dataset: str,
    regimes: List[str],
    proxy_family: str,
    selection_metric_mode: str,
    tag_filter: str,
) -> pd.DataFrame:
    phase0 = pd.read_csv(phase0_path)
    tag_filters = [t.strip() for t in str(tag_filter).split(",") if t.strip()]
    runs = _discover_runs(runs_root, dataset, regimes)
    if tag_filters:
        runs = [r for r in runs if any(t in r.tag for t in tag_filters)]
    if not runs:
        raise FileNotFoundError("No matching Camelyon runs found for calibration eval.")

    selected_rows = []
    for run in runs:
        proxy_df = phase0[
            (phase0["regime"] == run.regime)
            & (phase0["seed"] == run.seed)
            & (phase0["tag"] == run.tag)
            & (phase0["family"] == proxy_family)
        ].copy()
        if proxy_df.empty:
            continue
        epoch = _select_epoch(proxy_df, run.regime, mode=selection_metric_mode)
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
        raise FileNotFoundError("No selected Camelyon checkpoints found for calibration eval.")
    return selected_df


def _resolve_tag_dir(reg_dir: Path, requested_tag: str | None, tag_filter: str | None) -> Path:
    tag_dirs = [d for d in reg_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not tag_dirs:
        raise FileNotFoundError(f"No run tags found under {reg_dir}")

    req = (requested_tag or "").strip()
    if req:
        exact = reg_dir / req
        if exact.exists() and (exact / "config.json").exists():
            return exact
        raise FileNotFoundError(f"Requested tag '{req}' not found under {reg_dir}")

    candidates = tag_dirs
    if tag_filter:
        candidates = [d for d in candidates if tag_filter in d.name]
        if not candidates:
            raise FileNotFoundError(f"No tags matched tag_filter='{tag_filter}' under {reg_dir}")

    if len(candidates) != 1:
        names = sorted([d.name for d in candidates])
        raise ValueError(f"Ambiguous tags under {reg_dir}; provide tag in summary CSV. Candidates: {names}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="camelyon17")
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--metrics_suffix", required=True)
    ap.add_argument("--proxy_family", default="conf_teacher_wpl")
    ap.add_argument("--selection_metric_mode", default="auto", choices=["auto", "proxy_unclipped", "proxy_clip"])
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--selected_rows_csv", default="")
    ap.add_argument("--eval_split", default="test", choices=["test", "val_skew"])
    ap.add_argument("--num_bins", type=int, default=15)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset = cfg["dataset"]["name"]
    if dataset != "camelyon17":
        raise ValueError("This evaluator is only for camelyon17.")

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
        selected_df = _load_selected_rows(
            runs_root=runs_root,
            phase0_path=artifacts_dir / "metrics" / f"{dataset}_{backbone}_phase0_val_metrics_{args.metrics_suffix}.csv",
            dataset=dataset,
            regimes=[r.strip() for r in str(args.regimes).split(",") if r.strip()],
            proxy_family=str(args.proxy_family),
            selection_metric_mode=str(args.selection_metric_mode),
            tag_filter=str(args.tag_filter),
        )

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    eval_batch = int(cfg.get("finetune", {}).get("eval_batch_size", cfg["training"].get("eval_batch_size", 2048)))
    eval_split = str(args.eval_split)
    metric_prefix = "test" if eval_split == "test" else "val"

    X_test = y_test = a_test = None
    full_test_loader = None

    rows: List[Dict[str, object]] = []
    for _, sel in selected_df.iterrows():
        regime = str(sel["regime"])
        seed = int(sel["seed"])
        epoch = int(sel["epoch"])
        requested_tag = str(sel.get("tag", "")).strip()

        reg_dir = runs_root / dataset / regime / f"seed{seed}"
        tag_dir = _resolve_tag_dir(reg_dir, requested_tag=requested_tag, tag_filter=str(args.tag_filter))
        ckpt = tag_dir / f"ckpt_epoch{epoch:03d}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        state = torch.load(ckpt, map_location=device)
        model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        run_cfg = json.loads((tag_dir / "config.json").read_text(encoding="utf-8"))
        keys = list(model_state.keys())
        is_full_model = any(k.startswith("conv1.") or k.startswith("layer1.") or k.startswith("fc.") for k in keys)

        if eval_split == "val_skew":
            logits_val = np.load(tag_dir / "val_logits_by_epoch.npy", mmap_mode="r")
            y_val = np.load(feat_dir / "y_val_skew.npy")
            a_val = np.load(feat_dir / "a_val_skew.npy")
            logits_eval = np.asarray(logits_val[epoch - 1], dtype=np.float64)
            y_eval = np.asarray(y_val, dtype=np.int64)
            a_eval = np.asarray(a_val, dtype=np.int64)
        elif is_full_model:
            if full_test_loader is None:
                data_dir = cfg["dataset"].get("data_dir") or cfg["paths"]["wilds_data_dir"]
                wilds_name = cfg["dataset"]["wilds_dataset"]
                domain_field = str(cfg["dataset"].get("spurious_metadata_field", "hospital"))
                ds = load_wilds_dataset(wilds_name, data_dir, download=False)
                fields = list(getattr(ds, "metadata_fields", []))
                if domain_field not in fields:
                    raise ValueError(f"Domain field '{domain_field}' not found in metadata fields: {fields}")
                domain_col = int(fields.index(domain_field))
                model_tmp, tfm = build_backbone(backbone)
                del model_tmp
                test_base = ds.get_subset("test", frac=1.0, transform=tfm)
                test_ds = WildsWithDomain(test_base, domain_col=domain_col)
                full_test_loader = DataLoader(
                    test_ds,
                    batch_size=eval_batch,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )

            model, _ = build_backbone(backbone)
            model = model.to(device)
            model.load_state_dict(model_state, strict=True)
            logits_eval, y_eval, a_eval = eval_logits_full(model, full_test_loader, device)
        else:
            if X_test is None:
                X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
                y_test = np.load(feat_dir / "y_test.npy")
                a_test = np.load(feat_dir / "a_test.npy")
            d_in = int(run_cfg.get("d_in", int(X_test.shape[1])))
            hidden_dim = int(run_cfg.get("training", {}).get("hidden_dim", 0))
            dropout = float(run_cfg.get("training", {}).get("dropout", 0.0))
            eval_batch_run = int(run_cfg.get("training", {}).get("eval_batch_size", cfg["training"].get("eval_batch_size", 2048)))
            model = build_head(d_in, hidden_dim, dropout).to(device)
            model.load_state_dict(model_state, strict=True)
            logits_eval = eval_logits(model, X_test, y_test, eval_batch_run, device)
            y_eval = y_test
            a_eval = a_test

        logits_eval = np.asarray(logits_eval, dtype=np.float64)
        y_eval = np.asarray(y_eval, dtype=np.int64)
        a_eval = np.asarray(a_eval, dtype=np.int64)
        logits_clip = np.clip(logits_eval, -50.0, 50.0)
        probs = 1.0 / (1.0 + np.exp(-logits_clip))
        brier = np.square(probs - y_eval.astype(np.float64))
        preds = (probs >= 0.5).astype(np.int64)
        losses = np.maximum(logits_eval, 0.0) - logits_eval * y_eval.astype(np.float64) + np.log1p(np.exp(-np.abs(logits_eval)))

        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "tag": tag_dir.name,
                "epoch": epoch,
                f"{metric_prefix}_overall_acc": float(np.mean(preds == y_eval)),
                f"{metric_prefix}_overall_loss": float(np.mean(losses)),
                f"{metric_prefix}_brier": float(np.mean(brier)),
                f"{metric_prefix}_ece": _ece_binary(probs, y_eval, num_bins=int(args.num_bins)),
                f"{metric_prefix}_worst_domain_brier": _worst_domain_metric(probs, y_eval, a_eval, metric="brier", num_bins=int(args.num_bins)),
                f"{metric_prefix}_worst_domain_ece": _worst_domain_metric(probs, y_eval, a_eval, metric="ece", num_bins=int(args.num_bins)),
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
            f"{metric_prefix}_overall_acc",
            f"{metric_prefix}_overall_loss",
            f"{metric_prefix}_brier",
            f"{metric_prefix}_ece",
            f"{metric_prefix}_worst_domain_brier",
            f"{metric_prefix}_worst_domain_ece",
        ]:
            mean, ci = _mean_ci(pd.to_numeric(sub[col], errors="coerce").to_numpy())
            record[f"{col}_mean"] = mean
            record[f"{col}_ci"] = ci
        summary_rows.append(record)

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values("regime").to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
