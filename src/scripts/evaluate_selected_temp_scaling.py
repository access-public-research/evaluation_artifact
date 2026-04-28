import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.stats import ci95_mean
from .camelyon_domain_eval import (
    WildsWithDomain,
    build_backbone,
    build_head as build_camelyon_head,
    eval_logits,
    eval_logits_full,
)
from .evaluate_selected_civilcomments_test import CIVILCOMMENTS_IDENTITY_FIELDS
from .train import build_head as build_text_head
from .train import eval_logits_loss_correct


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


def _logits_to_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
    *,
    domains: np.ndarray | None = None,
    meta: np.ndarray | None = None,
) -> Dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits_clip = np.clip(logits, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(-logits_clip))
    preds = (probs >= 0.5).astype(np.int64)
    losses = np.maximum(logits, 0.0) - logits * labels.astype(np.float64) + np.log1p(np.exp(-np.abs(logits)))
    out = {
        "overall_acc": float(np.mean(preds == labels)),
        "overall_loss": float(np.mean(losses)),
        "brier": float(np.mean(np.square(probs - labels.astype(np.float64)))),
        "ece": _ece_binary(probs, labels, num_bins=num_bins),
    }
    if domains is not None:
        out["worst_domain_brier"] = _worst_domain_metric(probs, labels, domains, metric="brier", num_bins=num_bins)
        out["worst_domain_ece"] = _worst_domain_metric(probs, labels, domains, metric="ece", num_bins=num_bins)
    if meta is not None:
        out["wilds_wg_brier"] = _worst_group_metric(probs, labels, meta, metric="brier", num_bins=num_bins)
        out["wilds_wg_ece"] = _worst_group_metric(probs, labels, meta, metric="ece", num_bins=num_bins)
    return out


def _fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 50) -> float:
    logits_t = torch.as_tensor(np.asarray(val_logits, dtype=np.float32).reshape(-1))
    labels_t = torch.as_tensor(np.asarray(val_labels, dtype=np.float32).reshape(-1))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=1e3)
        loss = F.binary_cross_entropy_with_logits(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp.detach()).clamp(min=1e-3, max=1e3).item())


def _resolve_run_dir(runs_root: Path, dataset: str, regime: str, seed: int, tag: str) -> Path:
    run_dir = runs_root / dataset / regime / f"seed{seed}"
    if tag:
        run_dir = run_dir / tag
    return run_dir


def _camelyon_logits_for_selected(
    cfg: Dict,
    selected_df: pd.DataFrame,
    num_bins: int,
) -> pd.DataFrame:
    dataset = "camelyon17"
    backbone = cfg["embeddings"]["backbone"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_root = Path(cfg["project"]["runs_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset}_{backbone}"
    y_val = np.load(feat_dir / "y_val_skew.npy").astype(np.int64)
    y_test = np.load(feat_dir / "y_test.npy").astype(np.int64)
    a_test = np.load(feat_dir / "a_test.npy").astype(np.int64)
    X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    eval_batch = int(cfg.get("finetune", {}).get("eval_batch_size", cfg["training"].get("eval_batch_size", 2048)))

    full_test_loader = None
    rows: List[Dict[str, object]] = []
    for _, sel in selected_df.iterrows():
        regime = str(sel["regime"])
        seed = int(sel["seed"])
        epoch = int(sel["epoch"])
        tag = str(sel.get("tag", "")).strip()
        run_dir = _resolve_run_dir(runs_root, dataset, regime, seed, tag)
        ckpt = run_dir / f"ckpt_epoch{epoch:03d}.pt"
        state = torch.load(ckpt, map_location=device)
        model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        is_full_model = any(k.startswith("conv1.") or k.startswith("layer1.") or k.startswith("fc.") for k in model_state.keys())

        val_logits_all = np.load(run_dir / "val_logits_by_epoch.npy", mmap_mode="r")
        val_logits = np.asarray(val_logits_all[epoch - 1], dtype=np.float64)
        temperature = _fit_temperature(val_logits, y_val)

        if is_full_model:
            if full_test_loader is None:
                shared_data_dir = str(cfg.get("paths", {}).get("wilds_data_dir", "")).strip()
                dataset_data_dir = str(cfg.get("dataset", {}).get("data_dir", "")).strip()
                expected = "camelyon17_v1.0"
                candidates = [p for p in [shared_data_dir, dataset_data_dir] if p]
                data_dir = candidates[0]
                for cand in candidates:
                    if (Path(cand) / expected).exists():
                        data_dir = cand
                        break
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
            test_logits, _, _ = eval_logits_full(model, full_test_loader, device)
        else:
            d_in = int(run_cfg.get("d_in", int(X_test.shape[1])))
            hidden_dim = int(run_cfg.get("training", {}).get("hidden_dim", 0))
            dropout = float(run_cfg.get("training", {}).get("dropout", 0.0))
            eval_batch_run = int(run_cfg.get("training", {}).get("eval_batch_size", cfg["training"].get("eval_batch_size", 2048)))
            model = build_camelyon_head(d_in, hidden_dim, dropout).to(device)
            model.load_state_dict(model_state, strict=True)
            test_logits = eval_logits(model, X_test, y_test, eval_batch_run, device)

        raw = _logits_to_metrics(test_logits, y_test, num_bins, domains=a_test)
        scaled = _logits_to_metrics(np.asarray(test_logits, dtype=np.float64) / temperature, y_test, num_bins, domains=a_test)
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "tag": tag,
                "epoch": epoch,
                "temperature": temperature,
                **{f"raw_{k}": v for k, v in raw.items()},
                **{f"ts_{k}": v for k, v in scaled.items()},
            }
        )
    return pd.DataFrame(rows)


def _civilcomments_logits_for_selected(
    cfg: Dict,
    selected_df: pd.DataFrame,
    num_bins: int,
) -> pd.DataFrame:
    dataset = "civilcomments"
    backbone = cfg["embeddings"]["backbone"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_root = Path(cfg["project"]["runs_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset}_{backbone}"
    X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(feat_dir / "y_test.npy").astype(np.int64)
    meta_test = np.load(feat_dir / "meta_test.npy")
    y_val = np.load(feat_dir / "y_val_skew.npy").astype(np.int64)

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rows: List[Dict[str, object]] = []
    for _, sel in selected_df.iterrows():
        regime = str(sel["regime"])
        seed = int(sel["seed"])
        epoch = int(sel["epoch"])
        tag = str(sel.get("tag", "")).strip()
        run_dir = _resolve_run_dir(runs_root, dataset, regime, seed, tag)
        ckpt = torch.load(run_dir / f"ckpt_epoch{epoch:03d}.pt", map_location="cpu")
        run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model = build_text_head(
            d_in=int(run_cfg["d_in"]),
            hidden_dim=int(run_cfg.get("training", {}).get("hidden_dim", 0)),
            dropout=float(run_cfg.get("training", {}).get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        logits_val_all = np.load(run_dir / "val_logits_by_epoch.npy", mmap_mode="r")
        logits_val = np.asarray(logits_val_all[epoch - 1], dtype=np.float64)
        temperature = _fit_temperature(logits_val, y_val)
        logits_test, _, _ = eval_logits_loss_correct(
            model,
            X_test,
            y_test,
            batch_size=int(cfg["training"].get("eval_batch_size", 2048)),
            device=device,
        )
        raw = _logits_to_metrics(logits_test, y_test, num_bins, meta=meta_test)
        scaled = _logits_to_metrics(np.asarray(logits_test, dtype=np.float64) / temperature, y_test, num_bins, meta=meta_test)
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "tag": tag,
                "epoch": epoch,
                "temperature": temperature,
                **{f"raw_{k}": v for k, v in raw.items()},
                **{f"ts_{k}": v for k, v in scaled.items()},
            }
        )
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    metric_cols = [c for c in df.columns if c.startswith("raw_") or c.startswith("ts_") or c == "temperature"]
    for regime, sub in df.groupby("regime", dropna=False):
        rec: Dict[str, object] = {"regime": regime, "n": int(sub.shape[0])}
        for col in metric_cols:
            mean, ci = _mean_ci(pd.to_numeric(sub[col], errors="coerce").to_numpy())
            rec[f"{col}_mean"] = mean
            rec[f"{col}_ci"] = ci
        if "raw_overall_loss" in sub.columns and "ts_overall_loss" in sub.columns:
            delta = pd.to_numeric(sub["ts_overall_loss"], errors="coerce") - pd.to_numeric(sub["raw_overall_loss"], errors="coerce")
            rec["delta_overall_loss_ts_minus_raw_mean"], rec["delta_overall_loss_ts_minus_raw_ci"] = _mean_ci(delta.to_numpy())
        if "raw_ece" in sub.columns and "ts_ece" in sub.columns:
            delta = pd.to_numeric(sub["ts_ece"], errors="coerce") - pd.to_numeric(sub["raw_ece"], errors="coerce")
            rec["delta_ece_ts_minus_raw_mean"], rec["delta_ece_ts_minus_raw_ci"] = _mean_ci(delta.to_numpy())
        if "raw_brier" in sub.columns and "ts_brier" in sub.columns:
            delta = pd.to_numeric(sub["ts_brier"], errors="coerce") - pd.to_numeric(sub["raw_brier"], errors="coerce")
            rec["delta_brier_ts_minus_raw_mean"], rec["delta_brier_ts_minus_raw_ci"] = _mean_ci(delta.to_numpy())
        summary_rows.append(rec)
    return pd.DataFrame(summary_rows).sort_values("regime")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["camelyon17", "civilcomments"])
    ap.add_argument("--selected_rows_csv", required=True)
    ap.add_argument("--num_bins", type=int, default=15)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    selected_df = pd.read_csv(args.selected_rows_csv)
    for col in ("regime", "seed", "epoch"):
        if col not in selected_df.columns:
            raise ValueError(f"selected_rows_csv missing required column: {col}")
    if "tag" not in selected_df.columns:
        selected_df["tag"] = ""

    if args.dataset == "camelyon17":
        df = _camelyon_logits_for_selected(cfg, selected_df, int(args.num_bins))
    else:
        df = _civilcomments_logits_for_selected(cfg, selected_df, int(args.num_bins))

    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_rows, index=False)

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    _summarize(df).to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
