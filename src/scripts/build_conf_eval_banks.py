import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import load_config
from ..utils.io import ensure_dir
from ..utils.seed import set_seed


class MemmapDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        x = np.asarray(self.X[int(i)], dtype=np.float32)
        y = int(self.y[int(i)])
        return x, y


def build_head(d_in: int, hidden_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(d_in, 1)
    return nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(hidden_dim, 1),
    )


@torch.no_grad()
def eval_logits(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    ds = MemmapDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = []
    for xb, _yb in dl:
        xb = xb.to(device)
        logits = model(xb).squeeze(1)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits_all, axis=0)


def _split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "validation", "val_bal"}:
        return "validation"
    if split in {"val_skew"}:
        return "val_skew"
    if split in {"train", "train_full"}:
        return "train"
    raise ValueError(f"Unknown split: {split}")


def _load_embeddings(feat_dir: Path, split: str):
    s = _split_name(split)
    if s == "val_skew":
        X = np.load(feat_dir / "X_val_skew.npy", mmap_mode="r")
        y = np.load(feat_dir / "y_val_skew.npy")
        return X, y
    X = np.load(feat_dir / f"X_{s}.npy", mmap_mode="r")
    y = np.load(feat_dir / f"y_{s}.npy")
    return X, y


def _quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    q = np.linspace(0.0, 1.0, int(num_bins) + 1)
    edges = np.quantile(values, q)
    bins = np.digitize(values, edges[1:-1], right=True)
    return bins.astype(np.int64)


def _confidence_bins(logits: np.ndarray, num_bins: int, within_pred_label: bool) -> np.ndarray:
    logits = np.clip(logits, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    conf = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(np.int64)
    if within_pred_label:
        if int(num_bins) % 2 != 0:
            raise ValueError("within_pred_label requires num_bins divisible by 2.")
        per_label = int(num_bins) // 2
        bins = np.full_like(preds, fill_value=0, dtype=np.int64)
        for label in (0, 1):
            mask = preds == label
            if not np.any(mask):
                continue
            bins_l = _quantile_bins(conf[mask], per_label)
            bins[mask] = int(label) * per_label + bins_l
        return bins
    return _quantile_bins(conf, int(num_bins))


def _difficulty_bins(logits: np.ndarray, y: np.ndarray, num_bins: int) -> np.ndarray:
    logits = np.clip(logits, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = y.astype(np.float32)
    eps = 1e-6
    losses = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
    return _quantile_bins(losses, int(num_bins))


def _select_ckpt(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        best_epoch = None
        best_acc = -1.0
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            acc = float(rec.get("val_acc", -1.0))
            epoch = int(rec.get("epoch", 0))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
        if best_epoch is not None:
            ckpt = run_dir / f"ckpt_epoch{best_epoch:03d}.pt"
            if ckpt.exists():
                return ckpt
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return ckpts[-1]


def _write_meta(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2))


def _parse_paths(raw: str) -> List[Path]:
    items = [s.strip() for s in (raw or "").split(",") if s.strip()]
    return [Path(s) for s in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="train,val_skew,validation")
    ap.add_argument("--num_bins", type=int, default=16)
    ap.add_argument("--init_seeds", default="111,222")
    ap.add_argument("--teacher_runs", default="")
    ap.add_argument("--within_pred_label", type=int, default=1)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
    if not feat_dir.exists():
        raise FileNotFoundError(f"Embeddings not found at {feat_dir}. Run embed_cache first.")

    eval_version = cfg.get("partitions", {}).get("eval_version")
    if eval_version:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}" / str(eval_version)
    else:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}"
    ensure_dir(eval_root)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    num_bins = int(args.num_bins)
    within_pred_label = bool(int(args.within_pred_label))

    init_seeds = [int(s.strip()) for s in args.init_seeds.split(",") if s.strip()]
    teacher_runs = _parse_paths(args.teacher_runs)

    if not teacher_runs:
        runs_root = Path(cfg["project"]["runs_dir"]) / dataset_name / "erm"
        candidates = []
        if runs_root.exists():
            for seed_dir in sorted(runs_root.glob("seed*")):
                for tag_dir in sorted(seed_dir.iterdir()):
                    if not tag_dir.is_dir():
                        continue
                    if not (tag_dir / "config.json").exists():
                        continue
                    tag = tag_dir.name
                    if "v2wl" in tag or "v3" in tag or "finetune" in tag:
                        continue
                    candidates.append(tag_dir)
        teacher_runs = candidates[:2]
    if not teacher_runs:
        raise FileNotFoundError("No teacher runs found; provide --teacher_runs.")

    conf_init_root = eval_root / ("conf_init_wpl" if within_pred_label else "conf_init")
    ensure_dir(conf_init_root)
    _write_meta(
        conf_init_root / "meta.json",
        {
            "family": conf_init_root.name,
            "prefix": "conf",
            "num_partitions": 1,
            "num_cells": num_bins,
            "seeds": init_seeds,
            "within_pred_label": within_pred_label,
            "splits": splits,
        },
    )

    for bank, seed in zip(["A", "B"], init_seeds[:2]):
        bank_root = conf_init_root / f"bank{bank}"
        ensure_dir(bank_root)
        set_seed(int(seed))
        X0, _y0 = _load_embeddings(feat_dir, splits[0])
        d_in = int(X0.shape[1])
        model = build_head(
            d_in=d_in,
            hidden_dim=int(cfg["training"].get("hidden_dim", 0)),
            dropout=float(cfg["training"].get("dropout", 0.0)),
        )
        device = cfg["compute"]["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = model.to(device)
        for split in splits:
            s = _split_name(split)
            out_dir = bank_root / s
            ensure_dir(out_dir)
            out_path = out_dir / f"conf_m00_K{num_bins}.npy"
            if out_path.exists() and not int(args.overwrite):
                continue
            X, y = _load_embeddings(feat_dir, split)
            logits = eval_logits(
                model,
                X,
                y,
                batch_size=int(cfg["training"].get("eval_batch_size", 2048)),
                device=device,
            )
            bins = _confidence_bins(logits, num_bins=num_bins, within_pred_label=within_pred_label)
            np.save(out_path, bins.astype(np.int32))

    for fam_name, builder, prefix in [
        ("conf_teacher_wpl" if within_pred_label else "conf_teacher", _confidence_bins, "conf"),
        ("teacher_difficulty", _difficulty_bins, "diff"),
    ]:
        fam_root = eval_root / fam_name
        ensure_dir(fam_root)
        _write_meta(
            fam_root / "meta.json",
            {
                "family": fam_name,
                "prefix": prefix,
                "num_partitions": 1,
                "num_cells": num_bins,
                "teacher_runs": [str(p) for p in teacher_runs],
                "within_pred_label": within_pred_label if "conf" in fam_name else False,
                "splits": splits,
            },
        )

        for bank, run_dir in zip(["A", "B"], (teacher_runs + teacher_runs)[:2]):
            bank_root = fam_root / f"bank{bank}"
            ensure_dir(bank_root)
            run_cfg = json.loads((run_dir / "config.json").read_text())
            d_in = int(run_cfg.get("d_in"))
            hidden_dim = int(run_cfg.get("training", {}).get("hidden_dim", 0))
            dropout = float(run_cfg.get("training", {}).get("dropout", 0.0))
            ckpt_path = _select_ckpt(run_dir)
            model = build_head(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout)
            device = cfg["compute"]["device"]
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            model = model.to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

            for split in splits:
                s = _split_name(split)
                out_dir = bank_root / s
                ensure_dir(out_dir)
                out_path = out_dir / f"{prefix}_m00_K{num_bins}.npy"
                if out_path.exists() and not int(args.overwrite):
                    continue
                X, y = _load_embeddings(feat_dir, split)
                logits = eval_logits(
                    model,
                    X,
                    y,
                    batch_size=int(cfg["training"].get("eval_batch_size", 2048)),
                    device=device,
                )
                if prefix == "conf":
                    bins = builder(logits, num_bins=num_bins, within_pred_label=within_pred_label)
                else:
                    bins = builder(logits, y, num_bins=num_bins)
                np.save(out_path, bins.astype(np.int32))

    print(f"[build_conf_eval_banks] wrote eval banks under {eval_root}")


if __name__ == "__main__":
    main()
