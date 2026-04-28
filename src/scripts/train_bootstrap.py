import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import load_config
from ..utils.io import ensure_dir
from ..utils.seed import set_seed
from .train import MemmapDataset, _load_split_arrays, build_head, eval_logits_loss_correct


def _bootstrap_hard_losses(
    logits: torch.Tensor,
    y_int: torch.Tensor,
    bce: nn.BCEWithLogitsLoss,
    beta: float,
) -> torch.Tensor:
    if not (0.0 < beta <= 1.0):
        raise ValueError("bootstrap_hard_beta must lie in (0, 1].")
    y_f = y_int.float()
    pseudo = (logits.detach() >= 0.0).float()
    targets = beta * y_f + (1.0 - beta) * pseudo
    return bce(logits, targets)


def train_one(cfg, dataset_name: str, regime_name: str, seed: int) -> Path:
    set_seed(int(seed))
    torch.backends.cudnn.benchmark = True

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_dir = Path(cfg["project"]["runs_dir"])
    backbone = str(cfg["embeddings"]["backbone"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
    if not (feat_dir / "info.json").exists():
        raise FileNotFoundError(f"Missing embeddings at {feat_dir}. Run embed_cache first.")

    X_train, y_train = _load_split_arrays(feat_dir, "train")
    train_idx_path = feat_dir / "train_sub_idx.npy"
    train_idx = np.load(train_idx_path) if train_idx_path.exists() else np.arange(len(y_train), dtype=np.int64)
    X_val, y_val = _load_split_arrays(feat_dir, "val_skew")

    train_cfg = cfg["training"]
    reg_cfg = cfg["regime"]
    if str(reg_cfg.get("objective", "erm")) != "erm":
        raise ValueError("train_bootstrap.py only supports ERM-style heads.")

    d_in = int(X_train.shape[1])
    hidden_dim = int(train_cfg.get("hidden_dim", 0))
    dropout = float(train_cfg.get("dropout", 0.0))
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    shuffle_train = bool(train_cfg.get("shuffle", True))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 2048))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    beta = float(reg_cfg.get("bootstrap_hard_beta", 0.0) or 0.0)

    device = str(cfg["compute"]["device"])
    num_workers = int(cfg["compute"].get("num_workers", 0))

    tag = f"h{hidden_dim}_do{dropout}_lr{lr}_wd{weight_decay}_bs{batch_size}_ep{epochs}"
    tag_suffix = str(train_cfg.get("tag_suffix", "")).strip()
    if tag_suffix:
        tag += f"_{tag_suffix}"

    run_dir = runs_dir / dataset_name / regime_name / f"seed{int(seed)}" / tag
    ensure_dir(run_dir)
    final_ckpt_path = run_dir / f"ckpt_epoch{int(epochs):03d}.pt"
    if final_ckpt_path.exists():
        return run_dir

    model = build_head(d_in, hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    ds_train = MemmapDataset(X_train, y_train, indices=train_idx)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    val_logits = []
    val_losses = []
    val_correct = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb, _idx in dl_train:
            opt.zero_grad(set_to_none=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb).squeeze(1)
            losses = _bootstrap_hard_losses(logits=logits, y_int=yb, bce=bce, beta=beta)
            loss = losses.mean()
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu().item()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])

        train_loss = total_loss / max(n_seen, 1)
        logits_v, loss_v, corr_v = eval_logits_loss_correct(
            model,
            X_val,
            y_val,
            batch_size=eval_batch_size,
            device=device,
        )
        val_logits.append(logits_v)
        val_losses.append(loss_v)
        val_correct.append(corr_v)

        rec = {
            "epoch": int(ep),
            "train_loss": float(train_loss),
            "val_loss": float(loss_v.mean()),
            "val_acc": float(corr_v.mean()),
            "bootstrap_hard_beta": float(beta),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        ckpt_path = run_dir / f"ckpt_epoch{ep:03d}.pt"
        torch.save({"epoch": ep, "model_state": model.state_dict()}, ckpt_path)

    np.save(run_dir / "val_logits_by_epoch.npy", np.stack(val_logits, axis=0).astype(np.float32))
    np.save(run_dir / "val_loss_by_epoch.npy", np.stack(val_losses, axis=0).astype(np.float32))
    np.save(run_dir / "val_correct_by_epoch.npy", np.stack(val_correct, axis=0).astype(np.uint8))

    cfg_out = {
        "dataset": dataset_name,
        "regime": regime_name,
        "seed": int(seed),
        "d_in": int(d_in),
        "tag": tag,
        "training": train_cfg,
        "regime_cfg": reg_cfg,
        "clip_loss": 0.0,
        "clip_alpha": 1.0,
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
        "gce_q": 0.0,
        "bootstrap_hard_beta": float(beta),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        dataset_path=f"configs/datasets/{args.dataset}.yaml",
        regime_path=f"configs/regimes/{args.regime}.yaml",
    )
    dataset_name = str(cfg["dataset"]["name"])
    regime_name = str(cfg["regime"]["name"])
    seeds = list(cfg["training"]["seeds"])
    if int(args.seed) >= 0:
        seeds = [int(args.seed)]

    for seed in seeds:
        train_one(cfg, dataset_name=dataset_name, regime_name=regime_name, seed=int(seed))


if __name__ == "__main__":
    main()
