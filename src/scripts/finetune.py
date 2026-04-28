import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.io import ensure_dir
from ..utils.seed import set_seed


class SubsetWithIndex(Dataset):
    def __init__(self, base, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        x, y, _m = self.base[base_idx]
        return x, int(y), base_idx


class SubsetOverrideY(Dataset):
    def __init__(self, base, indices: np.ndarray, y_override: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y_override, dtype=np.int64)
        if self.indices.shape[0] != self.y.shape[0]:
            raise ValueError("indices and y_override must have same length.")

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        x, _y, _m = self.base[base_idx]
        y = int(self.y[i])
        return x, y


def build_model(backbone: str) -> Tuple[nn.Module, torchvision.transforms.Compose]:
    name = str(backbone).lower()
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        d = int(model.fc.in_features)
        model.fc = nn.Linear(d, 1)
        return model, weights.transforms()
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        d = int(model.fc.in_features)
        model.fc = nn.Linear(d, 1)
        return model, weights.transforms()
    raise ValueError(f"Unknown backbone: {backbone}")


def freeze_except(model: nn.Module, unfreeze_layers: List[str]):
    for p in model.parameters():
        p.requires_grad = False
    # Always keep head trainable
    for p in model.fc.parameters():
        p.requires_grad = True
    for name in unfreeze_layers:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True


def freeze_batchnorm_stats(model: nn.Module) -> None:
    # Keep BatchNorm running stats fixed to reduce domain drift.
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for m in model.modules():
        if isinstance(m, bn_types):
            m.eval()


def _binary_objective_losses(
    logits: torch.Tensor,
    y_int: torch.Tensor,
    bce: nn.BCEWithLogitsLoss,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    """Per-example binary training losses for finetuning.

    Mirrors the head-only trainer so objective-family controls use the same
    definition under end-to-end finetuning.
    """
    if label_smoothing > 0.0 and focal_gamma > 0.0:
        raise ValueError("Use either label_smoothing or focal_gamma, not both.")

    if label_smoothing > 0.0:
        y_t = y_int.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return bce(logits, y_t)

    y_f = y_int.float()
    ce = bce(logits, y_f)
    if focal_gamma <= 0.0:
        return ce

    probs = torch.sigmoid(logits)
    p_t = probs * y_f + (1.0 - probs) * (1.0 - y_f)
    focal_weight = torch.pow((1.0 - p_t).clamp_min(1e-8), focal_gamma)
    return focal_weight * ce


@torch.no_grad()
def eval_logits_loss_correct(model: nn.Module, loader: DataLoader, device: str, use_amp: bool):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    logits_all = []
    loss_all = []
    correct_all = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb).squeeze(1)
        else:
            logits = model(xb).squeeze(1)
        losses = bce(logits, yb.float())
        preds = (logits >= 0).long()
        correct = (preds == yb).long()
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        loss_all.append(losses.detach().cpu().numpy().astype(np.float32))
        correct_all.append(correct.detach().cpu().numpy().astype(np.uint8))
    return np.concatenate(logits_all), np.concatenate(loss_all), np.concatenate(correct_all)


def _load_partitions(part_root: Path, split: str, num_partitions: int, K: int) -> List[np.ndarray]:
    split_dir = part_root / split
    parts = []
    for m in range(num_partitions):
        p = split_dir / f"hash_m{m:02d}_K{int(K)}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing partition file: {p}")
        parts.append(np.load(p))
    return parts


def train_one(cfg, dataset_name: str, seed: int) -> Path:
    set_seed(int(seed))
    torch.backends.cudnn.benchmark = True

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_dir = Path(cfg["project"]["runs_dir"])
    backbone = cfg["embeddings"]["backbone"]
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
    part_base = artifacts_dir / "partitions" / f"{dataset_name}_{backbone}"
    part_version = cfg.get("partitions", {}).get("version")
    part_root = part_base / str(part_version) if part_version and (part_base / str(part_version)).exists() else part_base
    if not feat_dir.exists():
        raise FileNotFoundError(f"Missing embeddings at {feat_dir}. Run embed_cache first.")
    if not part_root.exists():
        raise FileNotFoundError(f"Missing partitions at {part_root}. Run build_partitions first.")

    data_dir = cfg["dataset"].get("data_dir") or cfg["paths"]["wilds_data_dir"]
    wilds_name = cfg["dataset"]["wilds_dataset"]
    ds = load_wilds_dataset(wilds_name, data_dir, download=False)

    model, tfm = build_model(backbone)
    unfreeze_layers = cfg["finetune"].get("unfreeze_layers", ["layer4"])
    freeze_except(model, unfreeze_layers)

    train_idx = np.load(feat_dir / "train_sub_idx.npy")
    train_frac = float(cfg["finetune"].get("train_frac", 1.0))
    if train_frac < 1.0:
        rng = np.random.default_rng(int(seed) + 123)
        n_keep = max(1, int(round(train_frac * len(train_idx))))
        train_idx = rng.choice(train_idx, size=n_keep, replace=False)

    # Base split (full order)
    train_base = ds.get_subset("train", frac=1.0, transform=tfm)
    train_ds = SubsetWithIndex(train_base, train_idx)

    # Val skew indices + noisy labels
    val_idx = np.load(feat_dir / "val_skew_idx.npy")
    y_val = np.load(feat_dir / "y_val_skew.npy")
    source_split = json.loads((feat_dir / "splits.json").read_text()).get("val_skew_source_split", "validation")
    val_base = ds.get_subset("val" if source_split == "validation" else source_split, frac=1.0, transform=tfm)
    val_ds = SubsetOverrideY(val_base, val_idx, y_val)

    device = cfg["compute"]["device"]
    use_amp = bool(cfg["compute"].get("amp", True))
    model.to(device)

    batch_size = int(cfg["finetune"]["batch_size"])
    eval_batch_size = int(cfg["finetune"]["eval_batch_size"])
    epochs = int(cfg["finetune"]["epochs"])
    lr = float(cfg["finetune"]["lr"])
    backbone_lr_raw = cfg["finetune"].get("backbone_lr", None)
    head_lr_raw = cfg["finetune"].get("head_lr", None)
    backbone_lr = float(backbone_lr_raw) if backbone_lr_raw is not None else None
    head_lr = float(head_lr_raw) if head_lr_raw is not None else None
    freeze_bn_stats = bool(cfg["finetune"].get("freeze_bn_stats", False))
    weight_decay = float(cfg["finetune"].get("weight_decay", 1e-4))
    grad_clip = float(cfg["finetune"].get("grad_clip", 1.0))
    num_workers = int(cfg["compute"]["num_workers"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Proxy partitions for rcgdro training
    reg_cfg = cfg["regime"]
    clip_loss = float(reg_cfg.get("clip_loss", 0.0) or 0.0)
    clip_alpha = float(reg_cfg.get("clip_alpha", 0.0) or 0.0)
    label_smoothing = float(reg_cfg.get("label_smoothing", 0.0) or 0.0)
    focal_gamma = float(reg_cfg.get("focal_gamma", 0.0) or 0.0)
    parts = []
    q = None
    if reg_cfg["objective"] == "rcgdro":
        proxy_root = part_root / "proxy"
        num_parts = int(cfg["partitions"]["proxy"]["num_partitions"])
        K = int(cfg["partitions"]["proxy"]["num_cells"])
        parts = _load_partitions(proxy_root, "train", num_parts, K)
        q = torch.ones((num_parts, K), device=device) / float(K)

    if backbone_lr is not None or head_lr is not None:
        eff_backbone_lr = float(backbone_lr if backbone_lr is not None else lr)
        eff_head_lr = float(head_lr if head_lr is not None else lr)
        head_params = [p for p in model.fc.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": eff_backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": eff_head_lr})
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    tag = f"finetune_lr{lr}_bs{batch_size}_ep{epochs}_frac{train_frac}"
    if backbone_lr is not None or head_lr is not None:
        eff_backbone_lr = float(backbone_lr if backbone_lr is not None else lr)
        eff_head_lr = float(head_lr if head_lr is not None else lr)
        tag += f"_blr{eff_backbone_lr}_hlr{eff_head_lr}"
    if freeze_bn_stats:
        tag += "_bnfreeze1"
    tag_suffix = cfg.get("training", {}).get("tag_suffix")
    if tag_suffix:
        tag += f"_{tag_suffix}"
    run_dir = runs_dir / dataset_name / f"{reg_cfg['name']}_finetune" / f"seed{int(seed)}" / tag
    ensure_dir(run_dir)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return run_dir

    val_logits = []
    val_losses = []
    val_correct = []

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for ep in range(1, epochs + 1):
        model.train()
        if freeze_bn_stats:
            freeze_batchnorm_stats(model)
        total_loss = 0.0
        n_seen = 0
        clipped_count = 0
        total_count = 0
        for xb, yb, idx in train_loader:
            opt.zero_grad(set_to_none=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if use_amp and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb).squeeze(1)
                    losses = _binary_objective_losses(
                        logits,
                        yb,
                        bce,
                        label_smoothing=label_smoothing,
                        focal_gamma=focal_gamma,
                    )
            else:
                logits = model(xb).squeeze(1)
                losses = _binary_objective_losses(
                    logits,
                    yb,
                    bce,
                    label_smoothing=label_smoothing,
                    focal_gamma=focal_gamma,
                )

            if clip_loss > 0:
                if clip_alpha > 0:
                    losses_obj = torch.where(
                        losses <= clip_loss,
                        losses,
                        clip_loss + clip_alpha * (losses - clip_loss),
                    )
                else:
                    losses_obj = torch.clamp(losses, max=clip_loss)
                clipped_count += int((losses > clip_loss).sum().item())
                total_count += int(losses.numel())
            else:
                losses_obj = losses
            q_next = None

            if reg_cfg["objective"] == "erm":
                loss = losses_obj.mean()
            else:
                K = int(cfg["partitions"]["proxy"]["num_cells"])
                num_parts = len(parts)
                loss_total = 0.0
                q_next = []
                idx_np = idx.cpu().numpy()
                for m in range(num_parts):
                    g = torch.from_numpy(parts[m][idx_np]).to(device)
                    group_sums = torch.zeros(K, device=device)
                    group_counts = torch.zeros(K, device=device)
                    group_sums.scatter_add_(0, g, losses_obj)
                    group_counts.scatter_add_(0, g, torch.ones_like(losses_obj))
                    group_means = torch.where(group_counts > 0, group_sums / group_counts.clamp_min(1.0), torch.zeros_like(group_sums))
                    eta = float(reg_cfg.get("eta", 0.1))
                    with torch.no_grad():
                        q_m = q[m] * torch.exp(eta * group_means.detach())
                        q_m = q_m / q_m.sum().clamp_min(1e-12)
                    q_next.append(q_m)
                    loss_total += (q_m * group_means).sum()
                loss = loss_total / float(num_parts)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if q is not None and q_next is not None:
                q = torch.stack([qn.detach() for qn in q_next], dim=0)

            total_loss += float(loss.detach().cpu().item()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])

        train_loss = total_loss / max(n_seen, 1)
        frac_clipped_train = None
        if clip_loss > 0 and total_count > 0:
            frac_clipped_train = float(clipped_count) / float(total_count)

        logits_v, loss_v, corr_v = eval_logits_loss_correct(model, val_loader, device=device, use_amp=use_amp)
        frac_clipped_val = None
        if clip_loss > 0:
            frac_clipped_val = float(np.mean(loss_v > clip_loss))
        val_logits.append(logits_v)
        val_losses.append(loss_v)
        val_correct.append(corr_v)

        rec = {"epoch": int(ep), "train_loss": float(train_loss), "val_loss": float(loss_v.mean()), "val_acc": float(corr_v.mean())}
        if clip_loss > 0:
            rec["train_frac_clipped"] = frac_clipped_train
            rec["val_frac_clipped"] = frac_clipped_val
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        ckpt_path = run_dir / f"ckpt_epoch{ep:03d}.pt"
        torch.save({"epoch": ep, "model_state": model.state_dict()}, ckpt_path)

    np.save(run_dir / "val_logits_by_epoch.npy", np.stack(val_logits, axis=0).astype(np.float32))
    np.save(run_dir / "val_loss_by_epoch.npy", np.stack(val_losses, axis=0).astype(np.float32))
    np.save(run_dir / "val_correct_by_epoch.npy", np.stack(val_correct, axis=0).astype(np.uint8))

    cfg_out = {
        "dataset": dataset_name,
        "seed": int(seed),
        "tag": tag,
        "finetune": cfg["finetune"],
        "regime": reg_cfg["name"],
        "unfreeze_layers": unfreeze_layers,
        "partition_version": part_version,
        "partition_root": str(part_root),
        "clip_loss": clip_loss,
        "clip_alpha": clip_alpha,
        "label_smoothing": label_smoothing,
        "focal_gamma": focal_gamma,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "freeze_bn_stats": freeze_bn_stats,
    }
    cfg_path.write_text(json.dumps(cfg_out, indent=2))
    return run_dir


def main():
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
    dataset_name = cfg["dataset"]["name"]

    seeds = cfg["training"]["seeds"]
    if args.seed >= 0:
        seeds = [int(args.seed)]

    for seed in seeds:
        train_one(cfg, dataset_name, int(seed))


if __name__ == "__main__":
    main()
