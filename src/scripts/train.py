import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import load_config
from ..utils.io import ensure_dir
from ..utils.seed import set_seed
from ..utils.stats import cvar_top_fraction


class MemmapDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None):
        self.X = X
        self.y = y
        if indices is None:
            self.indices = np.arange(len(y), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], dtype=np.float32)
        y = int(self.y[idx])
        return x, y, idx


def build_head(d_in: int, hidden_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(d_in, 1)
    return nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(hidden_dim, 1),
    )


def _binary_objective_losses(
    logits: torch.Tensor,
    y_int: torch.Tensor,
    bce: nn.BCEWithLogitsLoss,
    label_smoothing: float,
    focal_gamma: float,
    gce_q: float,
) -> torch.Tensor:
    """Per-example training losses for binary heads.

    Supports standard BCE, label-smoothed BCE, focal loss, or binary GCE.
    """
    active = int(label_smoothing > 0.0) + int(focal_gamma > 0.0) + int(gce_q > 0.0)
    if active > 1:
        raise ValueError("Use at most one of label_smoothing, focal_gamma, or gce_q.")

    if label_smoothing > 0.0:
        # Binary label smoothing toward uniform target 0.5.
        y_t = y_int.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return bce(logits, y_t)

    y_f = y_int.float()
    ce = bce(logits, y_f)
    probs = torch.sigmoid(logits)
    p_t = probs * y_f + (1.0 - probs) * (1.0 - y_f)
    if gce_q > 0.0:
        if not (0.0 < gce_q < 1.0):
            raise ValueError("gce_q must lie in (0,1) for binary GCE.")
        return (1.0 - torch.pow(p_t.clamp_min(1e-8), gce_q)) / gce_q
    if focal_gamma <= 0.0:
        return ce

    focal_weight = torch.pow((1.0 - p_t).clamp_min(1e-8), focal_gamma)
    return focal_weight * ce


@torch.no_grad()
def eval_logits_loss_correct(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ds = MemmapDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = []
    loss_all = []
    correct_all = []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for xb, yb, _ in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb).squeeze(1)
        losses = bce(logits, yb.float())
        preds = (logits >= 0).long()
        correct = (preds == yb).long()
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        loss_all.append(losses.detach().cpu().numpy().astype(np.float32))
        correct_all.append(correct.detach().cpu().numpy().astype(np.uint8))
    return np.concatenate(logits_all), np.concatenate(loss_all), np.concatenate(correct_all)


@torch.no_grad()
def eval_logits_on_indices(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    ds = MemmapDataset(X, y, indices=indices)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = []
    for xb, _yb, _idx in dl:
        xb = xb.to(device)
        logits = model(xb).squeeze(1)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits_all, axis=0)


def _quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    vals = np.asarray(values, dtype=np.float32)
    q = np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)
    edges = np.quantile(vals, q).astype(np.float32, copy=False)
    cuts = edges[1:-1]
    if cuts.size == 0:
        return np.zeros(vals.shape, dtype=np.int64)
    # Chunked searchsorted avoids large temporary allocations from np.digitize.
    out = np.empty(vals.shape, dtype=np.int64)
    chunk = 65536
    for s in range(0, vals.size, chunk):
        e = min(s + chunk, vals.size)
        out[s:e] = np.searchsorted(cuts, vals[s:e], side="left")
    return out


def compute_confidence_bins(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
    num_bins: int,
    within_pred_label: bool,
) -> np.ndarray:
    logits = eval_logits_on_indices(model, X, y, indices, batch_size=batch_size, device=device)
    logits = np.clip(logits, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    conf = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(np.int64)

    if within_pred_label:
        if int(num_bins) % 2 != 0:
            raise ValueError("within_pred_label requires num_bins divisible by 2.")
        per_label = int(num_bins) // 2
        bins_local = np.full_like(preds, fill_value=0, dtype=np.int64)
        for label in (0, 1):
            mask = preds == label
            if not np.any(mask):
                continue
            bins_l = _quantile_bins(conf[mask], per_label)
            bins_local[mask] = int(label) * per_label + bins_l
        bins = bins_local
    else:
        bins = _quantile_bins(conf, int(num_bins))

    bins_full = np.full((int(y.shape[0]),), -1, dtype=np.int64)
    bins_full[indices] = bins
    return bins_full


def _load_split_arrays(feat_dir: Path, split: str):
    if split == "train":
        X = np.load(feat_dir / "X_train.npy", mmap_mode="r")
        y = np.load(feat_dir / "y_train.npy")
        return X, y
    if split == "val_skew":
        X = np.load(feat_dir / "X_val_skew.npy", mmap_mode="r")
        y = np.load(feat_dir / "y_val_skew.npy")
        return X, y
    if split == "val_bal":
        X = np.load(feat_dir / "X_validation.npy", mmap_mode="r")
        y = np.load(feat_dir / "y_validation.npy")
        return X, y
    raise ValueError(f"Unknown split: {split}")


def _load_partitions(part_root: Path, split: str, num_partitions: int, K: int, prefix: str) -> List[np.ndarray]:
    split_dir = part_root / split
    parts = []
    for m in range(num_partitions):
        p = split_dir / f"{prefix}_m{m:02d}_K{int(K)}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing partition file: {p}")
        parts.append(np.load(p))
    return parts


def _load_eval_family_partitions(
    eval_root: Path,
    family: str,
    bank: str,
    split: str,
    num_partitions: int,
    K: int,
) -> List[np.ndarray]:
    fam_root = eval_root / family
    meta_path = fam_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing eval-family metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prefix = str(meta.get("prefix", "conf"))
    split_dir = fam_root / f"bank{bank}" / split
    parts = []
    for m in range(num_partitions):
        p = split_dir / f"{prefix}_m{m:02d}_K{int(K)}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing eval-family partition file: {p}")
        parts.append(np.load(p))
    return parts


def train_one(cfg, dataset_name: str, regime_name: str, seed: int) -> Path:
    set_seed(int(seed))
    torch.backends.cudnn.benchmark = True

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_dir = Path(cfg["project"]["runs_dir"])
    backbone = cfg["embeddings"]["backbone"]
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
    part_base = artifacts_dir / "partitions" / f"{dataset_name}_{backbone}"
    part_version = cfg.get("partitions", {}).get("version")
    part_root = part_base / str(part_version) if part_version and (part_base / str(part_version)).exists() else part_base
    eval_version = cfg.get("partitions", {}).get("eval_version")
    eval_base = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}"
    eval_root = eval_base / str(eval_version) if eval_version and (eval_base / str(eval_version)).exists() else eval_base
    proxy_family = str(cfg["partitions"]["proxy"].get("family", "random_hash"))
    noise_p = float(cfg["partitions"]["proxy"].get("noise_p", 0.0))
    noise_seed = int(cfg["partitions"]["proxy"].get("noise_seed", 0))
    fixed_source_family = str(cfg["partitions"]["proxy"].get("fixed_source_family", "conf_teacher_wpl"))
    fixed_source_bank = str(cfg["partitions"]["proxy"].get("fixed_source_bank", "A"))

    if not (feat_dir / "info.json").exists():
        raise FileNotFoundError(f"Missing embeddings at {feat_dir}. Run embed_cache first.")
    if proxy_family not in {"confidence_bins", "confidence_bins_fixed"} and not part_root.exists():
        raise FileNotFoundError(f"Missing partitions at {part_root}. Run build_partitions first.")

    X_train, y_train = _load_split_arrays(feat_dir, "train")
    train_idx_path = feat_dir / "train_sub_idx.npy"
    train_idx = np.load(train_idx_path) if train_idx_path.exists() else np.arange(len(y_train), dtype=np.int64)

    X_val, y_val = _load_split_arrays(feat_dir, "val_skew")

    d_in = int(X_train.shape[1])
    train_cfg = cfg["training"]
    reg_cfg = cfg["regime"]

    hidden_dim = int(train_cfg.get("hidden_dim", 0))
    dropout = float(train_cfg.get("dropout", 0.0))
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    shuffle_train = bool(train_cfg.get("shuffle", True))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 2048))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    label_smoothing = float(reg_cfg.get("label_smoothing", 0.0) or 0.0)
    focal_gamma = float(reg_cfg.get("focal_gamma", 0.0) or 0.0)
    gce_q = float(reg_cfg.get("gce_q", 0.0) or 0.0)
    clip_loss_cfg = float(reg_cfg.get("clip_loss", 0.0) or 0.0)
    clip_alpha_cfg = float(reg_cfg.get("clip_alpha", 0.0) or 0.0)

    # Optional dynamic clipping controller (disabled by default).
    dyn_cfg = reg_cfg.get("dynamic_clip", {})
    if not isinstance(dyn_cfg, dict):
        dyn_cfg = {}
    dyn_enabled = bool(dyn_cfg.get("enabled", False)) and (clip_loss_cfg > 0.0)
    dyn_target_distmass = float(dyn_cfg.get("target_distortion_mass", 0.0) or 0.0)
    dyn_alpha_init = float(dyn_cfg.get("alpha_init", clip_alpha_cfg) or clip_alpha_cfg)
    dyn_alpha_min = float(dyn_cfg.get("alpha_min", 0.0) or 0.0)
    dyn_alpha_max = float(dyn_cfg.get("alpha_max", 1.0) or 1.0)
    dyn_ema_beta = float(dyn_cfg.get("ema_beta", 0.8) or 0.8)
    dyn_warmup_epochs = int(dyn_cfg.get("warmup_epochs", 1) or 1)
    dyn_monotone_relax = bool(dyn_cfg.get("monotone_relax", True))
    dyn_max_alpha_step = float(dyn_cfg.get("max_alpha_step", 0.0) or 0.0)
    dyn_eps = float(dyn_cfg.get("eps", 1e-8) or 1e-8)
    # Optional secondary robustness guard layered on top of distortion control.
    # This does not change legacy behavior unless enabled in config.
    guard_cfg = dyn_cfg.get("secondary_tail_guard", {})
    if not isinstance(guard_cfg, dict):
        guard_cfg = {}
    guard_enabled = bool(guard_cfg.get("enabled", False))
    guard_tail_q = float(guard_cfg.get("tail_q", 0.1) or 0.1)
    guard_max_rho = float(guard_cfg.get("max_rho_cvar_clip", 1.0) or 1.0)
    guard_tail_delta = float(guard_cfg.get("max_tail_cvar_delta", np.inf))
    guard_relax_step = float(guard_cfg.get("relax_step", 0.05) or 0.05)
    guard_warmup_epochs = int(guard_cfg.get("warmup_epochs", dyn_warmup_epochs) or dyn_warmup_epochs)
    if dyn_enabled and guard_enabled:
        if not (0.0 < guard_tail_q <= 1.0):
            raise ValueError("secondary_tail_guard.tail_q must be in (0,1].")
        if guard_max_rho <= 0.0:
            raise ValueError("secondary_tail_guard.max_rho_cvar_clip must be > 0.")
        if guard_relax_step < 0.0:
            raise ValueError("secondary_tail_guard.relax_step must be >= 0.")
    if dyn_enabled:
        if dyn_target_distmass <= 0.0:
            raise ValueError("dynamic_clip.enabled=true requires target_distortion_mass > 0.")
        if not (0.0 <= dyn_alpha_min <= dyn_alpha_max <= 1.0):
            raise ValueError("dynamic_clip alpha bounds must satisfy 0 <= alpha_min <= alpha_max <= 1.")
        dyn_alpha_init = float(np.clip(dyn_alpha_init, dyn_alpha_min, dyn_alpha_max))

    device = cfg["compute"]["device"]
    num_workers = int(cfg["compute"]["num_workers"])

    tag = f"h{hidden_dim}_do{dropout}_lr{lr}_wd{weight_decay}_bs{batch_size}_ep{epochs}"
    if reg_cfg["objective"] == "rcgdro":
        tag += f"_K{int(cfg['partitions']['proxy']['num_cells'])}_eta{float(reg_cfg.get('eta', 0.1))}"
        if clip_loss_cfg > 0:
            clip_tag = int(round(clip_loss_cfg * 1000))
            tag += f"_clip{clip_tag}"
            if clip_alpha_cfg > 0:
                clipa_tag = int(round(clip_alpha_cfg * 100))
                tag += f"_clipa{clipa_tag}"
        if dyn_enabled:
            tag += (
                f"_dync_dm{int(round(dyn_target_distmass * 10000))}"
                f"_ainit{int(round(dyn_alpha_init * 100))}"
                f"_amin{int(round(dyn_alpha_min * 100))}"
                f"_b{int(round(dyn_ema_beta * 100))}"
                f"_wu{int(dyn_warmup_epochs)}"
            )
        if proxy_family == "confidence_bins":
            if bool(cfg["partitions"]["proxy"].get("within_pred_label", False)):
                tag += "_wpl"
            if noise_p > 0:
                tag += f"_noise{int(round(noise_p * 100))}"
        elif proxy_family == "confidence_bins_fixed":
            tag += f"_fixed{fixed_source_family}_{fixed_source_bank}"
            if noise_p > 0:
                tag += f"_noise{int(round(noise_p * 100))}"
    tag_suffix = train_cfg.get("tag_suffix")
    if tag_suffix:
        tag += f"_{tag_suffix}"

    run_dir = runs_dir / dataset_name / reg_cfg["name"] / f"seed{int(seed)}" / tag
    ensure_dir(run_dir)

    cfg_path = run_dir / "config.json"
    final_ckpt_path = run_dir / f"ckpt_epoch{int(epochs):03d}.pt"
    if final_ckpt_path.exists():
        return run_dir

    model = build_head(d_in, hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # Load proxy partitions for RCGDRO (or set up adaptive proxy bins).
    parts = []
    q = None
    if reg_cfg["objective"] == "rcgdro":
        num_parts = int(cfg["partitions"]["proxy"]["num_partitions"])
        K = int(cfg["partitions"]["proxy"]["num_cells"])
        if proxy_family == "confidence_bins":
            num_parts = 1
        elif proxy_family == "confidence_bins_fixed":
            num_parts = 1
            parts = _load_eval_family_partitions(
                eval_root=eval_root,
                family=fixed_source_family,
                bank=fixed_source_bank,
                split="train",
                num_partitions=num_parts,
                K=K,
            )
        else:
            proxy_root = part_root / "proxy"
            if proxy_family == "random_hash":
                prefix = "hash"
            elif proxy_family == "random_proj_bins":
                prefix = "proj"
            else:
                raise ValueError(f"Unsupported proxy family: {proxy_family}")
            parts = _load_partitions(proxy_root, "train", num_parts, K, prefix=prefix)
        q = torch.ones((num_parts, K), device=device) / float(K)

    ds_train = MemmapDataset(X_train, y_train, indices=train_idx)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_logits = []
    val_losses = []
    val_correct = []

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    bins_by_epoch = []
    bin_churns = []
    prev_bins = None
    noise_mask_full = None
    noise_bins_full = None
    controller_trace = []
    clip_alpha_current = float(dyn_alpha_init if dyn_enabled else clip_alpha_cfg)
    best_tail_cvar_seen = float("inf")
    if proxy_family in {"confidence_bins", "confidence_bins_fixed"} and noise_p > 0:
        rng = np.random.default_rng(int(noise_seed) + int(seed) * 1000)
        noise_mask_full = np.zeros((int(y_train.shape[0]),), dtype=bool)
        mask_local = rng.random(size=train_idx.shape[0]) < float(noise_p)
        noise_mask_full[train_idx] = mask_local
        noise_bins_full = np.full((int(y_train.shape[0]),), -1, dtype=np.int64)
        noise_bins_full[train_idx] = rng.integers(0, int(cfg["partitions"]["proxy"]["num_cells"]), size=train_idx.shape[0])
        if proxy_family == "confidence_bins_fixed" and parts:
            fixed_bins = np.asarray(parts[0]).astype(np.int64).copy()
            fixed_bins[noise_mask_full] = noise_bins_full[noise_mask_full]
            parts = [fixed_bins]

    for ep in range(1, epochs + 1):
        parts_epoch = parts
        churn = 0.0
        frac_clipped_train = None
        frac_clipped_val = None
        if reg_cfg["objective"] == "rcgdro" and proxy_family == "confidence_bins":
            within_pred_label = bool(cfg["partitions"]["proxy"].get("within_pred_label", False))
            bins_full = compute_confidence_bins(
                model=model,
                X=X_train,
                y=y_train,
                indices=train_idx,
                batch_size=eval_batch_size,
                device=device,
                num_bins=int(cfg["partitions"]["proxy"]["num_cells"]),
                within_pred_label=within_pred_label,
            )
            if prev_bins is not None:
                churn = float(np.mean(bins_full[train_idx] != prev_bins[train_idx]))
            prev_bins = bins_full
            if noise_mask_full is not None and noise_bins_full is not None:
                bins_full = bins_full.copy()
                bins_full[noise_mask_full] = noise_bins_full[noise_mask_full]
            bins_by_epoch.append(bins_full.astype(np.int32))
            bin_churns.append(float(churn))
            parts_epoch = [bins_full]
        model.train()
        total_loss = 0.0
        n_seen = 0
        clipped_count = 0
        total_count = 0
        clip_alpha_active = float(clip_alpha_current if dyn_enabled else clip_alpha_cfg)
        for xb, yb, idx in dl_train:
            opt.zero_grad(set_to_none=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb).squeeze(1)
            losses = _binary_objective_losses(
                logits=logits,
                y_int=yb,
                bce=bce,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                gce_q=gce_q,
            )
            clip_loss = float(clip_loss_cfg)
            clip_alpha = float(clip_alpha_active)
            if clip_loss > 0:
                if clip_alpha > 0:
                    losses_obj = torch.where(losses <= clip_loss, losses, clip_loss + clip_alpha * (losses - clip_loss))
                else:
                    losses_obj = torch.clamp(losses, max=clip_loss)
                clipped_count += int((losses > clip_loss).sum().item())
                total_count += int(losses.numel())
            else:
                losses_obj = losses
            q_next = None

            if reg_cfg["objective"] == "erm":
                # Use objective-transformed losses so ERM soft-clip regimes
                # actually optimize the clipped objective.
                loss = losses_obj.mean()
            else:
                K = int(cfg["partitions"]["proxy"]["num_cells"])
                num_parts = len(parts_epoch)
                loss_total = 0.0
                q_next = []
                idx_np = idx.cpu().numpy()
                for m in range(num_parts):
                    g = torch.from_numpy(parts_epoch[m][idx_np]).to(device)
                    group_sums = torch.zeros(K, device=device)
                    group_counts = torch.zeros(K, device=device)
                    group_sums.scatter_add_(0, g, losses_obj)
                    group_counts.scatter_add_(0, g, torch.ones_like(losses_obj))
                    group_means = torch.where(group_counts > 0, group_sums / group_counts.clamp_min(1.0), torch.zeros_like(group_sums))
                    eta = float(reg_cfg.get("eta", 0.1))
                    with torch.no_grad():
                        log_q = torch.log(q[m].clamp_min(1e-12)) + eta * group_means.detach()
                        log_q = log_q - torch.logsumexp(log_q, dim=0)
                        q_m = torch.exp(log_q)
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

        if clip_loss > 0 and total_count > 0:
            frac_clipped_train = float(clipped_count) / float(total_count)

        logits_v, loss_v, corr_v = eval_logits_loss_correct(model, X_val, y_val, batch_size=eval_batch_size, device=device)
        if clip_loss > 0:
            frac_clipped_val = float(np.mean(loss_v > clip_loss))
        val_logits.append(logits_v)
        val_losses.append(loss_v)
        val_correct.append(corr_v)

        # Tail proxy on unclipped validation losses (top-q CVaR).
        tail_cvar_val = float("nan")
        rho_cvar_clip_val = float("nan")
        if clip_loss > 0 and np.isfinite(clip_loss):
            tail_cvar_val = cvar_top_fraction(loss_v, float(guard_tail_q))
            if frac_clipped_val is not None and np.isfinite(frac_clipped_val):
                rho_cvar_clip_val = float(min(float(frac_clipped_val) / max(float(guard_tail_q), 1e-8), 1.0))
        if np.isfinite(tail_cvar_val):
            best_tail_cvar_seen = float(min(best_tail_cvar_seen, tail_cvar_val))

        rec = {
            "epoch": int(ep),
            "train_loss": float(train_loss),
            "val_loss": float(loss_v.mean()),
            "val_acc": float(corr_v.mean()),
        }
        if reg_cfg["objective"] == "rcgdro" and proxy_family == "confidence_bins":
            rec["train_bin_churn"] = float(churn)
        if clip_loss > 0:
            rec["train_frac_clipped"] = frac_clipped_train
            rec["val_frac_clipped"] = frac_clipped_val
            rec["clip_alpha_active"] = float(clip_alpha_active)
            rec["val_tail_cvar_q"] = float(guard_tail_q)
            rec["val_tail_cvar"] = float(tail_cvar_val)
            rec["val_rho_cvar_clip"] = float(rho_cvar_clip_val)

        mean_excess_val = None
        distortion_mass_val = None
        clip_alpha_next = float(clip_alpha_active)
        controller_status = "disabled"
        if clip_loss > 0 and dyn_enabled:
            mean_excess_val = float(np.maximum(loss_v - clip_loss, 0.0).mean())
            distortion_mass_val = float((1.0 - clip_alpha_active) * mean_excess_val)
            controller_status = "warmup" if ep <= int(dyn_warmup_epochs) else "active"
            if ep > int(dyn_warmup_epochs):
                if not np.isfinite(mean_excess_val):
                    controller_status = "invalid_mean_excess_hold"
                else:
                    alpha_raw = 1.0 - float(dyn_target_distmass) / max(float(mean_excess_val), float(dyn_eps))
                    alpha_raw = float(np.clip(alpha_raw, dyn_alpha_min, dyn_alpha_max))
                    clip_alpha_next = float(dyn_ema_beta * clip_alpha_active + (1.0 - dyn_ema_beta) * alpha_raw)
                    if dyn_max_alpha_step > 0.0:
                        lo = clip_alpha_active - float(dyn_max_alpha_step)
                        hi = clip_alpha_active + float(dyn_max_alpha_step)
                        clip_alpha_next = float(np.clip(clip_alpha_next, lo, hi))
                    if dyn_monotone_relax:
                        clip_alpha_next = float(max(clip_alpha_active, clip_alpha_next))
                    clip_alpha_next = float(np.clip(clip_alpha_next, dyn_alpha_min, dyn_alpha_max))
                    if not np.isfinite(clip_alpha_next):
                        clip_alpha_next = float(clip_alpha_active)
                        controller_status = "nonfinite_alpha_hold"
            guard_triggered = False
            guard_reasons = []
            if guard_enabled and ep > int(guard_warmup_epochs):
                # Guard 1: too much of CVaR-driving region is still clipped.
                if np.isfinite(rho_cvar_clip_val) and rho_cvar_clip_val > float(guard_max_rho):
                    guard_triggered = True
                    guard_reasons.append("rho")
                # Guard 2: tail CVaR drifts too far above best observed.
                if np.isfinite(tail_cvar_val) and np.isfinite(best_tail_cvar_seen):
                    tail_delta = float(tail_cvar_val - best_tail_cvar_seen)
                    if tail_delta > float(guard_tail_delta):
                        guard_triggered = True
                        guard_reasons.append("tail_delta")
                if guard_triggered:
                    clip_alpha_next = float(
                        np.clip(
                            max(clip_alpha_next, clip_alpha_active + float(guard_relax_step)),
                            dyn_alpha_min,
                            dyn_alpha_max,
                        )
                    )
                    suffix = "+".join(guard_reasons) if guard_reasons else "guard"
                    controller_status = f"{controller_status}|guard:{suffix}"
            rec["dynamic_clip_enabled"] = True
            rec["dynamic_target_distmass"] = float(dyn_target_distmass)
            rec["dynamic_mean_excess_val"] = float(mean_excess_val)
            rec["dynamic_distortion_mass_val"] = float(distortion_mass_val)
            rec["dynamic_clip_alpha_next"] = float(clip_alpha_next)
            rec["dynamic_controller_status"] = str(controller_status)
            rec["dynamic_guard_enabled"] = bool(guard_enabled)
            rec["dynamic_guard_tail_q"] = float(guard_tail_q)
            rec["dynamic_guard_max_rho"] = float(guard_max_rho)
            rec["dynamic_guard_max_tail_cvar_delta"] = float(guard_tail_delta)
            rec["dynamic_guard_relax_step"] = float(guard_relax_step)
            controller_trace.append(
                {
                    "epoch": int(ep),
                    "clip_alpha_active": float(clip_alpha_active),
                    "clip_alpha_next": float(clip_alpha_next),
                    "clip_loss": float(clip_loss),
                    "target_distortion_mass": float(dyn_target_distmass),
                    "mean_excess_val": float(mean_excess_val),
                    "distortion_mass_val": float(distortion_mass_val),
                    "val_frac_clipped": float(frac_clipped_val) if frac_clipped_val is not None else np.nan,
                    "val_tail_cvar": float(tail_cvar_val),
                    "val_rho_cvar_clip": float(rho_cvar_clip_val),
                    "best_tail_cvar_seen": float(best_tail_cvar_seen),
                    "controller_status": str(controller_status),
                }
            )
            clip_alpha_current = float(clip_alpha_next)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        ckpt_path = run_dir / f"ckpt_epoch{ep:03d}.pt"
        torch.save({"epoch": ep, "model_state": model.state_dict()}, ckpt_path)

    np.save(run_dir / "val_logits_by_epoch.npy", np.stack(val_logits, axis=0).astype(np.float32))
    np.save(run_dir / "val_loss_by_epoch.npy", np.stack(val_losses, axis=0).astype(np.float32))
    np.save(run_dir / "val_correct_by_epoch.npy", np.stack(val_correct, axis=0).astype(np.uint8))
    if reg_cfg["objective"] == "rcgdro" and proxy_family == "confidence_bins" and bins_by_epoch:
        np.save(run_dir / "train_proxy_bins_by_epoch.npy", np.stack(bins_by_epoch, axis=0))
        np.save(run_dir / "train_bin_churn_by_epoch.npy", np.asarray(bin_churns, dtype=np.float32))
    if dyn_enabled and controller_trace:
        pd_dyn = None
        try:
            import pandas as _pd
            pd_dyn = _pd.DataFrame(controller_trace)
            pd_dyn.to_csv(run_dir / "dynamic_clip_schedule.csv", index=False)
        except Exception:
            # Keep training robust even if pandas is unavailable.
            pass
        np.save(
            run_dir / "dynamic_clip_schedule.npy",
            np.asarray(
                [[
                    float(r["epoch"]),
                    float(r["clip_alpha_active"]),
                    float(r["clip_alpha_next"]),
                    float(r["mean_excess_val"]),
                    float(r["distortion_mass_val"]),
                ] for r in controller_trace],
                dtype=np.float32,
            ),
        )

    cfg_out = {
        "dataset": dataset_name,
        "regime": reg_cfg["name"],
        "seed": int(seed),
        "d_in": int(d_in),
        "tag": tag,
        "training": train_cfg,
        "regime_cfg": reg_cfg,
        "partition_version": part_version,
        "partition_root": str(part_root),
        "proxy_family": proxy_family,
        "proxy_noise_p": noise_p,
        "proxy_noise_seed": noise_seed,
        "clip_loss": float(clip_loss_cfg),
        "clip_alpha": float(clip_alpha_cfg),
        "label_smoothing": label_smoothing,
        "focal_gamma": focal_gamma,
        "gce_q": gce_q,
        "dynamic_clip": {
            "enabled": bool(dyn_enabled),
            "target_distortion_mass": float(dyn_target_distmass),
            "alpha_init": float(dyn_alpha_init),
            "alpha_min": float(dyn_alpha_min),
            "alpha_max": float(dyn_alpha_max),
            "ema_beta": float(dyn_ema_beta),
            "warmup_epochs": int(dyn_warmup_epochs),
            "monotone_relax": bool(dyn_monotone_relax),
            "max_alpha_step": float(dyn_max_alpha_step),
            "eps": float(dyn_eps),
            "secondary_tail_guard": {
                "enabled": bool(guard_enabled),
                "tail_q": float(guard_tail_q),
                "max_rho_cvar_clip": float(guard_max_rho),
                "max_tail_cvar_delta": float(guard_tail_delta),
                "relax_step": float(guard_relax_step),
                "warmup_epochs": int(guard_warmup_epochs),
            },
        },
    }
    cfg_path.write_text(json.dumps(cfg_out, indent=2))
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
    dataset_name = cfg["dataset"]["name"]
    regime_name = cfg["regime"]["name"]

    seeds = cfg["training"]["seeds"]
    if args.seed >= 0:
        seeds = [int(args.seed)]

    for seed in seeds:
        train_one(cfg, dataset_name, regime_name, int(seed))


if __name__ == "__main__":
    main()
