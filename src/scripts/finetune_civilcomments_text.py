import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.io import ensure_dir
from ..utils.seed import set_seed


def _normalize_text_input(text) -> str:
    if isinstance(text, str):
        return text
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="ignore")
    if isinstance(text, np.ndarray):
        if text.ndim == 0:
            return _normalize_text_input(text.item())
        return " ".join(_normalize_text_input(x) for x in text.tolist())
    if isinstance(text, (list, tuple)):
        if len(text) == 1:
            return _normalize_text_input(text[0])
        return " ".join(_normalize_text_input(x) for x in text)
    if hasattr(text, "item"):
        try:
            return _normalize_text_input(text.item())
        except Exception:
            pass
    if isinstance(text, float) and np.isnan(text):
        return ""
    return str(text)


def build_text_transform(model_name: str, max_token_length: int) -> object:
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    max_len = int(max_token_length)

    def transform(text: str) -> torch.Tensor:
        text = _normalize_text_input(text)
        toks = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        x = torch.stack((toks["input_ids"], toks["attention_mask"]), dim=2)
        return torch.squeeze(x, dim=0)

    return transform


class DistilBertBinaryClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = str(model_name)
        self.backbone = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1].long()
        logits = self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits.squeeze(-1)


class SubsetOverrideYWithIndex(Dataset):
    def __init__(self, base, indices: np.ndarray, y_override: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y_override, dtype=np.int64)
        if self.indices.shape[0] != self.y.shape[0]:
            raise ValueError("indices and y_override must have same length.")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        x, _y, _m = self.base[base_idx]
        return x, int(self.y[i]), base_idx


@torch.no_grad()
def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    logits_all = []
    losses_all = []
    correct_all = []
    for xb, yb, _idx in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")):
            logits = model(xb)
            losses = bce(logits, yb.float())
        preds = (logits >= 0).long()
        corr = (preds == yb).long()
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        losses_all.append(losses.detach().cpu().numpy().astype(np.float32))
        correct_all.append(corr.detach().cpu().numpy().astype(np.uint8))
    return np.concatenate(logits_all), np.concatenate(losses_all), np.concatenate(correct_all)


def _apply_proxy(losses: torch.Tensor, clip_loss: float, clip_alpha: float) -> Tuple[torch.Tensor, int]:
    if clip_loss <= 0.0:
        return losses, 0
    clipped = losses > clip_loss
    if clip_alpha > 0.0:
        prox = torch.where(
            losses <= clip_loss,
            losses,
            clip_loss + clip_alpha * (losses - clip_loss),
        )
    else:
        prox = torch.clamp(losses, max=clip_loss)
    return prox, int(clipped.sum().item())


def _run_dir(cfg: dict, dataset_name: str, regime_name: str, seed: int, tag: str) -> Path:
    runs_root = Path(cfg["project"]["runs_dir"])
    return runs_root / f"{dataset_name}_e2e" / regime_name / f"seed{int(seed)}" / tag


def _resolve_val_surface_dir(cfg: dict, dataset_name: str, backbone_name: str) -> Path:
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    skew_cfg = cfg.get("validation", {}).get("skewed_val", {})
    override_name = str(skew_cfg.get("artifact_name_override", "")).strip()
    if override_name:
        return artifacts_dir / "embeds" / override_name
    return artifacts_dir / "embeds" / f"{dataset_name}_{backbone_name}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one(
    cfg: dict,
    dataset_name: str,
    regime_name: str,
    seed: int,
    *,
    overwrite: bool,
    epochs_override: int,
    batch_size_override: int,
    eval_batch_size_override: int,
    amp_override: int,
) -> Path:
    if dataset_name != "civilcomments":
        raise ValueError("This trainer is specific to civilcomments.")

    set_seed(int(seed))
    torch.backends.cudnn.benchmark = True

    backbone_name = str(cfg["embeddings"]["backbone"])
    if backbone_name != "distilbert-base-uncased":
        raise ValueError("This pilot only supports distilbert-base-uncased.")

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    use_amp = bool(cfg["compute"].get("amp", True))
    if int(amp_override) in (0, 1):
        use_amp = bool(int(amp_override))

    data_dir = cfg["dataset"].get("data_dir") or cfg["paths"]["wilds_data_dir"]
    ds = load_wilds_dataset(cfg["dataset"]["wilds_dataset"], data_dir, download=False)

    max_token_length = int(cfg["dataset"].get("max_token_length", 300))
    transform = build_text_transform(backbone_name, max_token_length=max_token_length)

    train_subset = ds.get_subset("train", frac=1.0, transform=transform)

    feat_dir = _resolve_val_surface_dir(cfg, dataset_name, backbone_name)
    splits = json.loads((feat_dir / "splits.json").read_text(encoding="utf-8"))
    source_split = str(splits.get("val_skew_source_split", "validation"))
    val_idx = np.load(feat_dir / "val_skew_idx.npy")
    y_val = np.load(feat_dir / "y_val_skew.npy")
    val_base = ds.get_subset("val" if source_split == "validation" else source_split, frac=1.0, transform=transform)
    val_subset = SubsetOverrideYWithIndex(val_base, val_idx, y_val)

    batch_size = int(cfg["finetune"]["batch_size"])
    if int(batch_size_override) > 0:
        batch_size = int(batch_size_override)
    eval_batch_size = int(cfg["finetune"]["eval_batch_size"])
    if int(eval_batch_size_override) > 0:
        eval_batch_size = int(eval_batch_size_override)
    epochs = int(cfg["finetune"]["epochs"])
    if int(epochs_override) > 0:
        epochs = int(epochs_override)
    lr = float(cfg["finetune"]["lr"])
    weight_decay = float(cfg["finetune"].get("weight_decay", 0.01))
    grad_clip = float(cfg["finetune"].get("grad_clip", 1.0))
    num_workers = int(cfg["compute"].get("num_workers", 0))
    tag = str(cfg["training"]["tag_suffix"])

    run_dir = _run_dir(cfg, dataset_name, regime_name, int(seed), tag)
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)

    done_path = run_dir / "done.json"
    if done_path.exists() and not overwrite:
        return run_dir

    heartbeat_path = run_dir / "heartbeat.json"
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    regime_cfg = cfg["regime"]
    clip_loss = float(regime_cfg.get("clip_loss", 0.0) or 0.0)
    clip_alpha = float(regime_cfg.get("clip_alpha", 0.0) or 0.0)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = DistilBertBinaryClassifier(backbone_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.startswith("cuda"))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    val_logits = []
    val_standard_losses = []
    val_proxy_losses = []
    val_correct = []

    _write_json(
        heartbeat_path,
        {
            "status": "started",
            "seed": int(seed),
            "regime": regime_name,
            "wall_time": time.time(),
            "epoch": 0,
            "step": 0,
            "device": device,
            "amp": bool(use_amp),
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "epochs": int(epochs),
        },
    )

    try:
        for ep in range(1, epochs + 1):
            model.train()
            total_proxy_loss = 0.0
            total_standard_loss = 0.0
            total_seen = 0
            total_clipped = 0
            for step, (xb, yb, _idx) in enumerate(train_loader, start=1):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")):
                    logits = model(xb)
                    standard_losses = bce(logits, yb.float())
                    proxy_losses, clipped_count = _apply_proxy(standard_losses, clip_loss=clip_loss, clip_alpha=clip_alpha)
                    loss = proxy_losses.mean()

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip > 0.0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()

                bs = int(xb.shape[0])
                total_proxy_loss += float(proxy_losses.detach().mean().cpu().item()) * bs
                total_standard_loss += float(standard_losses.detach().mean().cpu().item()) * bs
                total_seen += bs
                total_clipped += int(clipped_count)

                if step == 1 or step % 200 == 0:
                    _write_json(
                        heartbeat_path,
                        {
                            "status": "running",
                            "seed": int(seed),
                            "regime": regime_name,
                            "wall_time": time.time(),
                            "epoch": int(ep),
                            "step": int(step),
                            "train_seen": int(total_seen),
                            "batch_size": int(batch_size),
                            "amp": bool(use_amp),
                        },
                    )

            logits_v, standard_v, correct_v = eval_loader(model, val_loader, device=device, use_amp=use_amp)
            proxy_v = standard_v.copy()
            if clip_loss > 0.0:
                above = proxy_v > clip_loss
                proxy_v[above] = clip_loss + clip_alpha * (proxy_v[above] - clip_loss)

            val_logits.append(logits_v)
            val_standard_losses.append(standard_v)
            val_proxy_losses.append(proxy_v)
            val_correct.append(correct_v)

            train_proxy_loss = total_proxy_loss / max(total_seen, 1)
            train_standard_loss = total_standard_loss / max(total_seen, 1)
            train_frac_clipped = float(total_clipped) / float(max(total_seen, 1)) if clip_loss > 0.0 else 0.0
            val_frac_clipped = float(np.mean(standard_v > clip_loss)) if clip_loss > 0.0 else 0.0
            rec = {
                "epoch": int(ep),
                "train_proxy_loss": float(train_proxy_loss),
                "train_standard_loss": float(train_standard_loss),
                "train_frac_clipped": float(train_frac_clipped),
                "val_proxy_loss": float(np.mean(proxy_v)),
                "val_standard_loss": float(np.mean(standard_v)),
                "val_acc": float(np.mean(correct_v)),
                "val_frac_clipped": float(val_frac_clipped),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            torch.save(
                {
                    "epoch": int(ep),
                    "model_state": model.state_dict(),
                },
                run_dir / f"ckpt_epoch{int(ep):03d}.pt",
            )
            _write_json(
                heartbeat_path,
                {
                    "status": "epoch_done",
                    "seed": int(seed),
                    "regime": regime_name,
                    "wall_time": time.time(),
                    "epoch": int(ep),
                    "step": 0,
                    "val_proxy_loss": float(np.mean(proxy_v)),
                    "val_standard_loss": float(np.mean(standard_v)),
                    "val_acc": float(np.mean(correct_v)),
                    "batch_size": int(batch_size),
                    "amp": bool(use_amp),
                },
            )

        np.save(run_dir / "val_logits_by_epoch.npy", np.stack(val_logits, axis=0).astype(np.float32))
        np.save(run_dir / "val_standard_loss_by_epoch.npy", np.stack(val_standard_losses, axis=0).astype(np.float32))
        np.save(run_dir / "val_proxy_loss_by_epoch.npy", np.stack(val_proxy_losses, axis=0).astype(np.float32))
        np.save(run_dir / "val_correct_by_epoch.npy", np.stack(val_correct, axis=0).astype(np.uint8))

        cfg_out = {
            "dataset": dataset_name,
            "seed": int(seed),
            "tag": tag,
            "backbone": backbone_name,
            "val_surface_dir": str(feat_dir),
            "regime": regime_name,
            "clip_loss": float(clip_loss),
            "clip_alpha": float(clip_alpha),
            "device": device,
            "use_amp": bool(use_amp),
            "finetune": {
                "batch_size": int(batch_size),
                "eval_batch_size": int(eval_batch_size),
                "epochs": int(epochs),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": float(grad_clip),
            },
        }
        _write_json(run_dir / "config.json", cfg_out)
        _write_json(
            done_path,
            {
                "status": "completed",
                "seed": int(seed),
                "regime": regime_name,
                "wall_time": time.time(),
                "epochs": int(epochs),
            },
        )
        _write_json(
            heartbeat_path,
            {
                "status": "completed",
                "seed": int(seed),
                "regime": regime_name,
                "wall_time": time.time(),
                "epoch": int(epochs),
                "step": 0,
                "batch_size": int(batch_size),
                "amp": bool(use_amp),
            },
        )
        return run_dir
    except Exception as exc:
        _write_json(
            heartbeat_path,
            {
                "status": "failed",
                "seed": int(seed),
                "regime": regime_name,
                "wall_time": time.time(),
                "error": repr(exc),
                "batch_size": int(batch_size),
                "amp": bool(use_amp),
            },
        )
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--epochs_override", type=int, default=-1)
    ap.add_argument("--batch_size_override", type=int, default=-1)
    ap.add_argument("--eval_batch_size_override", type=int, default=-1)
    ap.add_argument("--amp_override", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        dataset_path=f"configs/datasets/{args.dataset}.yaml",
        regime_path=f"configs/regimes/{args.regime}.yaml",
    )
    dataset_name = cfg["dataset"]["name"]

    seeds = list(cfg["training"]["seeds"])
    if int(args.seed) >= 0:
        seeds = [int(args.seed)]

    for seed in seeds:
        train_one(
            cfg,
            dataset_name,
            str(cfg["regime"]["name"]),
            int(seed),
            overwrite=bool(int(args.overwrite)),
            epochs_override=int(args.epochs_override),
            batch_size_override=int(args.batch_size_override),
            eval_batch_size_override=int(args.eval_batch_size_override),
            amp_override=int(args.amp_override),
        )


if __name__ == "__main__":
    main()
