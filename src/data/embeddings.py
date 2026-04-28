import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast


def resolve_cache_dtype(name: str):
    name = str(name).lower()
    if name in {"float16", "fp16"}:
        return np.float16
    if name in {"float32", "fp32"}:
        return np.float32
    raise ValueError(f"Unknown cache dtype: {name}")


class DistilBertCLSFeaturizer(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        self.d_out = int(self.model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1].long()
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden[:, 0]


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


def build_backbone(name: str, max_token_length: int | None = None) -> Tuple[nn.Module, int, object]:
    name = str(name).lower()
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        d = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, d, weights.transforms()
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        d = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, d, weights.transforms()
    if name == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizerFast.from_pretrained(name)
        max_len = int(max_token_length or 300)

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

        model = DistilBertCLSFeaturizer(name)
        return model, int(model.d_out), transform
    raise ValueError(f"Unknown backbone: {name}. Try resnet18 or resnet50.")


def _open_memmap(path: Path, shape, dtype):
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def embed_subset_to_cache(
    subset,
    backbone: nn.Module,
    device: str,
    batch_size: int,
    num_workers: int,
    out_dir: Path,
    split_name: str,
    cache_dtype: np.dtype,
    embed_dim: int,
    spurious_index: int,
    store_metadata: bool = True,
    use_amp: bool = True,
) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(len(subset))
    if n <= 0:
        raise ValueError(f"Subset {split_name} has zero samples.")

    backbone.eval()
    backbone.to(device)

    # Infer metadata width from a single sample.
    x0, y0, m0 = subset[0]
    meta_dim = int(m0.shape[0]) if hasattr(m0, "shape") else int(len(m0))

    X_path = out_dir / f"X_{split_name}.npy"
    y_path = out_dir / f"y_{split_name}.npy"
    a_path = out_dir / f"a_{split_name}.npy"
    g_path = out_dir / f"g_{split_name}.npy"
    meta_path = out_dir / f"meta_{split_name}.npy"

    X = _open_memmap(X_path, shape=(n, int(embed_dim)), dtype=cache_dtype)
    y = np.zeros((n,), dtype=np.int64)
    a = np.zeros((n,), dtype=np.int64)
    g = None
    meta = np.zeros((n, meta_dim), dtype=np.int64) if store_metadata else None

    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )

    offset = 0
    for xb, yb, mb in loader:
        bs = int(yb.shape[0])
        xb = xb.to(device, non_blocking=True)
        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = backbone(xb)
        else:
            feats = backbone(xb)
        feats = feats.detach().cpu().numpy().astype(cache_dtype, copy=False)

        yb_np = yb.detach().cpu().numpy().astype(np.int64, copy=False)
        mb_np = mb.detach().cpu().numpy().astype(np.int64, copy=False)

        X[offset:offset + bs] = feats
        y[offset:offset + bs] = yb_np
        if store_metadata:
            meta[offset:offset + bs] = mb_np

        a_batch = mb_np[:, int(spurious_index)]
        a[offset:offset + bs] = a_batch

        offset += bs

    if offset != n:
        raise RuntimeError(f"Embedding write mismatch for {split_name}: wrote {offset} of {n}.")

    # Flush memmap
    del X

    np.save(y_path, y)
    np.save(a_path, a)
    a_max = int(a.max()) if a.size else 0
    g = y * (a_max + 1) + a
    np.save(g_path, g.astype(np.int64))
    if store_metadata:
        np.save(meta_path, meta)

    return {
        "n": n,
        "embed_dim": int(embed_dim),
        "meta_dim": meta_dim,
    }


def save_json(path: Path, payload: Dict):
    path.write_text(json.dumps(payload, indent=2))
