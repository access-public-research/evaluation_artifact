import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.stats import ci95_mean


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


class WildsWithDomain(Dataset):
    def __init__(self, base, domain_col: int):
        self.base = base
        self.domain_col = int(domain_col)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, meta = self.base[int(i)]
        if hasattr(meta, "detach"):
            meta_np = meta.detach().cpu().numpy()
        else:
            meta_np = np.asarray(meta)
        d = int(meta_np[self.domain_col])
        return x, int(y), d


def build_head(d_in: int, hidden_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(d_in, 1)
    return nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(hidden_dim, 1),
    )


def build_backbone(backbone: str) -> tuple[nn.Module, torchvision.transforms.Compose]:
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
    raise ValueError(f"Unsupported backbone for full-model eval: {backbone}")


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


@torch.no_grad()
def eval_logits_full(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    y_all = []
    d_all = []
    for xb, yb, db in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(1)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        y_all.append(np.asarray(yb, dtype=np.int64))
        d_all.append(np.asarray(db, dtype=np.int64))
    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(d_all, axis=0),
    )


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


def _acc_by_domain(logits: np.ndarray, y: np.ndarray, domains: np.ndarray) -> Dict[int, float]:
    preds = (logits >= 0).astype(np.int64)
    out = {}
    for d in np.unique(domains):
        mask = domains == d
        if not np.any(mask):
            continue
        out[int(d)] = float(np.mean(preds[mask] == y[mask]))
    return out


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(ci95_mean(x))


def _bce_losses_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))


def _loss_by_domain(logits: np.ndarray, y: np.ndarray, domains: np.ndarray) -> Dict[int, float]:
    losses = _bce_losses_from_logits(logits, y)
    out = {}
    for d in np.unique(domains):
        mask = domains == d
        if not np.any(mask):
            continue
        out[int(d)] = float(np.mean(losses[mask]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--tag_filter", default="v7confclip")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_summary", default="")
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_dir = Path(cfg["project"]["runs_dir"])

    hidden_dim = int(cfg["training"].get("hidden_dim", 0))
    dropout = float(cfg["training"].get("dropout", 0.0))
    eval_batch = int(cfg.get("finetune", {}).get("eval_batch_size", cfg["training"].get("eval_batch_size", 2048)))
    device = cfg["compute"]["device"]
    num_workers = int(cfg["compute"].get("num_workers", 0)) if args.num_workers is None else int(args.num_workers)
    data_dir = cfg["dataset"].get("data_dir") or cfg["paths"]["wilds_data_dir"]
    wilds_name = cfg["dataset"]["wilds_dataset"]
    domain_field = str(cfg["dataset"].get("spurious_metadata_field", "hospital"))

    df = pd.read_csv(args.summary_csv)
    for col in ("regime", "seed", "epoch"):
        if col not in df.columns:
            raise ValueError(f"summary_csv missing required column: {col}")
    has_tag = "tag" in df.columns

    # Lazy caches for the two checkpoint families.
    X_val = y_val = a_val = X_test = y_test = a_test = None
    full_val_loader = full_test_loader = None

    merge_keys = ["regime", "seed", "epoch"] + (["tag"] if has_tag else [])
    unique_df = df[merge_keys].drop_duplicates().reset_index(drop=True)

    out_rows = []
    for _, row in unique_df.iterrows():
        regime = str(row["regime"])
        seed = int(row["seed"])
        epoch = int(row["epoch"])
        requested_tag = str(row["tag"]).strip() if has_tag and pd.notna(row["tag"]) else None

        reg_dir = runs_dir / dataset_name / regime / f"seed{seed}"
        if not reg_dir.exists():
            continue
        tag_dir = _resolve_tag_dir(reg_dir, requested_tag=requested_tag, tag_filter=args.tag_filter)
        ckpt = tag_dir / f"ckpt_epoch{epoch:03d}.pt"
        if not ckpt.exists():
            continue

        state = torch.load(ckpt, map_location=device)
        model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        run_cfg = json.loads((tag_dir / "config.json").read_text(encoding="utf-8"))
        keys = list(model_state.keys())
        is_full_model = any(k.startswith("conv1.") or k.startswith("layer1.") or k.startswith("fc.") for k in keys)

        if is_full_model:
            if full_val_loader is None or full_test_loader is None:
                ds = load_wilds_dataset(wilds_name, data_dir, download=False)
                fields = list(getattr(ds, "metadata_fields", []))
                if domain_field not in fields:
                    raise ValueError(f"Domain field '{domain_field}' not found in metadata fields: {fields}")
                domain_col = int(fields.index(domain_field))
                model_tmp, tfm = build_backbone(backbone)
                del model_tmp
                val_base = ds.get_subset("val", frac=1.0, transform=tfm)
                test_base = ds.get_subset("test", frac=1.0, transform=tfm)
                val_ds = WildsWithDomain(val_base, domain_col=domain_col)
                test_ds = WildsWithDomain(test_base, domain_col=domain_col)
                full_val_loader = DataLoader(
                    val_ds, batch_size=eval_batch, shuffle=False, num_workers=num_workers, pin_memory=True
                )
                full_test_loader = DataLoader(
                    test_ds, batch_size=eval_batch, shuffle=False, num_workers=num_workers, pin_memory=True
                )

            model, _tfm_unused = build_backbone(backbone)
            model = model.to(device)
            model.load_state_dict(model_state, strict=True)
            logits_val, y_val_run, a_val_run = eval_logits_full(model, full_val_loader, device)
            logits_test, y_test_run, a_test_run = eval_logits_full(model, full_test_loader, device)
        else:
            if X_val is None:
                feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
                X_val = np.load(feat_dir / "X_validation.npy", mmap_mode="r")
                y_val = np.load(feat_dir / "y_validation.npy")
                a_val = np.load(feat_dir / "a_validation.npy")
                X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
                y_test = np.load(feat_dir / "y_test.npy")
                a_test = np.load(feat_dir / "a_test.npy")
            d_in = int(run_cfg.get("d_in", int(X_val.shape[1])))
            hidden_dim_run = int(run_cfg.get("training", {}).get("hidden_dim", hidden_dim))
            dropout_run = float(run_cfg.get("training", {}).get("dropout", dropout))
            eval_batch_run = int(run_cfg.get("training", {}).get("eval_batch_size", eval_batch))
            model = build_head(d_in, hidden_dim_run, dropout_run).to(device)
            model.load_state_dict(model_state, strict=True)
            logits_val = eval_logits(model, X_val, y_val, eval_batch_run, device)
            logits_test = eval_logits(model, X_test, y_test, eval_batch_run, device)
            y_val_run = y_val
            y_test_run = y_test
            a_val_run = a_val
            a_test_run = a_test

        acc_val = float(np.mean((logits_val >= 0) == y_val_run))
        acc_test = float(np.mean((logits_test >= 0) == y_test_run))
        loss_val = float(np.mean(_bce_losses_from_logits(logits_val, y_val_run)))
        loss_test = float(np.mean(_bce_losses_from_logits(logits_test, y_test_run)))
        acc_val_domains = _acc_by_domain(logits_val, y_val_run, a_val_run)
        acc_test_domains = _acc_by_domain(logits_test, y_test_run, a_test_run)
        loss_val_domains = _loss_by_domain(logits_val, y_val_run, a_val_run)
        loss_test_domains = _loss_by_domain(logits_test, y_test_run, a_test_run)

        row_out = {
            "regime": regime,
            "seed": seed,
            "epoch": epoch,
            "val_acc": acc_val,
            "val_loss": loss_val,
            "test_acc": acc_test,
            "test_loss": loss_test,
            "val_worst_domain_acc": float(min(acc_val_domains.values())) if acc_val_domains else np.nan,
            "val_worst_domain_loss": float(max(loss_val_domains.values())) if loss_val_domains else np.nan,
            "test_worst_domain_acc": float(min(acc_test_domains.values())) if acc_test_domains else np.nan,
            "test_worst_domain_loss": float(max(loss_test_domains.values())) if loss_test_domains else np.nan,
        }
        if has_tag:
            row_out["tag"] = requested_tag or ""
        for d, v in acc_val_domains.items():
            row_out[f"val_hosp_{d}_acc"] = v
        for d, v in acc_test_domains.items():
            row_out[f"test_hosp_{d}_acc"] = v
        for d, v in loss_val_domains.items():
            row_out[f"val_hosp_{d}_loss"] = v
        for d, v in loss_test_domains.items():
            row_out[f"test_hosp_{d}_loss"] = v
        out_rows.append(row_out)

    # Write CSV
    if not out_rows:
        raise RuntimeError("No Camelyon domain results computed; check summary_csv and runs.")
    df_eval = pd.DataFrame(out_rows)
    df_out = df[merge_keys].merge(df_eval, on=merge_keys, how="left")
    all_keys = sorted(df_out.columns.tolist())
    out_path = args.out_csv or str(artifacts_dir / "metrics" / f"{dataset_name}_{backbone}_domain_acc.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(all_keys) + "\n")
        for _, r in df_out.iterrows():
            f.write(",".join(str(r.get(k, "")) for k in all_keys) + "\n")
    print(f"[camelyon_domain_eval] wrote {out_path}")

    out_summary = str(args.out_summary).strip()
    if out_summary:
        numeric_cols = [
            c
            for c in df_out.columns
            if c not in {"seed", "epoch"} and pd.api.types.is_numeric_dtype(df_out[c])
        ]
        summary_rows = []
        for regime, sub in df_out.groupby("regime", dropna=False):
            rec: Dict[str, object] = {"regime": regime, "n": int(sub.shape[0])}
            for col in numeric_cols:
                mean, ci = _mean_ci(sub[col].to_numpy())
                rec[f"{col}_mean"] = mean
                rec[f"{col}_ci"] = ci
            summary_rows.append(rec)
        out_summary_path = Path(out_summary)
        out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).sort_values("regime").to_csv(out_summary_path, index=False)
        print(f"[camelyon_domain_eval] wrote {out_summary_path}")


if __name__ == "__main__":
    main()
