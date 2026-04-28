import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..utils.io import ensure_dir


def _split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "validation", "val_bal"}:
        return "validation"
    if split in {"val_skew"}:
        return "val_skew"
    if split in {"train", "train_full"}:
        return "train"
    if split in {"test"}:
        return "test"
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


def _hash_cells(X: np.ndarray, R: np.ndarray, num_cells: int | None, chunk: int = 50000):
    n = int(X.shape[0])
    num_bits = int(R.shape[0])
    out = np.empty((n,), dtype=np.int32)
    for i in range(0, n, chunk):
        xb = np.asarray(X[i:i + chunk], dtype=np.float32)
        bits = (xb @ R.T) > 0
        ids = bits.astype(np.int32)
        cell_ids = np.zeros(ids.shape[0], dtype=np.int32)
        for b in range(num_bits):
            cell_ids |= (ids[:, b] << b)
        if num_cells is not None:
            cell_ids = cell_ids % int(num_cells)
        out[i:i + chunk] = cell_ids
    return out


def _proj_bins(X: np.ndarray, r: np.ndarray, num_cells: int, chunk: int = 50000):
    n = int(X.shape[0])
    proj = np.empty((n,), dtype=np.float64)
    for i in range(0, n, chunk):
        xb = np.asarray(X[i:i + chunk], dtype=np.float64)
        proj[i:i + chunk] = xb @ r
    order = np.argsort(proj, kind="mergesort")
    bins = np.empty((n,), dtype=np.int64)
    bin_ids = (np.arange(n, dtype=np.int64) * int(num_cells)) // int(n)
    bins[order] = bin_ids
    return bins


def _write_meta(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="train,val_skew,validation")
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50000)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]

    eval_cfg = cfg.get("partitions", {}).get("eval_banks", {})
    banks = eval_cfg.get("banks", ["A", "B"])
    proxy_seeds = eval_cfg.get("proxy_seeds", [101, 202])
    within_seeds = eval_cfg.get("within_label_seeds", [301, 402])
    dec_seeds = eval_cfg.get("decoupled_seeds", [501, 602])

    if len(proxy_seeds) < len(banks):
        raise ValueError(f"proxy_seeds has length {len(proxy_seeds)} < banks length {len(banks)}")
    if len(within_seeds) < len(banks):
        raise ValueError(f"within_label_seeds has length {len(within_seeds)} < banks length {len(banks)}")
    if len(dec_seeds) < len(banks):
        raise ValueError(f"decoupled_seeds has length {len(dec_seeds)} < banks length {len(banks)}")

    proxy_cfg = cfg["partitions"]["proxy"]
    dec_cfg = cfg["partitions"]["decoupled"]

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

    # Preload label info for within-label mapping.
    X0, y0 = _load_embeddings(feat_dir, splits[0])
    labels = np.unique(y0)
    num_labels = int(labels.size)

    num_cells_proxy = int(proxy_cfg["num_cells"])
    num_bits = int(np.ceil(np.log2(num_cells_proxy)))
    per_label_cells = int(num_cells_proxy // max(num_labels, 1))
    if num_cells_proxy % max(num_labels, 1) != 0:
        raise ValueError(f"num_cells {num_cells_proxy} must be divisible by num_labels {num_labels} for within-label eval.")

    # Build eval banks for three families.
    families = [
        ("global_hash", False, proxy_seeds),
        ("within_label_hash", True, within_seeds),
    ]

    for fam_name, within_label, seeds in families:
        fam_root = eval_root / fam_name
        ensure_dir(fam_root)
        meta = {
            "family": fam_name,
            "within_label": within_label,
            "num_partitions": int(proxy_cfg["num_partitions"]),
            "num_cells": num_cells_proxy,
            "num_bits": num_bits,
            "num_labels": num_labels,
            "per_label_cells": per_label_cells if within_label else None,
            "splits": splits,
            "banks": banks,
            "seeds": seeds,
        }
        _write_meta(fam_root / "meta.json", meta)

        for bank, seed in zip(banks, seeds):
            bank_root = fam_root / f"bank{bank}"
            ensure_dir(bank_root)
            for m in range(int(proxy_cfg["num_partitions"])):
                r_seed = int(seed) + 1000 * m
                rng = np.random.default_rng(r_seed)
                R = rng.standard_normal(size=(int(num_bits), int(X0.shape[1]))).astype(np.float64)
                R_path = bank_root / f"hash_R_m{m:02d}_seed{r_seed}.npy"
                if (not R_path.exists()) or int(args.overwrite):
                    np.save(R_path, R)

                for split in splits:
                    s = _split_name(split)
                    out_dir = bank_root / s
                    ensure_dir(out_dir)
                    out_path = out_dir / f"hash_m{m:02d}_K{num_cells_proxy}.npy"
                    if out_path.exists() and not int(args.overwrite):
                        continue
                    X, y = _load_embeddings(feat_dir, split)
                    raw = _hash_cells(X, R, num_cells=None, chunk=int(args.chunk))
                    if within_label:
                        cells = np.empty_like(raw)
                        for li, lab in enumerate(labels):
                            mask = y == lab
                            cells[mask] = li * per_label_cells + (raw[mask] % per_label_cells)
                    else:
                        cells = raw % num_cells_proxy
                    np.save(out_path, cells.astype(np.int32))

    # Decoupled family (random projection bins)
    dec_root = eval_root / "decoupled_proj"
    ensure_dir(dec_root)
    meta = {
        "family": "decoupled_proj",
        "num_partitions": int(dec_cfg["num_partitions"]),
        "num_cells": int(dec_cfg["num_cells"]),
        "splits": splits,
        "banks": banks,
        "seeds": dec_seeds,
    }
    _write_meta(dec_root / "meta.json", meta)

    for bank, seed in zip(banks, dec_seeds):
        bank_root = dec_root / f"bank{bank}"
        ensure_dir(bank_root)
        for m in range(int(dec_cfg["num_partitions"])):
            r_seed = int(seed) + 1000 * m
            rng = np.random.default_rng(r_seed)
            r = rng.standard_normal(int(X0.shape[1])).astype(np.float64)
            r = r / (np.linalg.norm(r) + 1e-12)
            r_path = bank_root / f"proj_r_m{m:02d}_seed{r_seed}.npy"
            if (not r_path.exists()) or int(args.overwrite):
                np.save(r_path, r)

            for split in splits:
                s = _split_name(split)
                out_dir = bank_root / s
                ensure_dir(out_dir)
                out_path = out_dir / f"proj_m{m:02d}_K{int(dec_cfg['num_cells'])}.npy"
                if out_path.exists() and not int(args.overwrite):
                    continue
                X, _y = _load_embeddings(feat_dir, split)
                bins = _proj_bins(X, r, int(dec_cfg["num_cells"]), chunk=int(args.chunk))
                np.save(out_path, bins.astype(np.int32))

    print(f"[build_eval_banks] wrote eval banks under {eval_root}")


if __name__ == "__main__":
    main()
