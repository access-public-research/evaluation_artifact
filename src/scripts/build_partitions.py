import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..utils.io import ensure_dir


def _split_name_for_file(split: str) -> str:
    split = split.lower()
    if split in {"val", "validation", "val_bal"}:
        return "validation"
    if split in {"val_skew"}:
        return "val_skew"
    if split in {"train_sub"}:
        return "train_sub"
    if split in {"train", "train_full"}:
        return "train"
    if split in {"test"}:
        return "test"
    raise ValueError(f"Unknown split: {split}")


def _load_embeddings(feat_dir: Path, split: str):
    split_name = _split_name_for_file(split)
    if split_name == "val_skew":
        X = np.load(feat_dir / "X_val_skew.npy", mmap_mode="r")
        y = np.load(feat_dir / "y_val_skew.npy")
        return X, y
    if split_name == "train_sub":
        X = np.load(feat_dir / "X_train.npy", mmap_mode="r")
        y = np.load(feat_dir / "y_train.npy")
        idx = np.load(feat_dir / "train_sub_idx.npy")
        return X[idx], y[idx]
    X = np.load(feat_dir / f"X_{split_name}.npy", mmap_mode="r")
    y = np.load(feat_dir / f"y_{split_name}.npy")
    return X, y


def _hash_cells_from_matrix(X: np.ndarray, R: np.ndarray, num_cells: int | None, chunk: int = 50000):
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


def _proj_bins_from_vector(X: np.ndarray, r: np.ndarray, num_cells: int, chunk: int = 50000):
    n = int(X.shape[0])
    proj = np.empty((n,), dtype=np.float64)
    for i in range(0, n, chunk):
        xb = np.asarray(X[i:i + chunk], dtype=np.float64)
        proj[i:i + chunk] = xb @ r
    # Rank-based equal-count bins
    order = np.argsort(proj, kind="mergesort")
    bins = np.empty((n,), dtype=np.int64)
    bin_ids = (np.arange(n, dtype=np.int64) * int(num_cells)) // int(n)
    bins[order] = bin_ids
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", default="train,val_skew,validation")
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=50000)
    args = parser.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"

    if not feat_dir.exists():
        raise FileNotFoundError(f"Embeddings not found at {feat_dir}. Run embed_cache first.")

    part_base = artifacts_dir / "partitions" / f"{dataset_name}_{backbone}"
    version = cfg.get("partitions", {}).get("version")
    part_root = part_base / str(version) if version else part_base
    proxy_root = part_root / "proxy"
    dec_root = part_root / "decoupled"
    ensure_dir(proxy_root)
    ensure_dir(dec_root)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Proxy partitions (random hash or random projection bins)
    proxy_cfg = cfg["partitions"]["proxy"]
    num_partitions = int(proxy_cfg["num_partitions"])
    num_cells = int(proxy_cfg["num_cells"])
    seed = int(proxy_cfg.get("seed", 0))
    family = str(proxy_cfg.get("family", "random_hash"))
    within_label = bool(proxy_cfg.get("within_label", False))
    num_bits = int(np.ceil(np.log2(num_cells)))

    proxy_meta = {
        "version": version,
        "root": str(part_root),
        "family": family,
        "within_label": within_label,
        "num_partitions": num_partitions,
        "num_cells": num_cells,
        "num_bits": num_bits,
        "seed": seed,
        "splits": splits,
    }
    (proxy_root / "meta.json").write_text(json.dumps(proxy_meta, indent=2))

    # Use first split to infer embedding dimension
    X0, y0 = _load_embeddings(feat_dir, splits[0])
    d = int(X0.shape[1])
    labels = np.unique(y0)
    num_labels = int(labels.size)

    if family == "random_hash":
        if within_label:
            if num_labels <= 0:
                raise ValueError("within_label partitions require at least one label.")
            if num_cells % num_labels != 0:
                raise ValueError(f"num_cells={num_cells} must be divisible by num_labels={num_labels} for within_label mode.")
            per_label_cells = int(num_cells // num_labels)
        else:
            per_label_cells = None
        proxy_meta["num_labels"] = num_labels
        proxy_meta["per_label_cells"] = per_label_cells

        for m in range(num_partitions):
            r_seed = seed + 1000 * m
            rng = np.random.default_rng(int(r_seed))
            R = rng.standard_normal(size=(int(num_bits), int(d))).astype(np.float64)
            R_path = proxy_root / f"hash_R_m{m:02d}_bits{num_bits}_seed{r_seed}.npy"
            if not R_path.exists() or int(args.overwrite):
                np.save(R_path, R)

            for split in splits:
                split_name = _split_name_for_file(split)
                out_dir = proxy_root / split_name
                ensure_dir(out_dir)
                out_path = out_dir / f"hash_m{m:02d}_K{num_cells}.npy"
                if out_path.exists() and not int(args.overwrite):
                    continue
                X, y = _load_embeddings(feat_dir, split)
                raw = _hash_cells_from_matrix(X, R, num_cells=None, chunk=int(args.chunk))
                if within_label:
                    cells = np.empty_like(raw)
                    for li, lab in enumerate(labels):
                        mask = y == lab
                        cells[mask] = li * int(per_label_cells) + (raw[mask] % int(per_label_cells))
                else:
                    cells = raw % int(num_cells)
                np.save(out_path, cells.astype(np.int32))
    elif family == "random_proj_bins":
        proxy_meta["num_labels"] = num_labels
        proxy_meta["per_label_cells"] = None
        if within_label:
            raise ValueError("within_label is not supported for random_proj_bins proxy family.")

        for m in range(num_partitions):
            r_seed = seed + 1000 * m
            rng = np.random.default_rng(int(r_seed))
            r = rng.standard_normal(d).astype(np.float64)
            r = r / (np.linalg.norm(r) + 1e-12)
            r_path = proxy_root / f"proj_r_m{m:02d}_seed{r_seed}.npy"
            if not r_path.exists() or int(args.overwrite):
                np.save(r_path, r)

            for split in splits:
                split_name = _split_name_for_file(split)
                out_dir = proxy_root / split_name
                ensure_dir(out_dir)
                out_path = out_dir / f"proj_m{m:02d}_K{num_cells}.npy"
                if out_path.exists() and not int(args.overwrite):
                    continue
                X, _y = _load_embeddings(feat_dir, split)
                bins = _proj_bins_from_vector(X, r, num_cells=num_cells, chunk=int(args.chunk))
                np.save(out_path, bins.astype(np.int32))
    else:
        raise ValueError(f"Unsupported proxy family: {family}")

    # Decoupled partitions (random projection bins)
    dec_cfg = cfg["partitions"]["decoupled"]
    dec_family = dec_cfg.get("family", "random_proj_bins")
    if dec_family != "random_proj_bins":
        raise ValueError(f"Unsupported decoupled family: {dec_family}")

    dec_parts = int(dec_cfg.get("num_partitions", 4))
    dec_cells = int(dec_cfg["num_cells"])
    dec_seed = int(dec_cfg.get("seed", seed + 123))

    dec_meta = {
        "version": version,
        "root": str(part_root),
        "family": dec_family,
        "num_partitions": dec_parts,
        "num_cells": dec_cells,
        "seed": dec_seed,
        "splits": splits,
    }
    (dec_root / "meta.json").write_text(json.dumps(dec_meta, indent=2))

    for m in range(dec_parts):
        r_seed = dec_seed + 1000 * m
        # Sample random vector using X from first split to get dimension.
        X0, _ = _load_embeddings(feat_dir, splits[0])
        rng = np.random.default_rng(int(r_seed))
        d = int(X0.shape[1])
        r = rng.standard_normal(d).astype(np.float64)
        r = r / (np.linalg.norm(r) + 1e-12)
        r_path = dec_root / f"proj_r_m{m:02d}_seed{r_seed}.npy"
        if not r_path.exists() or int(args.overwrite):
            np.save(r_path, r)

        for split in splits:
            split_name = _split_name_for_file(split)
            out_dir = dec_root / split_name
            ensure_dir(out_dir)
            out_path = out_dir / f"proj_m{m:02d}_K{dec_cells}.npy"
            if out_path.exists() and not int(args.overwrite):
                continue
            X, _y = _load_embeddings(feat_dir, split)
            bins = _proj_bins_from_vector(X, r, num_cells=dec_cells, chunk=int(args.chunk))
            np.save(out_path, bins.astype(np.int32))

    print(f"[build_partitions] wrote partitions under {part_root}")


if __name__ == "__main__":
    main()
