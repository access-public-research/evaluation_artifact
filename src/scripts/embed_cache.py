import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..data.embeddings import build_backbone, embed_subset_to_cache, resolve_cache_dtype, save_json
from ..data.skewed_val import make_skewed_val_indices
from ..data.wilds_loader import get_metadata_fields, load_wilds_dataset
from ..utils.io import ensure_dir
from ..utils.seed import set_seed


def _split_name_for_file(split: str) -> str:
    split = split.lower()
    if split in {"val", "validation", "val_bal"}:
        return "validation"
    if split in {"train", "train_full"}:
        return "train"
    if split in {"test"}:
        return "test"
    raise ValueError(f"Unknown split: {split}")


def _resolve_spurious_index(meta_fields, dataset_cfg):
    if dataset_cfg.get("spurious_metadata_index") is not None:
        return int(dataset_cfg["spurious_metadata_index"])
    field = dataset_cfg.get("spurious_metadata_field")
    if field is None:
        raise ValueError("Dataset config must set spurious_metadata_field or spurious_metadata_index.")
    if not meta_fields:
        raise ValueError("Dataset does not expose metadata fields; cannot resolve spurious field.")
    if field not in meta_fields:
        raise ValueError(f"Spurious field '{field}' not found in metadata fields: {meta_fields}")
    return int(meta_fields.index(field))


def _load_split_arrays(feat_dir: Path, split: str):
    split_name = _split_name_for_file(split)
    X = np.load(feat_dir / f"X_{split_name}.npy", mmap_mode="r")
    y = np.load(feat_dir / f"y_{split_name}.npy")
    a = np.load(feat_dir / f"a_{split_name}.npy")
    g = np.load(feat_dir / f"g_{split_name}.npy")
    return X, y, a, g


def _group_counts(g: np.ndarray) -> dict:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--download", type=int, default=0)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_cfg = cfg["dataset"]

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    ensure_dir(artifacts_dir)

    backbone_name = cfg["embeddings"]["backbone"]
    cache_dtype = resolve_cache_dtype(cfg["embeddings"]["cache_dtype"])
    batch_size = int(cfg["embeddings"]["batch_size"])
    store_metadata = bool(cfg["embeddings"].get("store_metadata", True))
    store_train_sub = bool(cfg["embeddings"].get("store_train_sub", False))

    dataset_name = dataset_cfg["name"]
    wilds_name = dataset_cfg["wilds_dataset"]
    data_dir = dataset_cfg.get("data_dir") or cfg["paths"]["wilds_data_dir"]

    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone_name}"
    ensure_dir(feat_dir)

    info_path = feat_dir / "info.json"
    splits_path = feat_dir / "splits.json"

    required = [
        "X_train.npy", "y_train.npy", "a_train.npy", "g_train.npy",
        "X_validation.npy", "y_validation.npy", "a_validation.npy", "g_validation.npy",
        "X_test.npy", "y_test.npy", "a_test.npy", "g_test.npy",
        "X_val_skew.npy", "y_val_skew.npy", "y_val_skew_true.npy", "a_val_skew.npy", "g_val_skew.npy",
        "val_skew_idx.npy", "train_sub_idx.npy", "splits.json",
    ]
    if store_metadata:
        required.extend([
            "meta_train.npy", "meta_validation.npy", "meta_test.npy",
        ])
    if store_train_sub:
        required.extend([
            "X_train_sub.npy", "y_train_sub.npy", "a_train_sub.npy", "g_train_sub.npy",
        ])

    if info_path.exists() and not int(args.overwrite):
        missing = [f for f in required if not (feat_dir / f).exists()]
        if not missing:
            print(f"[embed_cache] found existing cache: {info_path} (use --overwrite 1 to rebuild)")
            return
        print(f"[embed_cache] cache incomplete; missing {len(missing)} files; rebuilding")

    print(f"[embed_cache] loading WILDS dataset='{wilds_name}' from {data_dir}")
    ds = load_wilds_dataset(wilds_name, data_dir, download=bool(int(args.download)))

    meta_fields = get_metadata_fields(ds)
    spurious_idx = _resolve_spurious_index(meta_fields, dataset_cfg)

    device = cfg["compute"]["device"]
    num_workers = int(cfg["compute"]["num_workers"])
    use_amp = bool(cfg["compute"].get("amp", True))

    torch.backends.cudnn.benchmark = True

    max_token_length = dataset_cfg.get("max_token_length")
    backbone, embed_dim, transform = build_backbone(backbone_name, max_token_length=max_token_length)

    # Optional split subsampling for debug.
    frac_cfg = cfg.get("data", {}).get("fractions", {})
    subset_seed = int(cfg.get("data", {}).get("subset_seed", 0))
    frac_train = float(frac_cfg.get("train", 1.0))
    frac_val = float(frac_cfg.get("val", 1.0))
    frac_test = float(frac_cfg.get("test", 1.0))

    # Embed each split.
    split_specs = [
        ("train", frac_train),
        ("val", frac_val),
        ("test", frac_test),
    ]
    split_counts = {}
    for i, (split, frac) in enumerate(split_specs):
        set_seed(subset_seed + i)
        if frac < 1.0:
            np.random.seed(subset_seed + i)
        subset = ds.get_subset(split, frac=frac, transform=transform)
        out_split = _split_name_for_file(split)
        print(f"[embed_cache] embedding split={split} n={len(subset)} frac={frac}")
        stats = embed_subset_to_cache(
            subset=subset,
            backbone=backbone,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            out_dir=feat_dir,
            split_name=out_split,
            cache_dtype=cache_dtype,
            embed_dim=int(embed_dim),
            spurious_index=int(spurious_idx),
            store_metadata=store_metadata,
            use_amp=use_amp,
        )
        split_counts[out_split] = stats["n"]

        # Record subset indices if available.
        if hasattr(subset, "indices"):
            idx_path = feat_dir / f"indices_{out_split}.npy"
            np.save(idx_path, np.asarray(subset.indices, dtype=np.int64))

    # Build skewed val split.
    skew_cfg = cfg["validation"]["skewed_val"]
    if not bool(skew_cfg.get("enabled", True)):
        raise ValueError("validation.skewed_val.enabled=false is not supported in this pipeline.")

    source_split = skew_cfg.get("source_split", "val")
    source_split_name = _split_name_for_file(source_split)
    X_src, y_src, a_src, g_src = _load_split_arrays(feat_dir, source_split)

    worst_group_id = skew_cfg.get("worst_group_id")
    if worst_group_id is None:
        worst_source = skew_cfg.get("worst_group_source", "train")
        _, _, _, g_worst = _load_split_arrays(feat_dir, worst_source)
        counts = _group_counts(g_worst)
        worst_group_id = min(counts, key=lambda k: counts[k])

    val_size = int(skew_cfg["size"])
    seed = int(skew_cfg.get("seed", 0))
    val_idx, info = make_skewed_val_indices(
        group_ids=g_src,
        size=val_size,
        worst_group_id=int(worst_group_id),
        worst_group_frac=float(skew_cfg["worst_group_frac"]),
        seed=seed,
    )

    # Save val_skew indices (relative to source split).
    np.save(feat_dir / "val_skew_idx.npy", val_idx.astype(np.int64))

    X_val_skew = np.asarray(X_src[val_idx], dtype=cache_dtype)
    y_val_true = np.asarray(y_src[val_idx], dtype=np.int64)
    a_val = np.asarray(a_src[val_idx], dtype=np.int64)

    # Optional label noise on val_skew (selection only).
    noise = float(skew_cfg.get("label_noise", 0.0))
    if noise > 0:
        rng = np.random.default_rng(seed + 777)
        flip = rng.random(y_val_true.shape[0]) < noise
        y_val = y_val_true.copy()
        y_val[flip] = 1 - y_val[flip]
    else:
        y_val = y_val_true.copy()

    a_max = int(a_val.max()) if a_val.size else 0
    g_val = y_val * (a_max + 1) + a_val

    np.save(feat_dir / "X_val_skew.npy", X_val_skew)
    np.save(feat_dir / "y_val_skew.npy", y_val)
    np.save(feat_dir / "y_val_skew_true.npy", y_val_true)
    np.save(feat_dir / "a_val_skew.npy", a_val)
    np.save(feat_dir / "g_val_skew.npy", g_val)

    # Train sub-selection: if val_skew comes from train, exclude it.
    X_train, y_train, a_train, g_train = _load_split_arrays(feat_dir, "train")
    if _split_name_for_file(source_split) == "train":
        train_idx = np.setdiff1d(np.arange(len(y_train)), val_idx)
    else:
        train_idx = np.arange(len(y_train))
    np.save(feat_dir / "train_sub_idx.npy", train_idx.astype(np.int64))

    if store_train_sub:
        X_train_sub = np.asarray(X_train[train_idx], dtype=cache_dtype)
        y_train_sub = np.asarray(y_train[train_idx], dtype=np.int64)
        a_train_sub = np.asarray(a_train[train_idx], dtype=np.int64)
        g_train_sub = np.asarray(g_train[train_idx], dtype=np.int64)
        np.save(feat_dir / "X_train_sub.npy", X_train_sub)
        np.save(feat_dir / "y_train_sub.npy", y_train_sub)
        np.save(feat_dir / "a_train_sub.npy", a_train_sub)
        np.save(feat_dir / "g_train_sub.npy", g_train_sub)

    splits = {
        "val_skew_source_split": source_split_name,
        "val_skew_idx_len": int(val_idx.size),
        "val_skew_info": info,
        "worst_group_id": int(worst_group_id),
        "label_noise": float(noise),
        "train_sub_len": int(train_idx.size),
        "group_counts_train": _group_counts(g_train),
        "group_counts_val": _group_counts(g_src),
        "group_counts_val_skew_true": _group_counts(y_val_true * (a_max + 1) + a_val),
    }
    splits_path.write_text(json.dumps(splits, indent=2))

    info = {
        "dataset": dataset_name,
        "wilds_dataset": wilds_name,
        "data_dir": str(data_dir),
        "backbone": backbone_name,
        "embed_dim": int(embed_dim),
        "cache_dtype": str(np.dtype(cache_dtype).name),
        "metadata_fields": meta_fields,
        "spurious_index": int(spurious_idx),
        "counts": split_counts,
    }
    save_json(info_path, info)
    print(f"[embed_cache] wrote: {info_path}")
    print(f"[embed_cache] wrote: {splits_path}")


if __name__ == "__main__":
    main()
