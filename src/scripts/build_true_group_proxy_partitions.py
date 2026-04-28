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
    if split in {"train", "train_full"}:
        return "train"
    if split in {"test"}:
        return "test"
    raise ValueError(f"Unknown split: {split}")


def _to_contiguous_ids(group_ids: np.ndarray):
    uniq = np.unique(group_ids.astype(np.int64))
    mapping = {int(g): i for i, g in enumerate(uniq.tolist())}
    out = np.asarray([mapping[int(g)] for g in group_ids], dtype=np.int32)
    return out, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", default="train,val_skew,validation")
    parser.add_argument(
        "--group_source",
        default="g",
        choices=["g", "a"],
        help="Use cached g_* arrays (label-attribute groups) or a_* arrays (spurious/domain ids).",
    )
    parser.add_argument("--overwrite", type=int, default=0)
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
    ensure_dir(proxy_root)

    proxy_cfg = cfg["partitions"]["proxy"]
    family = str(proxy_cfg.get("family", "random_hash"))
    num_partitions = int(proxy_cfg.get("num_partitions", 1))
    num_cells = int(proxy_cfg["num_cells"])
    if family != "random_hash":
        raise ValueError(f"Expected partitions.proxy.family=random_hash, found: {family}")
    if num_partitions != 1:
        raise ValueError(f"Expected partitions.proxy.num_partitions=1, found: {num_partitions}")

    splits = [_split_name_for_file(s.strip()) for s in args.splits.split(",") if s.strip()]

    raw_by_split = {}
    all_raw_ids = []
    for split_name in splits:
        src_path = feat_dir / f"{args.group_source}_{split_name}.npy"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source group array: {src_path}")
        raw = np.load(src_path).astype(np.int64)
        raw_by_split[split_name] = raw
        all_raw_ids.append(raw)

    all_raw = np.concatenate(all_raw_ids, axis=0)
    _, mapping = _to_contiguous_ids(all_raw)

    per_split_counts = {}
    observed_groups = set()
    for split_name in splits:
        g_raw = raw_by_split[split_name]
        g = np.asarray([mapping[int(v)] for v in g_raw], dtype=np.int32)
        if int(g.max(initial=-1)) >= num_cells:
            raise ValueError(
                f"Contiguous group ids require {int(g.max()) + 1} cells, but num_cells={num_cells}."
            )
        observed_groups.update(np.unique(g).tolist())
        per_split_counts[split_name] = {
            "n": int(g.shape[0]),
            "group_counts": {int(k): int(v) for k, v in zip(*np.unique(g, return_counts=True))},
        }

        out_dir = proxy_root / split_name
        ensure_dir(out_dir)
        out_path = out_dir / f"hash_m00_K{num_cells}.npy"
        if out_path.exists() and not int(args.overwrite):
            continue
        np.save(out_path, g.astype(np.int32))

    meta = {
        "version": version,
        "root": str(part_root),
        "family": family,
        "prefix": "hash",
        "source": f"{args.group_source}_ids_from_embed_cache",
        "source_arrays": [f"{args.group_source}_{s}.npy" for s in splits],
        "mapping_raw_to_contiguous": mapping,
        "num_partitions": num_partitions,
        "num_cells": num_cells,
        "within_label": False,
        "splits": splits,
        "observed_group_ids": sorted(int(g) for g in observed_groups),
        "per_split_counts": per_split_counts,
    }
    (proxy_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build_true_group_proxy_partitions] wrote true-group proxy partitions under {part_root}")


if __name__ == "__main__":
    main()
