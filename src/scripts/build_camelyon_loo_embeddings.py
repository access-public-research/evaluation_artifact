import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import load_config
from ..utils.io import ensure_dir


def _load_split(feat_dir: Path, split: str) -> Dict[str, np.ndarray]:
    X = np.load(feat_dir / f"X_{split}.npy", mmap_mode="r")
    y = np.load(feat_dir / f"y_{split}.npy")
    a = np.load(feat_dir / f"a_{split}.npy")
    g = np.load(feat_dir / f"g_{split}.npy")
    meta = np.load(feat_dir / f"meta_{split}.npy")
    idx_path = feat_dir / f"indices_{split}.npy"
    if idx_path.exists():
        orig_idx = np.load(idx_path)
    else:
        orig_idx = np.arange(y.shape[0], dtype=np.int64)
    return {
        "X": X,
        "y": y.astype(np.int64, copy=False),
        "a": a.astype(np.int64, copy=False),
        "g": g.astype(np.int64, copy=False),
        "meta": meta.astype(np.int64, copy=False),
        "orig_idx": orig_idx.astype(np.int64, copy=False),
    }


def _save_split(out_dir: Path, name: str, X: np.ndarray, y: np.ndarray, a: np.ndarray, meta: np.ndarray, orig_idx: np.ndarray) -> None:
    np.save(out_dir / f"X_{name}.npy", X)
    np.save(out_dir / f"y_{name}.npy", y.astype(np.int64))
    np.save(out_dir / f"a_{name}.npy", a.astype(np.int64))
    a_max = int(a.max()) if a.size else 0
    g = y.astype(np.int64) * (a_max + 1) + a.astype(np.int64)
    np.save(out_dir / f"g_{name}.npy", g.astype(np.int64))
    np.save(out_dir / f"meta_{name}.npy", meta.astype(np.int64))
    np.save(out_dir / f"indices_{name}.npy", orig_idx.astype(np.int64))


def _concat_splits(splits: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    X = np.concatenate([np.asarray(s["X"]) for s in splits], axis=0)
    y = np.concatenate([s["y"] for s in splits], axis=0)
    a = np.concatenate([s["a"] for s in splits], axis=0)
    meta = np.concatenate([s["meta"] for s in splits], axis=0)
    orig_idx = np.concatenate([s["orig_idx"] for s in splits], axis=0)
    return {"X": X, "y": y, "a": a, "meta": meta, "orig_idx": orig_idx}


def _sample_idx(rng: np.random.Generator, idx: np.ndarray, n: int) -> np.ndarray:
    if n >= idx.shape[0]:
        return idx.copy()
    return rng.choice(idx, size=n, replace=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source_dataset", default="camelyon17")
    ap.add_argument("--holdouts", default="0,1,2,3,4")
    ap.add_argument("--out_prefix", default="camelyon17_loo_h")
    ap.add_argument("--val_skew_size", type=int, default=2000)
    ap.add_argument("--validation_size", type=int, default=34904)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.source_dataset}.yaml")
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    backbone = cfg["embeddings"]["backbone"]
    source_name = cfg["dataset"]["name"]
    src_dir = artifacts_dir / "embeds" / f"{source_name}_{backbone}"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source embeddings: {src_dir}")

    info_path = src_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing source info.json: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    train = _load_split(src_dir, "train")
    val = _load_split(src_dir, "validation")
    test = _load_split(src_dir, "test")
    pool = _concat_splits([train, val, test])
    hospitals = pool["meta"][:, 0].astype(np.int64)

    holdouts = [int(x.strip()) for x in str(args.holdouts).split(",") if x.strip()]
    for holdout in holdouts:
        fold_name = f"{args.out_prefix}{holdout}"
        out_dir = artifacts_dir / "embeds" / f"{fold_name}_{backbone}"
        if out_dir.exists() and not int(args.overwrite):
            if (out_dir / "info.json").exists():
                print(f"[loo-cache] exists, skipping: {fold_name}")
                continue
        ensure_dir(out_dir)

        rng = np.random.default_rng(int(args.seed) + int(holdout) * 1000)
        idx_holdout = np.where(hospitals == int(holdout))[0]
        idx_in = np.where(hospitals != int(holdout))[0]
        if idx_holdout.size == 0:
            raise ValueError(f"No samples found for holdout hospital={holdout}")

        # Train uses all non-holdout examples.
        idx_train = idx_in
        # Validation is a balanced in-domain subset for eval-bank construction.
        n_val = min(int(args.validation_size), idx_in.size)
        idx_validation = _sample_idx(rng, idx_in, n=n_val)
        # OOD test is full holdout hospital.
        idx_test = idx_holdout
        # Selection split is a fixed-size sample of OOD.
        n_skew = min(int(args.val_skew_size), idx_holdout.size)
        idx_val_skew = _sample_idx(rng, idx_holdout, n=n_skew)

        for name, idx in [
            ("train", idx_train),
            ("validation", idx_validation),
            ("test", idx_test),
            ("val_skew", idx_val_skew),
        ]:
            X = np.asarray(pool["X"][idx], dtype=np.float16)
            y = pool["y"][idx]
            a = pool["a"][idx]
            meta = pool["meta"][idx]
            orig_idx = pool["orig_idx"][idx]
            _save_split(out_dir, name, X, y, a, meta, orig_idx)
        np.save(out_dir / "y_val_skew_true.npy", pool["y"][idx_val_skew].astype(np.int64))

        # Full train is used; no extra exclusion.
        np.save(out_dir / "train_sub_idx.npy", np.arange(idx_train.size, dtype=np.int64))
        # val_skew indices are relative to test split here.
        test_orig = pool["orig_idx"][idx_test]
        skew_orig = pool["orig_idx"][idx_val_skew]
        pos_map = {int(v): i for i, v in enumerate(test_orig.tolist())}
        val_skew_idx = np.array([pos_map[int(v)] for v in skew_orig.tolist()], dtype=np.int64)
        np.save(out_dir / "val_skew_idx.npy", val_skew_idx)

        splits = {
            "mode": "camelyon_leave_one_hospital_out",
            "holdout_hospital": int(holdout),
            "source_pool": ["train", "validation", "test"],
            "counts": {
                "train": int(idx_train.size),
                "validation": int(idx_validation.size),
                "test": int(idx_test.size),
                "val_skew": int(idx_val_skew.size),
            },
            "val_skew_source_split": "test",
        }
        (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

        info_fold = dict(info)
        info_fold["dataset"] = fold_name
        info_fold["counts"] = {
            "train": int(idx_train.size),
            "validation": int(idx_validation.size),
            "test": int(idx_test.size),
        }
        (out_dir / "info.json").write_text(json.dumps(info_fold, indent=2), encoding="utf-8")
        print(
            f"[loo-cache] wrote {fold_name}: train={idx_train.size}, validation={idx_validation.size}, "
            f"val_skew={idx_val_skew.size}, test={idx_test.size}"
        )


if __name__ == "__main__":
    main()
