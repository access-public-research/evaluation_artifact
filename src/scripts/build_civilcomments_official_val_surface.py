import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..config import load_config
from ..utils.io import ensure_dir


CIVILCOMMENTS_IDENTITY_FIELDS = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]


def _sample_tail_surface(
    tail_mask: np.ndarray,
    *,
    size: int,
    tail_frac: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(tail_mask.shape[0]), dtype=np.int64)
    tail_idx = idx[np.asarray(tail_mask, dtype=bool)]
    other_idx = idx[~np.asarray(tail_mask, dtype=bool)]

    target_tail = int(round(int(size) * float(tail_frac)))
    tail_n = min(int(tail_idx.size), int(target_tail))
    other_n = min(int(other_idx.size), int(size) - int(tail_n))
    if tail_n + other_n <= 0:
        raise ValueError("Resolved surface size is zero.")

    tail_sel = rng.choice(tail_idx, size=tail_n, replace=False) if tail_n > 0 else np.array([], dtype=np.int64)
    other_sel = rng.choice(other_idx, size=other_n, replace=False) if other_n > 0 else np.array([], dtype=np.int64)
    sel = np.concatenate([tail_sel, other_sel]).astype(np.int64)
    rng.shuffle(sel)

    info = {
        "requested_size": int(size),
        "actual_size": int(sel.size),
        "tail_count_full": int(tail_idx.size),
        "other_count_full": int(other_idx.size),
        "tail_frac_target": float(tail_frac),
        "tail_selected": int(tail_n),
        "other_selected": int(other_n),
    }
    return sel, info


def _official_group_counts(meta: np.ndarray, y: np.ndarray) -> dict:
    out: dict[str, int] = {}
    for ident_idx, ident in enumerate(CIVILCOMMENTS_IDENTITY_FIELDS):
        for label in (0, 1):
            mask = (meta[:, ident_idx] == 1) & (y == label)
            out[f"{ident}={label}"] = int(np.sum(mask))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--src_artifact_name", default="")
    ap.add_argument("--out_artifact_name", required=True)
    ap.add_argument("--mode", default="minor_identity_positive", choices=["minor_identity_positive", "identity_any_positive"])
    ap.add_argument("--tail_identities", default="LGBTQ,christian,muslim,other_religions,black")
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--tail_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = str(cfg["dataset"]["name"])
    if dataset_name != "civilcomments":
        raise ValueError("This builder is specific to civilcomments.")

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    backbone_name = str(cfg["embeddings"]["backbone"])
    src_name = str(args.src_artifact_name).strip() or f"{dataset_name}_{backbone_name}"
    src_dir = artifacts_dir / "embeds" / src_name
    out_dir = artifacts_dir / "embeds" / str(args.out_artifact_name)

    if out_dir.exists() and int(args.overwrite):
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    meta_val = np.load(src_dir / "meta_validation.npy")
    y_val_true = np.load(src_dir / "y_validation.npy").astype(np.int64)
    a_val_src = np.load(src_dir / "a_validation.npy").astype(np.int64)

    if str(args.mode) == "identity_any_positive":
        identity_any_idx = 8
        tail_mask = (meta_val[:, identity_any_idx] == 1) & (y_val_true == 1)
        tail_desc = "identity_any=1,label=1"
        chosen_identities = ["identity_any"]
    else:
        chosen_identities = [s.strip() for s in str(args.tail_identities).split(",") if s.strip()]
        field_to_idx = {name: i for i, name in enumerate(CIVILCOMMENTS_IDENTITY_FIELDS)}
        missing = [name for name in chosen_identities if name not in field_to_idx]
        if missing:
            raise ValueError(f"Unknown identity fields: {missing}")
        tail_mask = np.zeros_like(y_val_true, dtype=bool)
        for name in chosen_identities:
            tail_mask |= meta_val[:, field_to_idx[name]] == 1
        tail_mask &= y_val_true == 1
        tail_desc = " OR ".join(chosen_identities) + ", label=1"

    val_idx, sample_info = _sample_tail_surface(
        tail_mask,
        size=int(args.size),
        tail_frac=float(args.tail_frac),
        seed=int(args.seed),
    )

    y_val = np.asarray(y_val_true[val_idx], dtype=np.int64)
    y_val_true_sel = y_val.copy()
    # Store a binary selector-tail attribute for later diagnostics.
    a_val = np.asarray(tail_mask[val_idx], dtype=np.int64)
    g_val = a_val.copy()

    np.save(out_dir / "val_skew_idx.npy", val_idx.astype(np.int64))
    np.save(out_dir / "y_val_skew.npy", y_val)
    np.save(out_dir / "y_val_skew_true.npy", y_val_true_sel)
    np.save(out_dir / "a_val_skew.npy", a_val)
    np.save(out_dir / "g_val_skew.npy", g_val)

    splits = {
        "val_skew_source_split": "validation",
        "val_skew_idx_len": int(val_idx.size),
        "val_skew_info": sample_info,
        "tail_mode": str(args.mode),
        "tail_description": tail_desc,
        "tail_identities": chosen_identities,
        "source_artifact_name": src_name,
        "group_counts_validation_official": _official_group_counts(meta_val, y_val_true),
        "group_counts_val_skew_official": _official_group_counts(meta_val[val_idx], y_val_true_sel),
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    info = {
        "dataset": dataset_name,
        "backbone": backbone_name,
        "source_artifact_name": src_name,
        "out_artifact_name": str(args.out_artifact_name),
        "mode": str(args.mode),
        "tail_identities": chosen_identities,
        "size": int(args.size),
        "tail_frac": float(args.tail_frac),
        "seed": int(args.seed),
        "tail_count_full": int(np.sum(tail_mask)),
        "tail_description": tail_desc,
        "spurious_field_from_source": str(cfg["dataset"].get("spurious_metadata_field", "")),
        "a_validation_source_unique": np.unique(a_val_src).astype(int).tolist(),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
