import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils.io import ensure_dir


def _infer_prefix(family: str, meta: dict) -> str:
    prefix = str(meta.get("prefix", "")).strip()
    if prefix:
        return prefix
    name = family.lower()
    if "hash" in name:
        return "hash"
    if "proj" in name:
        return "proj"
    if "conf" in name:
        return "conf"
    if "diff" in name or "difficulty" in name:
        return "diff"
    raise ValueError(f"Could not infer prefix for family {family}. Add prefix to meta.json.")


def _load_families(eval_root: Path, include: List[str] | None) -> List[Tuple[str, str, int, int]]:
    fams: List[Tuple[str, str, int, int]] = []
    for fam_dir in sorted(eval_root.iterdir()):
        if not fam_dir.is_dir():
            continue
        name = fam_dir.name
        if include and name not in include:
            continue
        meta_path = fam_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        prefix = _infer_prefix(name, meta)
        K = int(meta.get("num_cells"))
        M = int(meta.get("num_partitions", 1))
        fams.append((name, prefix, K, M))
    if include and not fams:
        raise FileNotFoundError(f"No eval families found for include={include} under {eval_root}.")
    return fams


def _load_partitions(eval_root: Path, family: str, bank: str, split: str, prefix: str, num_parts: int):
    split = split.lower()
    if split == "val_skew":
        split = "val_skew"
    elif split in {"val", "validation"}:
        split = "validation"
    elif split == "train":
        split = "train"
    else:
        raise ValueError(f"Unknown split: {split}")
    parts = []
    base = eval_root / family / f"bank{bank}" / split
    for m in range(int(num_parts)):
        matches = list(base.glob(f"{prefix}_m{m:02d}_K*.npy"))
        if not matches:
            raise FileNotFoundError(f"Missing partition for {family} bank{bank} split={split} m={m}")
        parts.append(np.load(matches[0]))
    return parts


def _mutual_info(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    n = int(x.shape[0])
    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((x_vals.size, y_vals.size), dtype=np.int64)
    np.add.at(joint, (x_inv, y_inv), 1)
    pxy = joint / float(n)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx <= 0 or hy <= 0:
        nmi = 0.0
    else:
        nmi = float(mi / np.sqrt(hx * hy))
    return mi, nmi


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_pos = float(ranks[labels == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _score_from_bins(bin_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    bin_ids = bin_ids.astype(np.int64)
    labels = labels.astype(np.int64)
    K = int(bin_ids.max()) + 1
    scores = np.zeros((K,), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        mask = bin_ids == k
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            scores[k] = float(np.mean(labels[mask]))
    return scores[bin_ids]


def _enrichment_worst_bins(
    bin_ids: np.ndarray,
    worst_mask: np.ndarray,
    diff_bins: np.ndarray,
    q: float,
) -> float:
    bin_ids = bin_ids.astype(np.int64)
    diff_bins = diff_bins.astype(np.int64)
    K = int(bin_ids.max()) + 1
    # Rank bins by mean difficulty-bin index.
    mean_diff = np.zeros((K,), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        mask = bin_ids == k
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            mean_diff[k] = float(np.mean(diff_bins[mask]))
        else:
            mean_diff[k] = -1.0
    valid = counts > 0
    valid_bins = np.where(valid)[0]
    if valid_bins.size == 0:
        return float("nan")
    k_sel = max(1, int(np.ceil(float(q) * valid_bins.size)))
    worst_bins = valid_bins[np.argsort(mean_diff[valid_bins])[-k_sel:]]
    in_worst = np.isin(bin_ids, worst_bins)
    p_worst = float(np.mean(in_worst))
    if p_worst <= 0:
        return float("nan")
    p_wg_in = float(np.mean(worst_mask[in_worst]))
    p_wg = float(np.mean(worst_mask))
    if p_wg <= 0:
        return float("nan")
    return float(p_wg_in / p_wg)


def _top_diff_mask(diff_bins: np.ndarray, q: float) -> np.ndarray:
    diff_bins = diff_bins.astype(np.int64)
    Kd = int(diff_bins.max()) + 1
    k = max(1, int(np.ceil(float(q) * Kd)))
    top_bins = np.arange(Kd - k, Kd)
    return np.isin(diff_bins, top_bins)


def _select_epoch(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return -1
    best_epoch = -1
    best_acc = -1.0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        acc = float(rec.get("val_acc", -1.0))
        epoch = int(rec.get("epoch", 0))
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
    return best_epoch


def _load_teacher_logits(run_dir: Path) -> np.ndarray:
    logits_path = run_dir / "val_logits_by_epoch.npy"
    if not logits_path.exists():
        raise FileNotFoundError(f"Missing val_logits_by_epoch.npy in {run_dir}")
    logits = np.load(logits_path)
    epoch = _select_epoch(run_dir / "metrics.jsonl")
    if epoch <= 0:
        epoch = logits.shape[0]
    return logits[int(epoch - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--families", default="")
    ap.add_argument("--difficulty_family", default="teacher_difficulty")
    ap.add_argument("--teacher_run", default="")
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    backbone = cfg["embeddings"]["backbone"]

    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"
    if not feat_dir.exists():
        raise FileNotFoundError(f"Missing embeddings at {feat_dir}.")

    eval_version = cfg.get("partitions", {}).get("eval_version")
    if eval_version:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}" / str(eval_version)
    else:
        eval_root = artifacts_dir / "partitions_eval" / f"{dataset_name}_{backbone}"
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval banks at {eval_root}.")

    include = [s.strip() for s in args.families.split(",") if s.strip()]
    families = _load_families(eval_root, include if include else None)

    diff_family = args.difficulty_family
    diff_meta_path = eval_root / diff_family / "meta.json"
    if not diff_meta_path.exists():
        raise FileNotFoundError(f"Missing difficulty family {diff_family} under {eval_root}")
    diff_meta = json.loads(diff_meta_path.read_text())
    diff_prefix = _infer_prefix(diff_family, diff_meta)
    diff_K = int(diff_meta.get("num_cells"))
    diff_M = int(diff_meta.get("num_partitions", 1))

    eval_cfg = cfg.get("partitions", {}).get("eval_banks", {})
    banks = eval_cfg.get("banks", ["A", "B"])

    # Load labels/groups.
    y_val = np.load(feat_dir / "y_val_skew.npy")
    g_val = np.load(feat_dir / "g_val_skew.npy")

    # Determine worst group using a baseline teacher run.
    teacher_run = args.teacher_run
    if not teacher_run:
        # default: first ERM run in seed0
        runs_root = Path(cfg["project"]["runs_dir"]) / dataset_name / "erm" / "seed0"
        if not runs_root.exists():
            raise FileNotFoundError("No teacher_run provided and default ERM run not found.")
        candidates = sorted([p for p in runs_root.iterdir() if p.is_dir() and (p / "val_logits_by_epoch.npy").exists()])
        if not candidates:
            raise FileNotFoundError("No teacher_run provided and no ERM logits found.")
        teacher_run = str(candidates[0])
    teacher_run = Path(teacher_run)
    logits_teacher = _load_teacher_logits(teacher_run)
    preds_teacher = (logits_teacher >= 0).astype(np.int64)
    worst_gid = None
    worst_acc = 1.0
    for gid in np.unique(g_val):
        mask = g_val == gid
        if mask.sum() == 0:
            continue
        acc = float(np.mean(preds_teacher[mask] == y_val[mask]))
        if acc < worst_acc:
            worst_acc = acc
            worst_gid = int(gid)
    if worst_gid is None:
        raise RuntimeError("Failed to determine worst group.")
    worst_mask = (g_val == worst_gid).astype(np.int64)

    rows: List[Dict] = []
    for bank in banks:
        diff_parts = _load_partitions(eval_root, diff_family, bank, "val_skew", diff_prefix, diff_M)
        if not diff_parts:
            raise FileNotFoundError(f"Missing difficulty partitions for bank{bank}")
        diff_bins = diff_parts[0].astype(np.int64)
        top_diff = _top_diff_mask(diff_bins, float(args.q)).astype(np.int64)

        for family, prefix, K, M in families:
            parts = _load_partitions(eval_root, family, bank, "val_skew", prefix, M)
            for m, bins in enumerate(parts):
                bins = bins.astype(np.int64)
                mi_g, nmi_g = _mutual_info(bins, g_val)
                auc_g = _auc(_score_from_bins(bins, worst_mask), worst_mask)
                enrich = _enrichment_worst_bins(bins, worst_mask, diff_bins, float(args.q))

                mi_d, nmi_d = _mutual_info(bins, diff_bins)
                auc_d = _auc(_score_from_bins(bins, top_diff), top_diff)

                rows.append(
                    {
                        "dataset": dataset_name,
                        "family": family,
                        "bank": bank,
                        "partition": int(m),
                        "n_examples": int(bins.shape[0]),
                        "n_bins": int(int(bins.max()) + 1),
                        "worst_gid": int(worst_gid),
                        "mi_group": mi_g,
                        "nmi_group": nmi_g,
                        "auc_worst_group": auc_g,
                        "enrich_worst_group_in_worst_bins": enrich,
                        "mi_difficulty": mi_d,
                        "nmi_difficulty": nmi_d,
                        "auc_top_difficulty": auc_d,
                    }
                )

    out_dir = artifacts_dir / "metrics"
    ensure_dir(out_dir)
    suffix = str(args.out_suffix).strip()
    suffix = f"_{suffix}" if suffix else ""
    out_path = out_dir / f"{dataset_name}_{backbone}_alignment_audit{suffix}.csv"
    if out_path.exists() and not int(args.overwrite):
        print(f"[alignment_audit] {out_path} exists; use --overwrite 1 to regenerate.")
        return
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[alignment_audit] wrote {out_path}")


if __name__ == "__main__":
    main()
