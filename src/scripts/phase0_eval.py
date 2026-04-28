import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import load_config
from ..metrics.proxy_eval import between_total_ratio, snr_between_total_multi
from ..utils.io import ensure_dir
from ..utils.stats import cvar_top_fraction


@dataclass
class RunInfo:
    dataset: str
    regime: str
    seed: int
    tag: str
    run_dir: Path


def discover_runs(runs_root: Path, dataset: str, regimes: Iterable[str]) -> List[RunInfo]:
    out: List[RunInfo] = []
    for regime in regimes:
        regime_dir = runs_root / dataset / regime
        if not regime_dir.exists():
            continue
        for seed_dir in sorted(regime_dir.glob("seed*")):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except Exception:
                continue
            for tag_dir in sorted(seed_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                if not (tag_dir / "config.json").exists():
                    continue
                out.append(RunInfo(dataset=dataset, regime=regime, seed=seed, tag=tag_dir.name, run_dir=tag_dir))
    return out


def build_head(d_in: int, hidden_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(d_in, 1)
    return nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(hidden_dim, 1),
    )


@torch.no_grad()
def compute_train_loss_by_epoch(
    run_dir: Path,
    feat_dir: Path,
    d_in: int,
    hidden_dim: int,
    dropout: float,
    device: str,
    batch_size: int,
):
    out_path = run_dir / "train_loss_by_epoch.npy"
    if out_path.exists():
        try:
            return np.load(out_path, mmap_mode="r")
        except Exception:
            # Corrupt or incompatible file; rebuild.
            out_path.unlink(missing_ok=True)

    X_train = np.load(feat_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(feat_dir / "y_train.npy")
    n = int(y_train.shape[0])
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    E = len(ckpts)
    losses_all = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(E, n))

    model = build_head(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout).to(device)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for e, ckpt_path in enumerate(ckpts):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        offset = 0
        i = 0
        curr_bs = int(batch_size)
        while i < n:
            bs = min(curr_bs, n - i)
            while True:
                try:
                    xb_np = np.asarray(X_train[i : i + bs], dtype=np.float32)
                    yb_np = np.asarray(y_train[i : i + bs], dtype=np.int64)
                    break
                except Exception as ex:
                    # Rare memory pressure on long eval sweeps: shrink batch and retry.
                    ex_name = type(ex).__name__.lower()
                    if ("memory" not in ex_name and "arraymemory" not in ex_name) or bs <= 64:
                        raise
                    bs = max(64, bs // 2)
                    curr_bs = min(curr_bs, bs)
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(yb_np).to(device)
            logits = model(xb).squeeze(1)
            losses = bce(logits, yb.float()).detach().cpu().numpy().astype(np.float32)
            losses_all[e, offset : offset + losses.shape[0]] = losses
            offset += losses.shape[0]
            i += bs

    # Flush memmap to disk.
    losses_all.flush()
    return losses_all


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
        p = base / f"{prefix}_m{m:02d}_K*.npy"
        # We stored exact filenames; use glob.
        matches = list(base.glob(f"{prefix}_m{m:02d}_K*.npy"))
        if not matches:
            raise FileNotFoundError(f"Missing partition for {family} bank{bank} split={split} m={m}")
        parts.append(np.load(matches[0]))
    return parts


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


def _cell_stats(cell_ids: np.ndarray, min_cell: int):
    counts = np.bincount(cell_ids.astype(np.int64))
    counts = counts[counts > 0]
    if counts.size == 0:
        return {
            "median_cell": 0.0,
            "p5_cell": 0.0,
            "eff_cells": 0.0,
        }
    med = float(np.median(counts))
    p5 = float(np.percentile(counts, 5))
    eff = float(np.sum(counts >= int(min_cell)))
    return {
        "median_cell": med,
        "p5_cell": p5,
        "eff_cells": eff,
    }


def _proxy_metrics(
    losses: np.ndarray,
    correct: np.ndarray,
    parts: List[np.ndarray],
    num_cells: int,
    min_cell: int,
    cvar_q: float,
):
    worst_accs = []
    worst_losses = []
    cvar_losses = []
    frac_small = []
    cell_stats = []

    for cells in parts:
        cells = cells.astype(np.int64)
        counts = np.bincount(cells, minlength=int(num_cells))
        small_mask = counts < int(min_cell)
        frac_small.append(float(np.mean(small_mask)))

        means_loss = np.full((int(num_cells),), np.nan, dtype=np.float64)
        means_acc = np.full((int(num_cells),), np.nan, dtype=np.float64)
        for k in range(int(num_cells)):
            idx = np.where(cells == k)[0]
            if idx.size == 0:
                continue
            means_loss[k] = float(np.mean(losses[idx]))
            means_acc[k] = float(np.mean(correct[idx]))

        valid = counts >= int(min_cell)
        if np.any(valid):
            worst_losses.append(float(np.nanmax(means_loss[valid])))
            worst_accs.append(float(np.nanmin(means_acc[valid])))
            # CVaR over cells (top q by loss)
            valid_losses = means_loss[valid]
            cvar_losses.append(cvar_top_fraction(valid_losses, float(cvar_q)))
        else:
            worst_losses.append(np.nan)
            worst_accs.append(np.nan)
            cvar_losses.append(np.nan)

        cell_stats.append(_cell_stats(cells, min_cell=min_cell))

    out = {
        "worst_acc": float(np.nanmean(worst_accs)) if worst_accs else np.nan,
        "worst_loss": float(np.nanmean(worst_losses)) if worst_losses else np.nan,
        "cvar_loss": float(np.nanmean(cvar_losses)) if cvar_losses else np.nan,
        "frac_small": float(np.nanmean(frac_small)) if frac_small else np.nan,
        "median_cell": float(np.nanmedian([c["median_cell"] for c in cell_stats])) if cell_stats else np.nan,
        "p5_cell": float(np.nanmedian([c["p5_cell"] for c in cell_stats])) if cell_stats else np.nan,
        "eff_cells": float(np.nanmedian([c["eff_cells"] for c in cell_stats])) if cell_stats else np.nan,
    }
    return out


def _between_avg(values: np.ndarray, parts: List[np.ndarray], num_cells: int) -> float:
    vals = []
    for cells in parts:
        vals.append(between_total_ratio(values, cells, num_cells))
    return float(np.mean(vals)) if vals else 0.0


def _apply_objective_transform(losses: np.ndarray, clip_loss: float, clip_alpha: float) -> np.ndarray:
    losses = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(clip_loss) or clip_loss <= 0:
        return losses
    if np.isfinite(clip_alpha) and clip_alpha > 0.0:
        return np.where(losses <= clip_loss, losses, clip_loss + clip_alpha * (losses - clip_loss))
    # Backward-compatible hard clipping path.
    return np.minimum(losses, clip_loss)


def _load_epoch_clip_alpha(run_dir: Path) -> Dict[int, float]:
    """Load per-epoch active clip alpha from metrics logs when available.

    Static regimes do not log `clip_alpha_active`; we simply return an empty map
    and callers should fall back to the run-level `clip_alpha`.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    out: Dict[int, float] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = int(rec.get("epoch", -1))
        if ep <= 0:
            continue
        a = rec.get("clip_alpha_active", None)
        if a is None:
            continue
        try:
            a_f = float(a)
        except (TypeError, ValueError):
            continue
        if np.isfinite(a_f):
            out[ep] = a_f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regimes", default="erm,rcgdro")
    ap.add_argument("--min_cell", type=int, default=20)
    ap.add_argument("--cvar_q", type=float, default=0.1)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--exclude_tag_filter", default="")
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
        raise FileNotFoundError(f"Missing eval banks at {eval_root}. Run build_eval_banks first.")

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    runs_root = Path(cfg["project"]["runs_dir"])
    runs = discover_runs(runs_root, dataset_name, regimes)
    if not runs:
        raise FileNotFoundError(f"No runs found under {runs_root / dataset_name} for regimes={regimes}")

    tag_filters = [t.strip() for t in args.tag_filter.split(",") if t.strip()]
    exclude_tag_filter = args.exclude_tag_filter.strip()
    if tag_filters:
        runs = [r for r in runs if any(t in r.tag for t in tag_filters)]
    if exclude_tag_filter:
        runs = [r for r in runs if exclude_tag_filter not in r.tag]

    include = [s.strip() for s in args.families.split(",") if s.strip()]
    families = _load_families(eval_root, include if include else None)

    eval_cfg = cfg.get("partitions", {}).get("eval_banks", {})
    banks = eval_cfg.get("banks", ["A", "B"])

    # Load labels for val.
    y_val = np.load(feat_dir / "y_val_skew.npy")
    a_val = np.load(feat_dir / "a_val_skew.npy")
    g_val = np.load(feat_dir / "g_val_skew.npy")

    out_dir = artifacts_dir / "metrics"
    ensure_dir(out_dir)
    suffix = str(args.out_suffix).strip()
    suffix = f"_{suffix}" if suffix else ""
    out_val = out_dir / f"{dataset_name}_{backbone}_phase0_val_metrics{suffix}.csv"
    out_flat = out_dir / f"{dataset_name}_{backbone}_phase0_flattening{suffix}.csv"

    if out_val.exists() and out_flat.exists() and not int(args.overwrite):
        print(f"[phase0] metrics already exist; use --overwrite 1 to regenerate.")
        return

    rows_val: List[Dict] = []
    rows_flat: List[Dict] = []

    device = cfg["compute"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    for run in runs:
        run_cfg = json.loads((run.run_dir / "config.json").read_text())
        d_in = run_cfg.get("d_in")
        if d_in is not None:
            d_in = int(d_in)
        hidden_dim = int(run_cfg.get("training", {}).get("hidden_dim", 0))
        dropout = float(run_cfg.get("training", {}).get("dropout", 0.0))
        clip_loss = float(run_cfg.get("clip_loss", 0.0) or 0.0)
        clip_alpha = float(run_cfg.get("clip_alpha", 0.0) or 0.0)

        val_loss = np.load(run.run_dir / "val_loss_by_epoch.npy")
        val_correct = np.load(run.run_dir / "val_correct_by_epoch.npy")
        E, N = val_loss.shape
        epoch_clip_alpha = _load_epoch_clip_alpha(run.run_dir)

        # Compute train losses for flattening (skip if model head is unknown, e.g., finetune runs).
        train_loss = None
        if d_in is not None:
            train_loss = compute_train_loss_by_epoch(
                run_dir=run.run_dir,
                feat_dir=feat_dir,
                d_in=d_in,
                hidden_dim=hidden_dim,
                dropout=dropout,
                device=device,
                batch_size=int(cfg["training"].get("eval_batch_size", 2048)),
            )

        for family, prefix, K, M in families:
            for bank in banks:
                parts_val = _load_partitions(eval_root, family, bank, "val_skew", prefix, M)
                parts_train = None
                if train_loss is not None:
                    parts_train = _load_partitions(eval_root, family, bank, "train", prefix, M)

                # Precompute cell stats (static across epochs)
                stats = _cell_stats(parts_val[0].astype(np.int64), min_cell=int(args.min_cell))

                for e in range(E):
                    losses_e = val_loss[e].astype(np.float64, copy=False)
                    correct_e = val_correct[e].astype(np.float64, copy=False)
                    ep = int(e + 1)
                    losses_clip = None
                    if clip_loss > 0:
                        clip_alpha_ep = float(epoch_clip_alpha.get(ep, clip_alpha))
                        losses_clip = _apply_objective_transform(
                            losses=losses_e,
                            clip_loss=clip_loss,
                            clip_alpha=clip_alpha_ep,
                        )

                    proxy = _proxy_metrics(
                        losses=losses_e,
                        correct=correct_e,
                        parts=parts_val,
                        num_cells=K,
                        min_cell=int(args.min_cell),
                        cvar_q=float(args.cvar_q),
                    )
                    proxy_clip = None
                    if losses_clip is not None:
                        proxy_clip = _proxy_metrics(
                            losses=losses_clip,
                            correct=correct_e,
                            parts=parts_val,
                            num_cells=K,
                            min_cell=int(args.min_cell),
                            cvar_q=float(args.cvar_q),
                        )
                    snr = snr_between_total_multi(
                        correct=correct_e,
                        partitions=parts_val,
                        num_cells=K,
                        null_trials=int(cfg.get("analysis", {}).get("snr_null_trials", 25)),
                        seed=int(run.seed) * 1000 + e,
                    )
                    between_loss = _between_avg(losses_e, parts_val, K)
                    between_correct = _between_avg(correct_e, parts_val, K)

                    rows_val.append(
                        {
                            "dataset": run.dataset,
                            "regime": run.regime,
                            "seed": run.seed,
                            "tag": run.tag,
                            "epoch": ep,
                            "family": family,
                            "bank": bank,
                            "val_overall_acc": float(correct_e.mean()),
                            "val_overall_loss": float(losses_e.mean()),
                            "proxy_worst_acc_min": proxy["worst_acc"],
                            "proxy_worst_loss_min": proxy["worst_loss"],
                            "proxy_cvar_loss": proxy["cvar_loss"],
                            "proxy_worst_loss_clip_min": proxy_clip["worst_loss"] if proxy_clip else np.nan,
                            "proxy_cvar_loss_clip": proxy_clip["cvar_loss"] if proxy_clip else np.nan,
                            "proxy_between_loss": between_loss,
                            "proxy_between_correct": between_correct,
                            "proxy_snr_correct": snr,
                            "frac_small_cells": proxy["frac_small"],
                            "median_cell": proxy["median_cell"],
                            "p5_cell": proxy["p5_cell"],
                            "eff_cells": proxy["eff_cells"],
                            "clip_alpha_active": float(epoch_clip_alpha.get(ep, clip_alpha)),
                        }
                    )

                # Flattening on train/val (loss-based)
                for e in range(E):
                    val_loss_e = np.asarray(val_loss[e], dtype=np.float64)
                    rows_flat.append(
                        {
                            "dataset": run.dataset,
                            "regime": run.regime,
                            "seed": run.seed,
                            "tag": run.tag,
                            "epoch": int(e + 1),
                            "family": family,
                            "bank": bank,
                            "split": "val",
                            "between_loss_ratio": _between_avg(val_loss_e, parts_val, K),
                        }
                    )
                    if train_loss is not None and parts_train is not None:
                        tr_loss_e = np.asarray(train_loss[e], dtype=np.float64)
                        rows_flat.append(
                            {
                                "dataset": run.dataset,
                                "regime": run.regime,
                                "seed": run.seed,
                                "tag": run.tag,
                                "epoch": int(e + 1),
                                "family": family,
                                "bank": bank,
                                "split": "train",
                                "between_loss_ratio": _between_avg(tr_loss_e, parts_train, K),
                            }
                        )

    pd.DataFrame(rows_val).to_csv(out_val, index=False)
    pd.DataFrame(rows_flat).to_csv(out_flat, index=False)
    print(f"[phase0] wrote {out_val}")
    print(f"[phase0] wrote {out_flat}")


if __name__ == "__main__":
    main()
