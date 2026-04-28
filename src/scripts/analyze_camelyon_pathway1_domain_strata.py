import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..utils.stats import ci95_mean


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
def eval_logits(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    logits_all = []
    use_amp = device.startswith("cuda")
    for i in range(0, int(X.shape[0]), int(batch_size)):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb).squeeze(1)
        else:
            logits = model(xb).squeeze(1)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits_all, axis=0)


def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Stable BCE: log(1 + exp(z)) - y * z
    z = logits.astype(np.float64, copy=False)
    yy = y.astype(np.float64, copy=False)
    return np.logaddexp(0.0, z) - yy * z


def mean_ci95(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if x.size == 1:
        return m, 0.0
    s = float(x.std(ddof=1))
    return m, ci95_mean(x)


def select_ckpt(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        best_epoch = None
        best_acc = -1.0
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            acc = float(rec.get("val_acc", -1.0))
            ep = int(rec.get("epoch", 0))
            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
        if best_epoch is not None:
            ckpt = run_dir / f"ckpt_epoch{best_epoch:03d}.pt"
            if ckpt.exists():
                return ckpt
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    return ckpts[-1]


def quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    q = np.linspace(0.0, 1.0, int(num_bins) + 1)
    edges = np.quantile(values, q)
    bins = np.digitize(values, edges[1:-1], right=True)
    return bins.astype(np.int64)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_tail_distortion_rows_cam_softclip_a10_p99_20260227.csv",
    )
    ap.add_argument(
        "--embed_dir",
        default="replication_rcg/artifacts/embeds/camelyon17_resnet50",
    )
    ap.add_argument(
        "--out_prefix",
        default="replication_rcg/artifacts/metrics/camelyon17_pathway1_domain_strata_20260302",
    )
    ap.add_argument(
        "--teacher_meta",
        default="replication_rcg/artifacts/partitions_eval/camelyon17_resnet50/teacher_difficulty/meta.json",
    )
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--include_train", type=int, default=0)
    ap.add_argument("--regime_base", default="rcgdro")
    ap.add_argument("--regime_p95", default="rcgdro_softclip_p95_a10_cam")
    ap.add_argument("--regime_p97", default="rcgdro_softclip_p97_a10_cam")
    ap.add_argument("--regime_p99", default="rcgdro_softclip_p99_a10_cam")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.rows_csv)
    regimes = [
        str(args.regime_base),
        str(args.regime_p95),
        str(args.regime_p97),
        str(args.regime_p99),
    ]
    rows = rows[rows["regime"].isin(regimes)].copy()

    embed_dir = Path(args.embed_dir)
    split_data: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ["validation", "test"]:
        X = np.load(embed_dir / f"X_{split}.npy", mmap_mode="r")
        y = np.load(embed_dir / f"y_{split}.npy")
        meta = np.load(embed_dir / f"meta_{split}.npy")
        split_data[split] = {"X": X, "y": y, "meta": meta}
    if int(args.include_train):
        X = np.load(embed_dir / "X_train.npy", mmap_mode="r")
        y = np.load(embed_dir / "y_train.npy")
        meta = np.load(embed_dir / "meta_train.npy")
        split_data["train"] = {"X": X, "y": y, "meta": meta}

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Teacher-difficulty deciles by split (bank A).
    teacher_meta = json.loads(Path(args.teacher_meta).read_text(encoding="utf-8"))
    teacher_bins: Dict[str, np.ndarray] = {}
    teacher_root = Path(args.teacher_meta).parent / "bankA"
    K_teacher = int(teacher_meta.get("num_cells", 64))
    for split, d in split_data.items():
        split_dir = teacher_root / split
        bin_path = split_dir / f"diff_m00_K{K_teacher}.npy"
        if bin_path.exists():
            bins = np.load(bin_path)
        else:
            # Compute missing split bins from teacher run A.
            teacher_run = Path(teacher_meta["teacher_runs"][0])
            cfg_t = json.loads((teacher_run / "config.json").read_text(encoding="utf-8"))
            model_t = build_head(
                d_in=int(cfg_t["d_in"]),
                hidden_dim=int(cfg_t.get("training", {}).get("hidden_dim", 0)),
                dropout=float(cfg_t.get("training", {}).get("dropout", 0.0)),
            ).to(device)
            ckpt = torch.load(select_ckpt(teacher_run), map_location=device)
            model_t.load_state_dict(ckpt["model_state"])
            logits_t = eval_logits(model_t, d["X"], batch_size=int(args.batch_size), device=device)
            losses_t = bce_from_logits(logits_t, d["y"])
            bins = quantile_bins(losses_t, K_teacher)
        teacher_bins[split] = np.minimum((bins * 10) // K_teacher, 9) + 1

    trans_rows = []
    score_rows = []
    seeds = sorted(rows["seed"].unique().tolist())
    for seed in seeds:
        by_seed = rows[rows["seed"] == seed]
        if not all((by_seed["regime"] == r).any() for r in regimes):
            continue

        # Per-seed thresholds from clipped runs.
        t95 = float(by_seed[by_seed["regime"] == str(args.regime_p95)]["clip_loss"].iloc[0])
        t97 = float(by_seed[by_seed["regime"] == str(args.regime_p97)]["clip_loss"].iloc[0])
        t99 = float(by_seed[by_seed["regime"] == str(args.regime_p99)]["clip_loss"].iloc[0])

        # Per-seed predictions/losses for each regime and split.
        pred: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in regimes}
        loss: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in regimes}
        for regime in regimes:
            rr = by_seed[by_seed["regime"] == regime].iloc[0]
            run_dir = Path(rr["run_dir"])
            ep = int(rr["epoch_selected"])
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            d_in = int(cfg["d_in"])
            hidden_dim = int(cfg.get("training", {}).get("hidden_dim", 0))
            dropout = float(cfg.get("training", {}).get("dropout", 0.0))
            model = build_head(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout).to(device)
            ckpt = torch.load(run_dir / f"ckpt_epoch{ep:03d}.pt", map_location=device)
            model.load_state_dict(ckpt["model_state"])

            for split, d in split_data.items():
                logits = eval_logits(model, d["X"], batch_size=int(args.batch_size), device=device)
                y = d["y"]
                p = (logits >= 0.0).astype(np.int64)
                pred[regime][split] = p
                loss[regime][split] = bce_from_logits(logits, y)

                score_rows.append(
                    {
                        "seed": int(seed),
                        "regime": regime,
                        "split": split,
                        "n": int(y.shape[0]),
                        "acc": float((p == y).mean()),
                        "mean_loss": float(loss[regime][split].mean()),
                    }
                )

        for split, d in split_data.items():
            y = d["y"]
            hospital = d["meta"][:, 0]
            dec_teacher = teacher_bins[split]
            # Anchor strata from proper baseline losses for this seed/split.
            l0 = loss[str(args.regime_base)][split]
            band0 = np.full(l0.shape[0], 0, dtype=np.int64)
            band0[(l0 > t95) & (l0 <= t97)] = 1
            band0[(l0 > t97) & (l0 <= t99)] = 2
            band0[l0 > t99] = 3

            c95 = pred[str(args.regime_p95)][split] == y
            c97 = pred[str(args.regime_p97)][split] == y
            c99 = pred[str(args.regime_p99)][split] == y

            transitions = {
                "recover_p95_to_p97": (~c95) & c97,
                "recover_p97_to_p99": (~c97) & c99,
                "lose_p95_to_p97": c95 & (~c97),
                "lose_p97_to_p99": c97 & (~c99),
            }
            for name, mask in transitions.items():
                n = int(mask.sum())
                if n == 0:
                    row = {
                        "seed": int(seed),
                        "split": split,
                        "transition": name,
                        "n_examples": 0,
                        "frac_le_p95": np.nan,
                        "frac_p95_p97": np.nan,
                        "frac_p97_p99": np.nan,
                        "frac_gt_p99": np.nan,
                        "mean_anchor_loss": np.nan,
                        "positive_frac": np.nan,
                        "hospital_mode": np.nan,
                    }
                else:
                    bb = band0[mask]
                    row = {
                        "seed": int(seed),
                        "split": split,
                        "transition": name,
                        "n_examples": n,
                        "frac_le_p95": float((bb == 0).mean()),
                        "frac_p95_p97": float((bb == 1).mean()),
                        "frac_p97_p99": float((bb == 2).mean()),
                        "frac_gt_p99": float((bb == 3).mean()),
                        "mean_anchor_loss": float(l0[mask].mean()),
                        "positive_frac": float(y[mask].mean()),
                        "hospital_mode": int(pd.Series(hospital[mask]).mode().iloc[0]),
                        "hospital0_frac": float((hospital[mask] == 0).mean()),
                        "hospital1_frac": float((hospital[mask] == 1).mean()),
                        "hospital2_frac": float((hospital[mask] == 2).mean()),
                        "hospital3_frac": float((hospital[mask] == 3).mean()),
                        "hospital4_frac": float((hospital[mask] == 4).mean()),
                        "teacher_top10_frac": float((dec_teacher[mask] == 10).mean()),
                        "teacher_decile9plus_frac": float((dec_teacher[mask] >= 9).mean()),
                        "teacher_mean_decile": float(dec_teacher[mask].mean()),
                    }
                trans_rows.append(row)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    trans_df = pd.DataFrame(trans_rows)
    score_df = pd.DataFrame(score_rows)
    trans_seed_path = out_prefix.with_name(out_prefix.name + "_seed_rows.csv")
    score_seed_path = out_prefix.with_name(out_prefix.name + "_scores_seed_rows.csv")
    trans_df.to_csv(trans_seed_path, index=False)
    score_df.to_csv(score_seed_path, index=False)

    # Summaries.
    sum_rows = []
    for (split, transition), sub in trans_df.groupby(["split", "transition"]):
        n = int(sub.shape[0])
        out = {"split": split, "transition": transition, "n_seeds": n}
        for col in [
            "n_examples",
            "frac_le_p95",
            "frac_p95_p97",
            "frac_p97_p99",
            "frac_gt_p99",
            "mean_anchor_loss",
            "positive_frac",
            "hospital0_frac",
            "hospital1_frac",
            "hospital2_frac",
            "hospital3_frac",
            "hospital4_frac",
            "teacher_top10_frac",
            "teacher_decile9plus_frac",
            "teacher_mean_decile",
        ]:
            m, c = mean_ci95(sub[col].to_numpy())
            out[f"{col}_mean"] = m
            out[f"{col}_ci95"] = c
        sum_rows.append(out)
    trans_sum_df = pd.DataFrame(sum_rows)
    trans_sum_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    trans_sum_df.to_csv(trans_sum_path, index=False)

    score_sum = (
        score_df.groupby(["regime", "split"])
        .agg(
            n_seeds=("seed", "nunique"),
            acc_mean=("acc", "mean"),
            acc_ci95=("acc", lambda x: ci95_mean(np.asarray(x, dtype=np.float64)) if len(x) > 1 else 0.0),
            mean_loss_mean=("mean_loss", "mean"),
            mean_loss_ci95=(
                "mean_loss",
                lambda x: ci95_mean(np.asarray(x, dtype=np.float64)) if len(x) > 1 else 0.0,
            ),
        )
        .reset_index()
    )
    score_sum_path = out_prefix.with_name(out_prefix.name + "_scores_summary.csv")
    score_sum.to_csv(score_sum_path, index=False)

    print("[pathway1-domain] wrote:")
    print(f" - {trans_seed_path}")
    print(f" - {trans_sum_path}")
    print(f" - {score_seed_path}")
    print(f" - {score_sum_path}")


if __name__ == "__main__":
    main()
