import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


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
    out: List[np.ndarray] = []
    use_amp = device.startswith("cuda")
    for i in range(0, int(X.shape[0]), int(batch_size)):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb).squeeze(1)
        else:
            logits = model(xb).squeeze(1)
        out.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64, copy=False)
    yy = y.astype(np.float64, copy=False)
    return np.logaddexp(0.0, z) - yy * z


def select_ckpt(run_dir: Path) -> Path:
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        best_epoch = None
        best_acc = -1.0
        for line in metrics.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = int(rec.get("epoch", 0))
            acc = float(rec.get("val_acc", -1.0))
            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
        if best_epoch is not None:
            p = run_dir / f"ckpt_epoch{best_epoch:03d}.pt"
            if p.exists():
                return p
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return ckpts[-1]


def _teacher_losses_and_deciles(
    fold_res: str,
    split: str,
    embeds_root: Path,
    partitions_root: Path,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    teacher_meta_path = partitions_root / fold_res / "teacher_difficulty" / "meta.json"
    tmeta = json.loads(teacher_meta_path.read_text(encoding="utf-8"))
    teacher_run = Path(tmeta["teacher_runs"][0])
    cfg = json.loads((teacher_run / "config.json").read_text(encoding="utf-8"))
    model = build_head(
        d_in=int(cfg["d_in"]),
        hidden_dim=int(cfg.get("training", {}).get("hidden_dim", 0)),
        dropout=float(cfg.get("training", {}).get("dropout", 0.0)),
    ).to(device)
    ckpt = torch.load(select_ckpt(teacher_run), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    X = np.load(embeds_root / fold_res / f"X_{split}.npy", mmap_mode="r")
    y = np.load(embeds_root / fold_res / f"y_{split}.npy")
    logits = eval_logits(model, X, batch_size=batch_size, device=device)
    losses = bce_from_logits(logits, y)
    K = int(tmeta.get("num_cells", 64))
    q = np.linspace(0.0, 1.0, K + 1)
    edges = np.quantile(losses, q)
    bins = np.digitize(losses, edges[1:-1], right=True)
    dec = np.minimum((bins * 10) // K, 9) + 1
    return losses.astype(np.float32), dec.astype(np.int16)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--suffix", default="camloo_foldcal_a10_10s_20260304")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metrics_root", default="replication_rcg/artifacts/metrics")
    ap.add_argument("--embeds_root", default="replication_rcg/artifacts/embeds")
    ap.add_argument("--partitions_eval_root", default="replication_rcg/artifacts/partitions_eval")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_transition_canonical_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    folds = [int(x) for x in str(args.folds).split(",") if x.strip()]
    split = str(args.split)
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    metrics_root = Path(args.metrics_root)
    embeds_root = Path(args.embeds_root)
    partitions_root = Path(args.partitions_eval_root)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    major_rows: List[dict] = []
    seed_rows: List[dict] = []
    fold_rows: List[dict] = []

    for h in folds:
        fold = f"camelyon17_loo_h{h}"
        fold_res = f"{fold}_resnet50"
        selected_csv = metrics_root / f"{fold}_pathway1_selected_rows_{args.suffix}.csv"
        sel = pd.read_csv(selected_csv)
        regimes = [
            "rcgdro",
            f"rcgdro_softclip_p95_a10_cam_loo_h{h}cal",
            f"rcgdro_softclip_p97_a10_cam_loo_h{h}cal",
            f"rcgdro_softclip_p99_a10_cam_loo_h{h}cal",
        ]

        X = np.load(embeds_root / fold_res / f"X_{split}.npy", mmap_mode="r")
        y = np.load(embeds_root / fold_res / f"y_{split}.npy").astype(np.int64)
        meta = np.load(embeds_root / fold_res / f"meta_{split}.npy")
        hospital = meta[:, 0].astype(np.int16)
        teacher_loss, teacher_dec = _teacher_losses_and_deciles(
            fold_res=fold_res,
            split=split,
            embeds_root=embeds_root,
            partitions_root=partitions_root,
            device=device,
            batch_size=int(args.batch_size),
        )

        seeds = sorted(sel["seed"].unique().tolist())
        good_seeds = []
        for s in seeds:
            by_seed = sel[sel["seed"] == s]
            if all((by_seed["regime"] == r).any() for r in regimes):
                good_seeds.append(int(s))
        seeds = good_seeds

        corr: Dict[str, np.ndarray] = {}
        losses: Dict[str, np.ndarray] = {}
        taus: Dict[str, np.ndarray] = {}
        for regime in regimes:
            corr[regime] = np.zeros((len(seeds), y.shape[0]), dtype=np.uint8)
            losses[regime] = np.zeros((len(seeds), y.shape[0]), dtype=np.float32)
            taus[regime] = np.full((len(seeds),), np.nan, dtype=np.float32)

        for si, s in enumerate(seeds):
            by_seed = sel[sel["seed"] == s]
            for regime in regimes:
                rr = by_seed[by_seed["regime"] == regime].iloc[0]
                run_dir = Path(rr["run_dir"])
                ep = int(rr["epoch_selected"])
                cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                model = build_head(
                    d_in=int(cfg["d_in"]),
                    hidden_dim=int(cfg.get("training", {}).get("hidden_dim", 0)),
                    dropout=float(cfg.get("training", {}).get("dropout", 0.0)),
                ).to(device)
                ckpt = torch.load(run_dir / f"ckpt_epoch{ep:03d}.pt", map_location=device)
                model.load_state_dict(ckpt["model_state"])
                logits = eval_logits(model, X, batch_size=int(args.batch_size), device=device)
                pred = (logits >= 0).astype(np.int64)
                l = bce_from_logits(logits, y).astype(np.float32)
                corr[regime][si] = (pred == y).astype(np.uint8)
                losses[regime][si] = l
                if np.isfinite(float(rr.get("clip_loss", np.nan))):
                    taus[regime][si] = float(rr["clip_loss"])

        c0 = corr[regimes[0]].mean(axis=0) >= 0.5
        c95 = corr[regimes[1]].mean(axis=0) >= 0.5
        c97 = corr[regimes[2]].mean(axis=0) >= 0.5
        c99 = corr[regimes[3]].mean(axis=0) >= 0.5
        state_major = np.full(y.shape[0], "fail_p99", dtype=object)
        state_major[c0] = "base_ok"
        state_major[(~c0) & c95] = "rec_p95"
        state_major[(~c95) & c97] = "rec_p97"
        state_major[(~c97) & c99] = "rec_p99"

        for i in range(y.shape[0]):
            major_rows.append(
                {
                    "fold": fold,
                    "holdout_hospital": h,
                    "idx": i,
                    "state_majority": state_major[i],
                    "y": int(y[i]),
                    "hospital": int(hospital[i]),
                    "teacher_decile": int(teacher_dec[i]),
                    "teacher_loss": float(teacher_loss[i]),
                }
            )

        for si, s in enumerate(seeds):
            s0 = corr[regimes[0]][si].astype(bool)
            s95 = corr[regimes[1]][si].astype(bool)
            s97 = corr[regimes[2]][si].astype(bool)
            s99 = corr[regimes[3]][si].astype(bool)
            state_seed = np.full(y.shape[0], "fail_p99", dtype=object)
            state_seed[s0] = "base_ok"
            state_seed[(~s0) & s95] = "rec_p95"
            state_seed[(~s95) & s97] = "rec_p97"
            state_seed[(~s97) & s99] = "rec_p99"
            for st in ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]:
                m = state_seed == st
                if not np.any(m):
                    continue
                seed_rows.append(
                    {
                        "fold": fold,
                        "holdout_hospital": h,
                        "seed": int(s),
                        "state": st,
                        "n": int(m.sum()),
                        "frac": float(m.mean()),
                        "teacher_decile_mean": float(teacher_dec[m].mean()),
                        "teacher_loss_mean": float(teacher_loss[m].mean()),
                    }
                )

        fold_rows.append(
            {
                "fold": fold,
                "holdout_hospital": h,
                "n_examples": int(y.shape[0]),
                "n_seeds": int(len(seeds)),
                "major_base_ok_frac": float(np.mean(state_major == "base_ok")),
                "major_rec_p95_frac": float(np.mean(state_major == "rec_p95")),
                "major_rec_p97_frac": float(np.mean(state_major == "rec_p97")),
                "major_rec_p99_frac": float(np.mean(state_major == "rec_p99")),
                "major_fail_p99_frac": float(np.mean(state_major == "fail_p99")),
            }
        )

        cache_path = out_prefix.with_name(out_prefix.name + f"_cache_{fold}.npz")
        np.savez_compressed(
            cache_path,
            y=y.astype(np.int8),
            hospital=hospital.astype(np.int8),
            teacher_decile=teacher_dec.astype(np.int16),
            teacher_loss=teacher_loss.astype(np.float32),
            seeds=np.asarray(seeds, dtype=np.int16),
            correct_base=corr[regimes[0]].astype(np.uint8),
            correct_p95=corr[regimes[1]].astype(np.uint8),
            correct_p97=corr[regimes[2]].astype(np.uint8),
            correct_p99=corr[regimes[3]].astype(np.uint8),
            loss_base=losses[regimes[0]].astype(np.float32),
            loss_p95=losses[regimes[1]].astype(np.float32),
            loss_p97=losses[regimes[2]].astype(np.float32),
            loss_p99=losses[regimes[3]].astype(np.float32),
            tau_p95=taus[regimes[1]].astype(np.float32),
            tau_p97=taus[regimes[2]].astype(np.float32),
            tau_p99=taus[regimes[3]].astype(np.float32),
        )
        print(f"[transition-canonical] wrote cache: {cache_path}")

    major_df = pd.DataFrame(major_rows)
    seed_df = pd.DataFrame(seed_rows)
    fold_df = pd.DataFrame(fold_rows)
    major_csv = out_prefix.with_name(out_prefix.name + "_majority_rows.csv")
    seed_csv = out_prefix.with_name(out_prefix.name + "_seed_summary.csv")
    fold_csv = out_prefix.with_name(out_prefix.name + "_fold_summary.csv")
    major_df.to_csv(major_csv, index=False)
    seed_df.to_csv(seed_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)

    print("[transition-canonical] wrote:")
    print(f" - {major_csv}")
    print(f" - {seed_csv}")
    print(f" - {fold_csv}")


if __name__ == "__main__":
    main()
