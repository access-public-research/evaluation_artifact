import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


def _teacher_deciles(
    fold_res: str,
    split: str,
    embeds_root: Path,
    partitions_root: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    meta_path = partitions_root / fold_res / "teacher_difficulty" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    K = int(meta.get("num_cells", 64))
    bin_path = partitions_root / fold_res / "teacher_difficulty" / "bankA" / split / f"diff_m00_K{K}.npy"
    if bin_path.exists():
        bins = np.load(bin_path)
        return np.minimum((bins * 10) // K, 9) + 1

    teacher_run = Path(meta["teacher_runs"][0])
    cfg_t = json.loads((teacher_run / "config.json").read_text(encoding="utf-8"))
    model_t = build_head(
        d_in=int(cfg_t["d_in"]),
        hidden_dim=int(cfg_t.get("training", {}).get("hidden_dim", 0)),
        dropout=float(cfg_t.get("training", {}).get("dropout", 0.0)),
    ).to(device)
    ckpt = torch.load(sorted(teacher_run.glob("ckpt_epoch*.pt"))[-1], map_location=device)
    model_t.load_state_dict(ckpt["model_state"])
    X = np.load(embeds_root / fold_res / f"X_{split}.npy", mmap_mode="r")
    y = np.load(embeds_root / fold_res / f"y_{split}.npy")
    logits = eval_logits(model_t, X, batch_size=batch_size, device=device)
    loss = np.logaddexp(0.0, logits.astype(np.float64)) - y.astype(np.float64) * logits.astype(np.float64)
    q = np.linspace(0.0, 1.0, K + 1)
    edges = np.quantile(loss, q)
    bins = np.digitize(loss, edges[1:-1], right=True)
    return np.minimum((bins * 10) // K, 9) + 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="camloo_foldcal_a10_10s_20260304")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeds_root", default="replication_rcg/artifacts/embeds")
    ap.add_argument("--partitions_eval_root", default="replication_rcg/artifacts/partitions_eval")
    ap.add_argument("--metrics_root", default="replication_rcg/artifacts/metrics")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_points_per_state", type=int, default=4000)
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_loo_transition_geometry_20260305")
    return ap.parse_args()


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    evr = var[:2] / max(var.sum(), 1e-12)
    return Z.astype(np.float32), evr.astype(np.float32)


def main() -> None:
    args = parse_args()
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    folds = [int(x) for x in str(args.folds).split(",") if x.strip()]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_points = []
    rng = np.random.default_rng(0)
    for h in folds:
        fold = f"camelyon17_loo_h{h}"
        fold_res = f"{fold}_resnet50"
        selected_csv = Path(args.metrics_root) / f"{fold}_pathway1_selected_rows_{args.suffix}.csv"
        sel = pd.read_csv(selected_csv)
        regimes = ["rcgdro", f"rcgdro_softclip_p95_a10_cam_loo_h{h}cal", f"rcgdro_softclip_p97_a10_cam_loo_h{h}cal", f"rcgdro_softclip_p99_a10_cam_loo_h{h}cal"]

        X = np.load(Path(args.embeds_root) / fold_res / f"X_{args.split}.npy", mmap_mode="r")
        y = np.load(Path(args.embeds_root) / fold_res / f"y_{args.split}.npy").astype(np.int64)
        meta = np.load(Path(args.embeds_root) / fold_res / f"meta_{args.split}.npy")
        hospital = meta[:, 0].astype(np.int64)
        teacher_decile = _teacher_deciles(
            fold_res=fold_res,
            split=args.split,
            embeds_root=Path(args.embeds_root),
            partitions_root=Path(args.partitions_eval_root),
            batch_size=int(args.batch_size),
            device=device,
        )

        preds: Dict[str, List[np.ndarray]] = {r: [] for r in regimes}
        for seed in sorted(sel["seed"].unique().tolist()):
            by_seed = sel[sel["seed"] == seed]
            ok = True
            for r in regimes:
                if not (by_seed["regime"] == r).any():
                    ok = False
                    break
            if not ok:
                continue

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
                preds[regime].append((logits >= 0).astype(np.int64))

        maj_correct = {}
        for regime in regimes:
            if not preds[regime]:
                raise RuntimeError(f"No predictions assembled for {fold} {regime}")
            p = np.stack(preds[regime], axis=0)
            maj = (p.mean(axis=0) >= 0.5).astype(np.int64)
            maj_correct[regime] = (maj == y)

        c0 = maj_correct[regimes[0]]
        c95 = maj_correct[regimes[1]]
        c97 = maj_correct[regimes[2]]
        c99 = maj_correct[regimes[3]]
        state = np.full(y.shape[0], "fail_p99", dtype=object)
        state[c0] = "base_ok"
        state[(~c0) & c95] = "rec_p95"
        state[(~c95) & c97] = "rec_p97"
        state[(~c97) & c99] = "rec_p99"

        # Low-cost PCA for visual anatomy.
        Z, evr = _pca_2d(np.asarray(X, dtype=np.float32))
        all_rows.append(
            {
                "fold": fold,
                "holdout_hospital": h,
                "n": int(y.shape[0]),
                "pca_var1": float(evr[0]),
                "pca_var2": float(evr[1]),
                "base_ok_frac": float(np.mean(state == "base_ok")),
                "rec_p95_frac": float(np.mean(state == "rec_p95")),
                "rec_p97_frac": float(np.mean(state == "rec_p97")),
                "rec_p99_frac": float(np.mean(state == "rec_p99")),
                "fail_p99_frac": float(np.mean(state == "fail_p99")),
                "rec_p95_teacher_decile_mean": float(teacher_decile[state == "rec_p95"].mean()) if np.any(state == "rec_p95") else np.nan,
                "rec_p97_teacher_decile_mean": float(teacher_decile[state == "rec_p97"].mean()) if np.any(state == "rec_p97") else np.nan,
                "rec_p99_teacher_decile_mean": float(teacher_decile[state == "rec_p99"].mean()) if np.any(state == "rec_p99") else np.nan,
            }
        )

        fold_df = pd.DataFrame(
            {
                "fold": fold,
                "holdout_hospital": h,
                "idx": np.arange(y.shape[0], dtype=np.int64),
                "pca1": Z[:, 0],
                "pca2": Z[:, 1],
                "state": state,
                "teacher_decile": teacher_decile,
                "hospital": hospital,
                "y": y,
            }
        )
        all_points.append(fold_df)

    rows_df = pd.DataFrame(all_rows).sort_values("holdout_hospital")
    points_df = pd.concat(all_points, ignore_index=True)
    rows_csv = out_prefix.with_name(out_prefix.name + "_fold_summary.csv")
    points_csv = out_prefix.with_name(out_prefix.name + "_points.csv")
    rows_df.to_csv(rows_csv, index=False)
    points_df.to_csv(points_csv, index=False)

    # Faceted plot (one panel per fold).
    states = ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]
    colors = {
        "base_ok": "#999999",
        "rec_p95": "#1f77b4",
        "rec_p97": "#2ca02c",
        "rec_p99": "#ff7f0e",
        "fail_p99": "#d62728",
    }
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axs = axs.ravel()
    for i, h in enumerate(folds):
        ax = axs[i]
        fold = f"camelyon17_loo_h{h}"
        sub = points_df[points_df["fold"] == fold]
        for st in states:
            s = sub[sub["state"] == st]
            if s.empty:
                continue
            n = min(int(args.max_points_per_state), int(s.shape[0]))
            pick = s.sample(n=n, random_state=0) if s.shape[0] > n else s
            ax.scatter(pick["pca1"], pick["pca2"], s=6, alpha=0.6, c=colors[st], label=st)
        ax.set_title(f"{fold}")
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
    if len(folds) < len(axs):
        axs[-1].axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01), fontsize=8)
    png = out_prefix.with_name(out_prefix.name + "_facet_pca.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print("[loo-geometry] wrote:")
    print(f" - {rows_csv}")
    print(f" - {points_csv}")
    print(f" - {png}")


if __name__ == "__main__":
    main()
