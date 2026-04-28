import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean
import torch


def _project(X: np.ndarray, out_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = int(X.shape[1])
    W = rng.standard_normal((d, int(out_dim)), dtype=np.float32) / np.sqrt(float(d))
    Z = np.asarray(X, dtype=np.float32) @ W
    return Z.astype(np.float32)


def _knn_stats(
    query: np.ndarray,
    ref: np.ndarray,
    ref_is_hard: np.ndarray,
    ks: List[int],
    batch_size: int,
    device: str,
) -> Dict[str, np.ndarray]:
    max_k = int(max(ks))
    q = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(device)
    r = torch.from_numpy(np.asarray(ref, dtype=np.float32)).to(device)
    hard = torch.from_numpy(np.asarray(ref_is_hard, dtype=np.uint8)).to(device)

    out_dist = {k: [] for k in ks}
    out_hfrac = {k: [] for k in ks}

    for i in range(0, q.shape[0], int(batch_size)):
        qb = q[i : i + int(batch_size)]
        dmat = torch.cdist(qb, r)  # [b, n_ref]
        vals, idx = torch.topk(dmat, k=max_k, largest=False, dim=1)
        hard_nn = hard[idx].float()
        for k in ks:
            vv = vals[:, :k].mean(dim=1).detach().cpu().numpy().astype(np.float32)
            hh = hard_nn[:, :k].mean(dim=1).detach().cpu().numpy().astype(np.float32)
            out_dist[k].append(vv)
            out_hfrac[k].append(hh)

    dist = {k: np.concatenate(out_dist[k], axis=0) for k in ks}
    hfrac = {k: np.concatenate(out_hfrac[k], axis=0) for k in ks}
    return {"dist": dist, "hfrac": hfrac}


def _ci95(x: np.ndarray) -> float:
    return ci95_mean(np.asarray(x, dtype=np.float64))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--majority_rows_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_transition_canonical_20260305_majority_rows.csv",
    )
    ap.add_argument("--embeds_root", default="replication_rcg/artifacts/embeds")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--proj_dim", type=int, default=32)
    ap.add_argument("--k_list", default="25,50,100")
    ap.add_argument("--hard_decile_min", type=int, default=9)
    ap.add_argument("--max_query_per_state", type=int, default=2500)
    ap.add_argument("--max_ref_full", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_density_topology_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.majority_rows_csv)
    folds = [int(x) for x in str(args.folds).split(",") if x.strip()]
    ks = [int(x) for x in str(args.k_list).split(",") if x.strip()]
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    row_out: List[dict] = []
    sum_out: List[dict] = []

    for h in folds:
        fold = f"camelyon17_loo_h{h}"
        fold_res = f"{fold}_resnet50"
        sub = rows[rows["fold"] == fold].copy()
        if sub.empty:
            continue

        X = np.load(Path(args.embeds_root) / fold_res / f"X_{args.split}.npy", mmap_mode="r")
        idx = sub["idx"].to_numpy(dtype=np.int64)
        Xf = np.asarray(X[idx], dtype=np.float32)
        Z = _project(Xf, out_dim=int(args.proj_dim), seed=123 + h)

        teacher_dec = sub["teacher_decile"].to_numpy(dtype=np.int16)
        hard_mask = teacher_dec >= int(args.hard_decile_min)
        if not np.any(hard_mask):
            continue

        ref_hard = Z[hard_mask]
        ref_full_idx = np.arange(Z.shape[0], dtype=np.int64)
        if ref_full_idx.size > int(args.max_ref_full):
            ref_full_idx = rng.choice(ref_full_idx, size=int(args.max_ref_full), replace=False)
        ref_full = Z[ref_full_idx]
        ref_full_hard = hard_mask[ref_full_idx].astype(np.uint8)

        for st in ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]:
            sidx = np.where(sub["state_majority"].to_numpy() == st)[0]
            if sidx.size == 0:
                continue
            if sidx.size > int(args.max_query_per_state):
                sidx = rng.choice(sidx, size=int(args.max_query_per_state), replace=False)
            q = Z[sidx]

            hard_stats = _knn_stats(
                query=q,
                ref=ref_hard,
                ref_is_hard=np.ones((ref_hard.shape[0],), dtype=np.uint8),
                ks=ks,
                batch_size=int(args.batch_size),
                device=device,
            )
            full_stats = _knn_stats(
                query=q,
                ref=ref_full,
                ref_is_hard=ref_full_hard,
                ks=ks,
                batch_size=int(args.batch_size),
                device=device,
            )

            for j, local_idx in enumerate(sidx):
                for k in ks:
                    d_hard = float(hard_stats["dist"][k][j])
                    hf = float(full_stats["hfrac"][k][j])
                    row_out.append(
                        {
                            "fold": fold,
                            "holdout_hospital": h,
                            "state": st,
                            "idx": int(local_idx),
                            "k": int(k),
                            "knn_dist_hard": d_hard,
                            "hard_neighbor_frac": hf,
                            "local_hard_density": float(1.0 / max(d_hard, 1e-8)),
                        }
                    )

    rows_df = pd.DataFrame(row_out)
    rows_csv = out_prefix.with_name(out_prefix.name + "_rows.csv")
    rows_df.to_csv(rows_csv, index=False)

    if rows_df.empty:
        raise RuntimeError("No density rows produced.")

    g = rows_df.groupby(["fold", "holdout_hospital", "state", "k"])
    for (fold, h, st, k), d in g:
        for col in ["knn_dist_hard", "hard_neighbor_frac", "local_hard_density"]:
            x = d[col].to_numpy(dtype=np.float64)
            sum_out.append(
                {
                    "fold": fold,
                    "holdout_hospital": int(h),
                    "state": st,
                    "k": int(k),
                    "metric": col,
                    "mean": float(x.mean()),
                    "ci95": _ci95(x),
                    "n": int(x.size),
                }
            )

    sum_df = pd.DataFrame(sum_out).sort_values(["holdout_hospital", "k", "state", "metric"])
    sum_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    sum_df.to_csv(sum_csv, index=False)

    # Simple fold-faceted plot for k=50 local hard density.
    k_show = 50 if 50 in set(rows_df["k"].tolist()) else int(rows_df["k"].min())
    p = rows_df[rows_df["k"] == k_show]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axs = axs.ravel()
    order = ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]
    for i, h in enumerate(folds):
        ax = axs[i]
        fold = f"camelyon17_loo_h{h}"
        s = p[p["fold"] == fold]
        data = [s[s["state"] == st]["local_hard_density"].to_numpy(dtype=np.float64) for st in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title(f"{fold} (k={k_show})")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Local hard density")
    if len(folds) < len(axs):
        axs[-1].axis("off")
    png = out_prefix.with_name(out_prefix.name + "_boxplot.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print("[density-topology] wrote:")
    print(f" - {rows_csv}")
    print(f" - {sum_csv}")
    print(f" - {png}")


if __name__ == "__main__":
    main()
