import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = x[idx].mean(axis=1)
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


def _permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 0) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = float(x.mean() - y.mean())
    z = np.concatenate([x, y], axis=0)
    n = x.size
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        d = float(z[:n].mean() - z[n:].mean())
        if abs(d) >= abs(obs):
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--majority_rows_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_transition_canonical_20260305_majority_rows.csv",
    )
    ap.add_argument("--embeds_root", default="replication_rcg/artifacts/embeds")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--easy_decile_max", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--max_points_plot", type=int, default=5000)
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_alignment_geometry_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    folds = [int(x) for x in str(args.folds).split(",") if x.strip()]
    rows = pd.read_csv(args.majority_rows_csv)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    control_rows: List[dict] = []
    point_rows: List[dict] = []
    rng = np.random.default_rng(0)

    for h in folds:
        fold = f"camelyon17_loo_h{h}"
        fold_res = f"{fold}_resnet50"
        sub = rows[rows["fold"] == fold].copy()
        if sub.empty:
            continue
        X = np.load(Path(args.embeds_root) / fold_res / f"X_{args.split}.npy", mmap_mode="r")
        idx_all = sub["idx"].to_numpy(dtype=np.int64)
        E = np.asarray(X[idx_all], dtype=np.float32)
        E_norm = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-8, None)

        easy_mask = (sub["state_majority"].to_numpy() == "base_ok") & (sub["teacher_decile"].to_numpy() <= int(args.easy_decile_max))
        if int(easy_mask.sum()) < 128:
            easy_mask = sub["state_majority"].to_numpy() == "base_ok"
        proto = E_norm[easy_mask].mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-8)

        cos = np.clip(E_norm @ proto, -1.0, 1.0)
        sub["cos_easy_core"] = cos

        for st in ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]:
            s = sub[sub["state_majority"] == st]
            if s.empty:
                continue
            vals = s["cos_easy_core"].to_numpy(dtype=np.float64)
            lo, hi = _bootstrap_ci(vals, n_boot=int(args.n_boot), seed=123 + h)
            summary_rows.append(
                {
                    "fold": fold,
                    "holdout_hospital": h,
                    "state": st,
                    "n": int(vals.size),
                    "cos_mean": float(vals.mean()),
                    "cos_ci_lo": lo,
                    "cos_ci_hi": hi,
                }
            )

        # Key contrast: late vs early recovery.
        a = sub[sub["state_majority"] == "rec_p99"]["cos_easy_core"].to_numpy(dtype=np.float64)
        b = sub[sub["state_majority"] == "rec_p95"]["cos_easy_core"].to_numpy(dtype=np.float64)
        if a.size > 0 and b.size > 0:
            p_perm = _permutation_p(a, b, n_perm=int(args.n_perm), seed=1234 + h)
            control_rows.append(
                {
                    "fold": fold,
                    "holdout_hospital": h,
                    "contrast": "rec_p99_minus_rec_p95",
                    "mean_diff": float(a.mean() - b.mean()),
                    "p_perm_two_sided": p_perm,
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                }
            )

        # Label+hospital matched contrast against base_ok.
        base = sub[sub["state_majority"] == "base_ok"].copy()
        for st in ["rec_p95", "rec_p97", "rec_p99", "fail_p99"]:
            tgt = sub[sub["state_majority"] == st].copy()
            if tgt.empty or base.empty:
                continue
            diffs = []
            weights = []
            for (yy, hh), tcell in tgt.groupby(["y", "hospital"]):
                bcell = base[(base["y"] == yy) & (base["hospital"] == hh)]
                if bcell.empty:
                    continue
                diffs.append(float(tcell["cos_easy_core"].mean() - bcell["cos_easy_core"].mean()))
                weights.append(float(len(tcell)))
            if weights:
                w = np.asarray(weights, dtype=np.float64)
                d = np.asarray(diffs, dtype=np.float64)
                matched = float(np.sum(w * d) / np.sum(w))
            else:
                matched = np.nan
            control_rows.append(
                {
                    "fold": fold,
                    "holdout_hospital": h,
                    "contrast": f"{st}_minus_baseok_matched_label_hospital",
                    "mean_diff": matched,
                    "p_perm_two_sided": np.nan,
                    "n_a": int(len(tgt)),
                    "n_b": int(len(base)),
                }
            )

        # Downsample points for optional visualization tables.
        for st in ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]:
            sidx = np.where(sub["state_majority"].to_numpy() == st)[0]
            if sidx.size == 0:
                continue
            n = min(int(args.max_points_plot), int(sidx.size))
            choose = rng.choice(sidx, size=n, replace=False) if sidx.size > n else sidx
            tmp = sub.iloc[choose][["fold", "idx", "state_majority", "y", "hospital", "teacher_decile", "teacher_loss", "cos_easy_core"]].copy()
            point_rows.extend(tmp.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows).sort_values(["holdout_hospital", "state"])
    control_df = pd.DataFrame(control_rows).sort_values(["holdout_hospital", "contrast"])
    points_df = pd.DataFrame(point_rows)
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    control_csv = out_prefix.with_name(out_prefix.name + "_controls.csv")
    points_csv = out_prefix.with_name(out_prefix.name + "_points.csv")
    summary_df.to_csv(summary_csv, index=False)
    control_df.to_csv(control_csv, index=False)
    points_df.to_csv(points_csv, index=False)

    # Faceted violin-like boxplot using sampled points.
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axs = axs.ravel()
    order = ["base_ok", "rec_p95", "rec_p97", "rec_p99", "fail_p99"]
    for i, h in enumerate(folds):
        ax = axs[i]
        fold = f"camelyon17_loo_h{h}"
        p = points_df[points_df["fold"] == fold]
        data = [p[p["state_majority"] == st]["cos_easy_core"].to_numpy(dtype=np.float64) for st in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title(f"{fold}")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Cosine to easy-core prototype")
    if len(folds) < len(axs):
        axs[-1].axis("off")
    png = out_prefix.with_name(out_prefix.name + "_boxplot.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print("[alignment-geometry] wrote:")
    print(f" - {summary_csv}")
    print(f" - {control_csv}")
    print(f" - {points_csv}")
    print(f" - {png}")


if __name__ == "__main__":
    main()
