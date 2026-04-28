import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

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


def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64, copy=False)
    yy = y.astype(np.float64, copy=False)
    return np.logaddexp(0.0, z) - yy * z


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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _skew_kurt(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd <= 1e-12:
        return {"skew": 0.0, "kurt_excess": 0.0}
    z = (x - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt_excess = float(np.mean(z ** 4) - 3.0)
    return {"skew": skew, "kurt_excess": kurt_excess}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_fold_summary_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument(
        "--selected_rows_pattern",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_h{h}_pathway1_selected_rows_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument("--embeds_root", default="replication_rcg/artifacts/embeds")
    ap.add_argument("--partitions_eval_root", default="replication_rcg/artifacts/partitions_eval")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_loo_conditionality_camloo_foldcal_a10_10s_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold_df = pd.read_csv(args.fold_summary_csv)
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    hist_cache: Dict[str, np.ndarray] = {}
    bin_edges = np.linspace(0.0, 4.0, 60)
    for _, r in fold_df.iterrows():
        fold = str(r["fold"])
        h = int(r["holdout_hospital"])
        fold_res = fold if fold.endswith("_resnet50") else f"{fold}_resnet50"

        embed_dir = Path(args.embeds_root) / fold_res
        X_test = np.load(embed_dir / "X_test.npy", mmap_mode="r")
        y_test = np.load(embed_dir / "y_test.npy")
        meta_test = np.load(embed_dir / "meta_test.npy")
        hospital = meta_test[:, 0].astype(np.int64)

        teacher_meta_path = Path(args.partitions_eval_root) / fold_res / "teacher_difficulty" / "meta.json"
        teacher_meta = json.loads(teacher_meta_path.read_text(encoding="utf-8"))
        teacher_run = Path(teacher_meta["teacher_runs"][0])
        cfg_t = json.loads((teacher_run / "config.json").read_text(encoding="utf-8"))
        model_t = build_head(
            d_in=int(cfg_t["d_in"]),
            hidden_dim=int(cfg_t.get("training", {}).get("hidden_dim", 0)),
            dropout=float(cfg_t.get("training", {}).get("dropout", 0.0)),
        ).to(device)
        ckpt = torch.load(select_ckpt(teacher_run), map_location=device)
        model_t.load_state_dict(ckpt["model_state"])
        logits_t = eval_logits(model_t, X_test, batch_size=int(args.batch_size), device=device)
        loss_t = bce_from_logits(logits_t, y_test)

        # Fold-specific thresholds from selected clipped runs.
        sel_path = Path(str(args.selected_rows_pattern).format(h=h))
        sel = pd.read_csv(sel_path)
        med_t95 = float(
            sel[sel["regime"].str.contains(f"p95_a10_cam_loo_h{h}cal", regex=False)]["clip_loss"].median()
        )
        med_t97 = float(
            sel[sel["regime"].str.contains(f"p97_a10_cam_loo_h{h}cal", regex=False)]["clip_loss"].median()
        )
        med_t99 = float(
            sel[sel["regime"].str.contains(f"p99_a10_cam_loo_h{h}cal", regex=False)]["clip_loss"].median()
        )

        q50, q90, q95, q99 = np.quantile(loss_t, [0.5, 0.9, 0.95, 0.99]).tolist()
        moments = _skew_kurt(loss_t)

        hist_cache[fold] = loss_t
        rows.append(
            {
                "fold": fold,
                "holdout_hospital": h,
                "stage_success_rate": float(r["seeds_late_top10_gt_early"]) / float(r["n_seeds"]),
                "stage_p_one_sided_sign": float(r["p_one_sided_sign"]),
                "late_minus_early_top10": float(r["mean_diff_top10_rec_late_minus_early"]),
                "late_minus_early_decile": float(r["mean_diff_decile_rec_late_minus_early"]),
                "test_acc_base": float(r["test_acc_base"]),
                "test_acc_p95": float(r["test_acc_p95"]),
                "test_acc_p97": float(r["test_acc_p97"]),
                "test_acc_p99": float(r["test_acc_p99"]),
                "teacher_loss_mean_test": float(loss_t.mean()),
                "teacher_loss_std_test": float(loss_t.std(ddof=0)),
                "teacher_loss_q50_test": float(q50),
                "teacher_loss_q90_test": float(q90),
                "teacher_loss_q95_test": float(q95),
                "teacher_loss_q99_test": float(q99),
                "teacher_tail_gap_99_90": float(q99 - q90),
                "teacher_tail_gap_95_50": float(q95 - q50),
                "teacher_tail_ratio_99_90_over_95_50": float((q99 - q90) / max(q95 - q50, 1e-8)),
                "teacher_skew_test": float(moments["skew"]),
                "teacher_kurt_excess_test": float(moments["kurt_excess"]),
                "frac_test_above_t95med": float(np.mean(loss_t > med_t95)),
                "frac_test_above_t97med": float(np.mean(loss_t > med_t97)),
                "frac_test_above_t99med": float(np.mean(loss_t > med_t99)),
                "teacher_mean_loss_hospital0": float(loss_t[hospital == 0].mean()) if np.any(hospital == 0) else np.nan,
                "teacher_mean_loss_hospital1": float(loss_t[hospital == 1].mean()) if np.any(hospital == 1) else np.nan,
                "teacher_mean_loss_hospital2": float(loss_t[hospital == 2].mean()) if np.any(hospital == 2) else np.nan,
                "teacher_mean_loss_hospital3": float(loss_t[hospital == 3].mean()) if np.any(hospital == 3) else np.nan,
                "teacher_mean_loss_hospital4": float(loss_t[hospital == 4].mean()) if np.any(hospital == 4) else np.nan,
            }
        )

    feat = pd.DataFrame(rows).sort_values("holdout_hospital")
    feat_csv = out_prefix.with_name(out_prefix.name + "_fold_features.csv")
    feat.to_csv(feat_csv, index=False)

    key = "stage_success_rate"
    corr_rows = []
    for col in feat.columns:
        if col in {"fold"}:
            continue
        if col == key:
            continue
        if not np.issubdtype(feat[col].dtype, np.number):
            continue
        corr_rows.append({"feature": col, "pearson_r_vs_stage_success_rate": _safe_corr(feat[col].to_numpy(), feat[key].to_numpy())})
    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r_vs_stage_success_rate", key=lambda s: np.abs(s), ascending=False)
    corr_csv = out_prefix.with_name(out_prefix.name + "_feature_corr.csv")
    corr_df.to_csv(corr_csv, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (fold, arr) in enumerate(sorted(hist_cache.items(), key=lambda x: int(x[0].split("_h")[-1]))):
        h = int(fold.split("_h")[-1])
        ax = axes[i]
        ax.hist(arr, bins=bin_edges, density=True, alpha=0.5, color=colors[i], label=f"h{h}")
        ax.set_title(f"{fold} (holdout h{h})")
        ax.set_xlabel("Teacher BCE loss (test)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)
    if len(hist_cache) < len(axes):
        axes[-1].axis("off")
    hist_png = out_prefix.with_name(out_prefix.name + "_teacher_loss_hists.png")
    fig.savefig(hist_png, dpi=180)
    plt.close(fig)

    # Scatter quick-read panel for top correlated features.
    top_feats = [x for x in corr_df["feature"].head(4).tolist() if x in feat.columns]
    if top_feats:
        fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axs2 = axs2.ravel()
        for i, col in enumerate(top_feats):
            ax = axs2[i]
            x = feat[col].to_numpy(dtype=float)
            y = feat[key].to_numpy(dtype=float)
            ax.scatter(x, y, s=55, c=np.arange(len(x)), cmap="viridis")
            for _, rr in feat.iterrows():
                ax.annotate(f"h{int(rr['holdout_hospital'])}", (rr[col], rr[key]), fontsize=8, xytext=(2, 2), textcoords="offset points")
            ax.set_xlabel(col)
            ax.set_ylabel(key)
            ax.set_title(f"r={_safe_corr(x, y):.3f}")
        for j in range(len(top_feats), 4):
            axs2[j].axis("off")
        sc_png = out_prefix.with_name(out_prefix.name + "_feature_scatter.png")
        fig2.savefig(sc_png, dpi=180)
        plt.close(fig2)
    else:
        sc_png = None

    print("[loo-conditionality] wrote:")
    print(f" - {feat_csv}")
    print(f" - {corr_csv}")
    print(f" - {hist_png}")
    if sc_png is not None:
        print(f" - {sc_png}")


if __name__ == "__main__":
    main()
