import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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


def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64, copy=False)
    yy = y.astype(np.float64, copy=False)
    return np.logaddexp(0.0, z) - yy * z


def _resolve_run_dir(runs_root: Path, regime: str, seed: int, run_dir_hint: Optional[str]) -> Path:
    if run_dir_hint:
        p = Path(run_dir_hint)
        if p.exists():
            return p
    seed_dir = runs_root / "camelyon17" / regime / f"seed{int(seed)}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed dir: {seed_dir}")
    cands = sorted([d for d in seed_dir.iterdir() if d.is_dir()])
    if not cands:
        raise FileNotFoundError(f"No run tags under {seed_dir}")
    if len(cands) > 1:
        # Keep deterministic behavior; latest folder tends to be the rerun.
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _per_example_grad_mag(
    logits_np: np.ndarray,
    y_np: np.ndarray,
    label_smoothing: float,
    focal_gamma: float,
    clip_loss: float,
    clip_alpha: float,
    batch_size: int = 32768,
) -> np.ndarray:
    out = np.zeros_like(logits_np, dtype=np.float32)
    n = int(logits_np.shape[0])
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        logits = torch.from_numpy(np.array(logits_np[s:e], dtype=np.float32, copy=True)).requires_grad_(True)
        y = torch.from_numpy(np.asarray(y_np[s:e], dtype=np.float32))
        ce = nn.BCEWithLogitsLoss(reduction="none")(logits, y)

        if label_smoothing > 0.0 and focal_gamma > 0.0:
            raise ValueError("Both label_smoothing and focal_gamma are active; unsupported.")

        if label_smoothing > 0.0:
            y_s = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            losses = nn.BCEWithLogitsLoss(reduction="none")(logits, y_s)
        elif focal_gamma > 0.0:
            p = torch.sigmoid(logits)
            p_t = p * y + (1.0 - p) * (1.0 - y)
            losses = ce * torch.pow((1.0 - p_t).clamp_min(1e-8), float(focal_gamma))
        else:
            losses = ce

        if clip_loss > 0.0:
            if clip_alpha > 0.0:
                losses = torch.where(losses <= clip_loss, losses, clip_loss + clip_alpha * (losses - clip_loss))
            else:
                losses = torch.clamp(losses, max=clip_loss)

        grad = torch.autograd.grad(losses.sum(), logits, retain_graph=False, create_graph=False)[0]
        out[s:e] = grad.abs().detach().cpu().numpy().astype(np.float32)
    return out


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="replication_rcg/runs")
    ap.add_argument("--embeds_dir", default="replication_rcg/artifacts/embeds/camelyon17_resnet50")
    ap.add_argument(
        "--teacher_bins_validation",
        default="replication_rcg/artifacts/partitions_eval/camelyon17_resnet50/teacher_difficulty/bankA/val_skew/diff_m00_K64.npy",
    )
    ap.add_argument(
        "--softclip_rows_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_tail_distortion_rows_cam_softclip_a10_p99_20260227.csv",
    )
    ap.add_argument(
        "--labelsmooth_rows_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_cam_labelsmooth_n10_20260304_distortion_rows_20260304.csv",
    )
    ap.add_argument(
        "--focal_rows_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_cam_focal_n10_20260304_distortion_rows_20260304.csv",
    )
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_objective_weighting_signflip_20260305")
    ap.add_argument("--max_runs", type=int, default=0, help="Optional cap for quick smoke tests (0 = all).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    y_val = np.load(Path(args.embeds_dir) / "y_val_skew.npy").astype(np.int64)
    bins = np.load(args.teacher_bins_validation)
    K = int(np.nanmax(bins)) + 1
    tail_start = int(np.floor(0.9 * K))
    tail_mask = bins >= tail_start
    core_mask = ~tail_mask

    rows_soft = pd.read_csv(args.softclip_rows_csv)[["regime", "seed", "epoch_selected", "run_dir", "clip_loss", "clip_alpha"]].copy()
    rows_soft = rows_soft.rename(columns={"epoch_selected": "epoch"})
    rows_soft["label_smoothing"] = 0.0
    rows_soft["focal_gamma"] = 0.0
    rows_soft["source"] = "softclip"

    rows_ls = pd.read_csv(args.labelsmooth_rows_csv)[["regime", "seed", "epoch", "label_smoothing", "focal_gamma"]].copy()
    rows_ls["run_dir"] = ""
    rows_ls["clip_loss"] = 0.0
    rows_ls["clip_alpha"] = 1.0
    rows_ls["source"] = "labelsmooth"

    rows_fc = pd.read_csv(args.focal_rows_csv)[["regime", "seed", "epoch", "label_smoothing", "focal_gamma"]].copy()
    rows_fc["run_dir"] = ""
    rows_fc["clip_loss"] = 0.0
    rows_fc["clip_alpha"] = 1.0
    rows_fc["source"] = "focal"

    all_rows = pd.concat([rows_soft, rows_ls, rows_fc], ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=["regime", "seed", "epoch"]).reset_index(drop=True)
    if int(args.max_runs) > 0:
        all_rows = all_rows.head(int(args.max_runs)).copy()

    out = []
    runs_root = Path(args.runs_root)
    eps = 1e-8
    for _, rr in all_rows.iterrows():
        regime = str(rr["regime"])
        seed = int(rr["seed"])
        epoch = int(rr["epoch"])
        run_dir = _resolve_run_dir(runs_root, regime, seed, str(rr.get("run_dir", "")))
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        ls = float(rr.get("label_smoothing", cfg.get("label_smoothing", 0.0)) or 0.0)
        fg = float(rr.get("focal_gamma", cfg.get("focal_gamma", 0.0)) or 0.0)
        cl = float(rr.get("clip_loss", cfg.get("clip_loss", 0.0)) or 0.0)
        ca = float(rr.get("clip_alpha", cfg.get("clip_alpha", 1.0)) or 1.0)

        logits_path = run_dir / "val_logits_by_epoch.npy"
        logits_all = np.load(logits_path, mmap_mode="r")
        if epoch < 1 or epoch > logits_all.shape[0]:
            raise ValueError(f"Epoch {epoch} out of range for {run_dir}")
        logits = np.asarray(logits_all[epoch - 1], dtype=np.float32)

        ce_grad_mag = np.abs(1.0 / (1.0 + np.exp(-logits)) - y_val.astype(np.float32))
        obj_grad_mag = _per_example_grad_mag(
            logits_np=logits,
            y_np=y_val,
            label_smoothing=ls,
            focal_gamma=fg,
            clip_loss=cl,
            clip_alpha=ca,
        )
        ratio = obj_grad_mag / np.maximum(ce_grad_mag, eps)

        # Loss-decile profile for readability.
        ce_loss = bce_from_logits(logits, y_val)
        q = np.quantile(ce_loss, np.linspace(0.0, 1.0, 11))
        dec = np.digitize(ce_loss, q[1:-1], right=True)
        dec_means = []
        for d in range(10):
            m = dec == d
            dec_means.append(float(ratio[m].mean()) if np.any(m) else np.nan)

        out.append(
            {
                "regime": regime,
                "seed": seed,
                "epoch": epoch,
                "label_smoothing": ls,
                "focal_gamma": fg,
                "clip_loss": cl,
                "clip_alpha": ca,
                "mean_ratio_all": float(ratio.mean()),
                "mean_ratio_core": float(ratio[core_mask].mean()),
                "mean_ratio_tail": float(ratio[tail_mask].mean()),
                "tail_over_core_ratio": float(ratio[tail_mask].mean() / max(ratio[core_mask].mean(), eps)),
                "mean_grad_ce_all": float(ce_grad_mag.mean()),
                "mean_grad_obj_all": float(obj_grad_mag.mean()),
                "source": str(rr["source"]),
                **{f"ratio_lossdecile_{d+1:02d}": dec_means[d] for d in range(10)},
            }
        )

    out_df = pd.DataFrame(out).sort_values(["source", "regime", "seed"])
    rows_csv = out_prefix.with_name(out_prefix.name + "_rows.csv")
    out_df.to_csv(rows_csv, index=False)

    summary = (
        out_df.groupby(["source", "regime"])
        .agg(
            n=("seed", "count"),
            mean_ratio_all=("mean_ratio_all", "mean"),
            mean_ratio_core=("mean_ratio_core", "mean"),
            mean_ratio_tail=("mean_ratio_tail", "mean"),
            tail_over_core_ratio=("tail_over_core_ratio", "mean"),
        )
        .reset_index()
    )
    for col in ["mean_ratio_all", "mean_ratio_core", "mean_ratio_tail", "tail_over_core_ratio"]:
        cis = out_df.groupby(["source", "regime"])[col].apply(lambda s: _ci95(s.to_numpy())).reset_index(name=f"{col}_ci95")
        summary = summary.merge(cis, on=["source", "regime"], how="left")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Plot 1: tail/core ratio by regime.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(summary.shape[0])
    ax.bar(x, summary["tail_over_core_ratio"].to_numpy(), color="tab:blue", alpha=0.8)
    ax.errorbar(
        x,
        summary["tail_over_core_ratio"].to_numpy(),
        yerr=summary["tail_over_core_ratio_ci95"].to_numpy(),
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}:{r}" for s, r in zip(summary["source"], summary["regime"])], rotation=45, ha="right")
    ax.set_ylabel("Tail/Core implicit gradient ratio")
    ax.set_title("Objective-specific weighting sign/strength by regime")
    fig.tight_layout()
    bar_png = out_prefix.with_name(out_prefix.name + "_tail_core_bar.png")
    fig.savefig(bar_png, dpi=180)
    plt.close(fig)

    # Plot 2: loss-decile profiles for selected regimes.
    fig2, ax2 = plt.subplots(figsize=(8.5, 5))
    show_regimes = []
    for cand in ["rcgdro_softclip_p95_a10_cam", "erm_labelsmooth_e10_cam", "erm_focal_g2_cam", "erm"]:
        if cand in out_df["regime"].values:
            show_regimes.append(cand)
    for reg in show_regimes:
        sub = out_df[out_df["regime"] == reg]
        prof = np.array([sub[f"ratio_lossdecile_{d+1:02d}"].mean() for d in range(10)], dtype=float)
        ax2.plot(np.arange(1, 11), prof, marker="o", label=reg)
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("CE-loss decile (validation)")
    ax2.set_ylabel("Implicit gradient ratio vs CE")
    ax2.set_title("Sign-flip view: objective weighting over difficulty deciles")
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    line_png = out_prefix.with_name(out_prefix.name + "_decile_profiles.png")
    fig2.savefig(line_png, dpi=180)
    plt.close(fig2)

    print("[objective-weighting] wrote:")
    print(f" - {rows_csv}")
    print(f" - {summary_csv}")
    print(f" - {bar_png}")
    print(f" - {line_png}")


if __name__ == "__main__":
    main()
