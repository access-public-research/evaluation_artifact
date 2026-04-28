import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
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
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _optional_float(v) -> Optional[float]:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _cfg_or_row(rr: pd.Series, key: str, cfg: dict, default: float) -> float:
    row_val = _optional_float(rr.get(key))
    if row_val is not None:
        return row_val
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    cfg_val = _optional_float(cfg.get(key, train_cfg.get(key, default)))
    if cfg_val is not None:
        return cfg_val
    return float(default)


def _load_softclip_rows(path: str) -> pd.DataFrame:
    rows = pd.read_csv(path).copy()
    if "epoch_selected" in rows.columns and "epoch" not in rows.columns:
        rows = rows.rename(columns={"epoch_selected": "epoch"})
    required = {"regime", "seed", "epoch"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"softclip_rows_csv missing required columns: {missing}")
    if "run_dir" not in rows.columns:
        rows["run_dir"] = ""
    if "clip_loss" not in rows.columns:
        rows["clip_loss"] = np.nan
    if "clip_alpha" not in rows.columns:
        rows["clip_alpha"] = np.nan
    return rows[["regime", "seed", "epoch", "run_dir", "clip_loss", "clip_alpha"]].copy()


def _objective_losses(
    logits: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float,
    focal_gamma: float,
    clip_loss: float,
    clip_alpha: float,
) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss(reduction="none")
    if label_smoothing > 0.0 and focal_gamma > 0.0:
        raise ValueError("Both label_smoothing and focal_gamma are active; unsupported.")
    if label_smoothing > 0.0:
        ys = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        losses = bce(logits, ys)
    else:
        losses = bce(logits, y)
        if focal_gamma > 0.0:
            p = torch.sigmoid(logits)
            p_t = p * y + (1.0 - p) * (1.0 - y)
            losses = losses * torch.pow((1.0 - p_t).clamp_min(1e-8), focal_gamma)
    if clip_loss > 0.0:
        if clip_alpha > 0.0:
            losses = torch.where(losses <= clip_loss, losses, clip_loss + clip_alpha * (losses - clip_loss))
        else:
            losses = torch.clamp(losses, max=clip_loss)
    return losses


def _grad_norm_for_subset(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    label_smoothing: float,
    focal_gamma: float,
    clip_loss: float,
    clip_alpha: float,
    device: str,
) -> float:
    model.zero_grad(set_to_none=True)
    xb = torch.from_numpy(np.asarray(X[indices], dtype=np.float32)).to(device)
    yb = torch.from_numpy(np.asarray(y[indices], dtype=np.float32)).to(device)
    logits = model(xb).squeeze(1)
    losses = _objective_losses(
        logits=logits,
        y=yb,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        clip_loss=clip_loss,
        clip_alpha=clip_alpha,
    )
    loss = losses.mean()
    loss.backward()
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float(torch.sum(g * g).item())
    return float(np.sqrt(max(sq, 0.0)))


def _ci95(x: np.ndarray) -> float:
    return ci95_mean(np.asarray(x, dtype=np.float64))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="replication_rcg/runs")
    ap.add_argument("--embeds_dir", default="replication_rcg/artifacts/embeds/camelyon17_resnet50")
    ap.add_argument(
        "--teacher_bins_train",
        default="replication_rcg/artifacts/partitions_eval/camelyon17_resnet50/teacher_difficulty/bankA/train/diff_m00_K64.npy",
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
    ap.add_argument("--n_per_group", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_runs", type=int, default=0, help="Optional cap for quick smoke tests (0 = all).")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/camelyon17_tail_core_grad_ratio_20260305")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    X_train = np.load(Path(args.embeds_dir) / "X_train.npy", mmap_mode="r")
    y_train = np.load(Path(args.embeds_dir) / "y_train.npy").astype(np.int64)
    train_sub_idx_path = Path(args.embeds_dir) / "train_sub_idx.npy"
    train_sub_idx = np.load(train_sub_idx_path) if train_sub_idx_path.exists() else np.arange(y_train.shape[0], dtype=np.int64)

    bins = np.load(args.teacher_bins_train)
    K = int(np.nanmax(bins)) + 1
    tail_start = int(np.floor(0.9 * K))
    tail_mask_full = bins >= tail_start
    core_mask_full = ~tail_mask_full
    tail_idx = train_sub_idx[tail_mask_full[train_sub_idx]]
    core_idx = train_sub_idx[core_mask_full[train_sub_idx]]
    if tail_idx.size == 0 or core_idx.size == 0:
        raise RuntimeError("Tail/core split empty under teacher bins.")

    rows_soft = _load_softclip_rows(args.softclip_rows_csv)
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

    rng = np.random.default_rng(123)
    out = []
    runs_root = Path(args.runs_root)
    for _, rr in all_rows.iterrows():
        regime = str(rr["regime"])
        seed = int(rr["seed"])
        epoch = int(rr["epoch"])
        run_dir = _resolve_run_dir(runs_root, regime, seed, str(rr.get("run_dir", "")))
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        d_in = int(cfg["d_in"])
        hidden_dim = int(cfg.get("training", {}).get("hidden_dim", 0))
        dropout = float(cfg.get("training", {}).get("dropout", 0.0))
        model = build_head(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout).to(device)
        ckpt = torch.load(run_dir / f"ckpt_epoch{epoch:03d}.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])

        ls = _cfg_or_row(rr, "label_smoothing", cfg, 0.0)
        fg = _cfg_or_row(rr, "focal_gamma", cfg, 0.0)
        cl = _cfg_or_row(rr, "clip_loss", cfg, 0.0)
        ca = _cfg_or_row(rr, "clip_alpha", cfg, 1.0)

        nt = min(int(args.n_per_group), int(tail_idx.shape[0]))
        nc = min(int(args.n_per_group), int(core_idx.shape[0]))
        pick_tail = rng.choice(tail_idx, size=nt, replace=False)
        pick_core = rng.choice(core_idx, size=nc, replace=False)

        g_tail = _grad_norm_for_subset(
            model=model,
            X=X_train,
            y=y_train,
            indices=pick_tail,
            label_smoothing=ls,
            focal_gamma=fg,
            clip_loss=cl,
            clip_alpha=ca,
            device=device,
        )
        g_core = _grad_norm_for_subset(
            model=model,
            X=X_train,
            y=y_train,
            indices=pick_core,
            label_smoothing=ls,
            focal_gamma=fg,
            clip_loss=cl,
            clip_alpha=ca,
            device=device,
        )
        out.append(
            {
                "regime": regime,
                "seed": seed,
                "epoch": epoch,
                "source": str(rr["source"]),
                "label_smoothing": ls,
                "focal_gamma": fg,
                "clip_loss": cl,
                "clip_alpha": ca,
                "n_tail": nt,
                "n_core": nc,
                "grad_norm_tail": g_tail,
                "grad_norm_core": g_core,
                "tail_over_core_grad_ratio": float(g_tail / max(g_core, 1e-8)),
            }
        )

    out_df = pd.DataFrame(out).sort_values(["source", "regime", "seed"])
    rows_csv = out_prefix.with_name(out_prefix.name + "_rows.csv")
    out_df.to_csv(rows_csv, index=False)

    summary = (
        out_df.groupby(["source", "regime"])
        .agg(
            n=("seed", "count"),
            grad_norm_tail=("grad_norm_tail", "mean"),
            grad_norm_core=("grad_norm_core", "mean"),
            tail_over_core_grad_ratio=("tail_over_core_grad_ratio", "mean"),
        )
        .reset_index()
    )
    for col in ["grad_norm_tail", "grad_norm_core", "tail_over_core_grad_ratio"]:
        cis = out_df.groupby(["source", "regime"])[col].apply(lambda s: _ci95(s.to_numpy())).reset_index(name=f"{col}_ci95")
        summary = summary.merge(cis, on=["source", "regime"], how="left")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(summary.shape[0])
    ax.bar(x, summary["tail_over_core_grad_ratio"].to_numpy(), color="tab:orange", alpha=0.85)
    ax.errorbar(
        x,
        summary["tail_over_core_grad_ratio"].to_numpy(),
        yerr=summary["tail_over_core_grad_ratio_ci95"].to_numpy(),
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}:{r}" for s, r in zip(summary["source"], summary["regime"])], rotation=45, ha="right")
    ax.set_ylabel("Tail/Core parameter-gradient norm ratio")
    ax.set_title("Observed gradient redistribution by objective family")
    fig.tight_layout()
    png = out_prefix.with_name(out_prefix.name + "_bar.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print("[grad-ratio] wrote:")
    print(f" - {rows_csv}")
    print(f" - {summary_csv}")
    print(f" - {png}")


if __name__ == "__main__":
    main()
