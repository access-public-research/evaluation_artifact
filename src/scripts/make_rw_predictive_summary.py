import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..utils.stats import ci95_mean


def _resolve_run_dir(runs_root: Path, regime: str, seed: int, run_dir_hint: Optional[str]) -> Path:
    if run_dir_hint:
        p = Path(run_dir_hint)
        if p.exists():
            return p
    seed_dir = runs_root / "camelyon17" / regime / f"seed{int(seed)}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed dir: {seed_dir}")
    cands = sorted([d for d in seed_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No run tags under {seed_dir}")
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


def _bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.logaddexp(0.0, z) - yy * z


def _per_example_grad_mag(
    logits_np: np.ndarray,
    y_np: np.ndarray,
    *,
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
        ce = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, y)

        if label_smoothing > 0.0 and focal_gamma > 0.0:
            raise ValueError("At most one of label_smoothing or focal_gamma may be active.")

        if label_smoothing > 0.0:
            y_s = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            losses = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, y_s)
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


def _rw_from_logits(
    logits: np.ndarray,
    y: np.ndarray,
    tail_mask: np.ndarray,
    *,
    label_smoothing: float,
    focal_gamma: float,
    clip_loss: float,
    clip_alpha: float,
) -> tuple[float, float]:
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    ce_grad = np.abs(probs - np.asarray(y, dtype=np.float64))
    ce_loss = _bce_from_logits(logits, y)
    obj_grad = _per_example_grad_mag(
        logits,
        y,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        clip_loss=clip_loss,
        clip_alpha=clip_alpha,
    ).astype(np.float64)
    core_mask = ~tail_mask
    eps = 1e-8
    mean_ce_tail = float(np.mean(ce_grad[tail_mask]))
    mean_ce_core = float(np.mean(ce_grad[core_mask]))
    mean_obj_tail = float(np.mean(obj_grad[tail_mask]))
    mean_obj_core = float(np.mean(obj_grad[core_mask]))
    rw = (mean_obj_tail / max(mean_ce_tail, eps)) / max((mean_obj_core / max(mean_ce_core, eps)), eps)
    gap = float(np.mean(ce_loss[tail_mask] > float(clip_loss)) - np.mean(ce_loss[core_mask] > float(clip_loss))) if clip_loss > 0.0 else 0.0
    return float(rw), gap


def _family_from_regime(regime: str) -> str:
    r = str(regime)
    if "softclip" in r:
        return "softclip"
    if "labelsmooth" in r:
        return "labelsmooth"
    if "focal" in r:
        return "focal"
    return "other"


def _short_label(regime: str, row: pd.Series) -> str:
    if str(regime) == "rcgdro":
        return "ERM baseline"
    fam = _family_from_regime(regime)
    if fam == "softclip":
        if "p95" in regime:
            return "SoftClip P95"
        if "p97" in regime:
            return "SoftClip P97"
        if "p99" in regime:
            return "SoftClip P99"
        return regime
    if fam == "labelsmooth":
        eps = _optional_float(row.get("label_smoothing"))
        return f"LS {eps:.2f}" if eps is not None else regime
    if fam == "focal":
        gamma = _optional_float(row.get("focal_gamma"))
        return f"Focal {gamma:.0f}" if gamma is not None else regime
    return regime


def _ci(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(ci95_mean(arr))


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    xs = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    ys = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    if np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument(
        "--teacher_bins_validation",
        default="artifacts/partitions_eval/camelyon17_resnet50/teacher_difficulty/bankA/val_skew/diff_m00_K64.npy",
    )
    ap.add_argument("--embeds_dir", default="artifacts/embeds/camelyon17_resnet50")
    ap.add_argument(
        "--softclip_rows_csv",
        default="artifacts/metrics/camelyon17_tail_distortion_rows_cam_softclip_a10_p99_20260227.csv",
    )
    ap.add_argument(
        "--labelsmooth_rows_csv",
        default="artifacts/metrics/camelyon17_cam_labelsmooth_n10_20260304_distortion_rows_20260304.csv",
    )
    ap.add_argument(
        "--focal_rows_csv",
        default="artifacts/metrics/camelyon17_cam_focal_n10_20260304_distortion_rows_20260304.csv",
    )
    ap.add_argument("--early_lo", type=int, default=1)
    ap.add_argument("--early_hi", type=int, default=5)
    ap.add_argument(
        "--out_prefix",
        default="artifacts/metrics/camelyon17_rw_predictive_summary_20260329",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    runs_root = repo_root / "runs"
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    y_val = np.load(Path(args.embeds_dir).resolve() / "y_val_skew.npy").astype(np.int64)
    bins = np.load(Path(args.teacher_bins_validation).resolve())
    k = int(np.nanmax(bins)) + 1
    tail_start = int(np.floor(0.9 * k))
    tail_mask = bins >= tail_start

    soft = pd.read_csv(args.softclip_rows_csv).rename(columns={"epoch_selected": "epoch"})
    soft = soft[
        [
            "regime",
            "seed",
            "epoch",
            "run_dir",
            "clip_loss",
            "clip_alpha",
            "tail_delta_vs_baseline",
        ]
    ].copy()
    soft["label_smoothing"] = 0.0
    soft["focal_gamma"] = 0.0
    soft["tail_delta_vs_erm"] = soft["tail_delta_vs_baseline"]
    soft["proxy_delta_vs_erm"] = np.nan
    soft["source"] = "softclip"

    ls = pd.read_csv(args.labelsmooth_rows_csv).copy()
    ls = ls[ls["regime"] != "erm"].copy()
    ls["run_dir"] = ""
    ls["clip_loss"] = 0.0
    ls["clip_alpha"] = 1.0
    ls["source"] = "labelsmooth"

    focal = pd.read_csv(args.focal_rows_csv).copy()
    focal = focal[focal["regime"] != "erm"].copy()
    focal["run_dir"] = ""
    focal["clip_loss"] = 0.0
    focal["clip_alpha"] = 1.0
    focal["source"] = "focal"

    cols = [
        "regime",
        "seed",
        "epoch",
        "run_dir",
        "clip_loss",
        "clip_alpha",
        "label_smoothing",
        "focal_gamma",
        "tail_delta_vs_erm",
        "proxy_delta_vs_erm",
        "source",
    ]
    all_rows = pd.concat([soft[cols], ls[cols], focal[cols]], ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=["regime", "seed"], keep="first").reset_index(drop=True)

    out_rows: list[dict[str, object]] = []
    for _, rr in all_rows.iterrows():
        regime = str(rr["regime"])
        seed = int(rr["seed"])
        run_dir = _resolve_run_dir(runs_root, regime, seed, str(rr.get("run_dir", "")))
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        ls_eps = _cfg_or_row(rr, "label_smoothing", cfg, 0.0)
        focal_gamma = _cfg_or_row(rr, "focal_gamma", cfg, 0.0)
        clip_loss = _cfg_or_row(rr, "clip_loss", cfg, 0.0)
        clip_alpha = _cfg_or_row(rr, "clip_alpha", cfg, 1.0)

        logits_all = np.load(run_dir / "val_logits_by_epoch.npy", mmap_mode="r")
        hi = min(int(args.early_hi), int(logits_all.shape[0]))
        lo = max(1, int(args.early_lo))
        if hi < lo:
            raise ValueError(f"Invalid early window [{lo}, {hi}] for {run_dir}")

        rw_vals = []
        gap_vals = []
        for epoch in range(lo, hi + 1):
            logits = np.asarray(logits_all[epoch - 1], dtype=np.float32)
            rw, gap = _rw_from_logits(
                logits,
                y_val,
                tail_mask,
                label_smoothing=ls_eps,
                focal_gamma=focal_gamma,
                clip_loss=clip_loss,
                clip_alpha=clip_alpha,
            )
            rw_vals.append(rw)
            gap_vals.append(gap)

        early_rw = float(np.mean(rw_vals))
        early_gap = float(np.mean(gap_vals))
        tail_delta = float(rr["tail_delta_vs_erm"])
        proxy_delta = float(rr["proxy_delta_vs_erm"])
        pred_harm = early_rw < 1.0
        actual_harm = tail_delta > 0.0
        out_rows.append(
            {
                "regime": regime,
                "seed": seed,
                "family": _family_from_regime(regime),
                "label": _short_label(regime, rr),
                "selected_epoch": int(rr["epoch"]),
                "early_epoch_lo": lo,
                "early_epoch_hi": hi,
                "early_rw": early_rw,
                "early_gap": early_gap,
                "tail_delta_vs_erm": tail_delta,
                "proxy_delta_vs_erm": proxy_delta,
                "pred_harm_rw_lt1": int(pred_harm),
                "actual_harm_tail_gt0": int(actual_harm),
                "sign_correct": int(pred_harm == actual_harm),
            }
        )

    rows_df = pd.DataFrame(out_rows).sort_values(["family", "regime", "seed"]).reset_index(drop=True)
    rows_csv = out_prefix.with_name(out_prefix.name + "_rows.csv")
    rows_df.to_csv(rows_csv, index=False)

    summary_rows: list[dict[str, object]] = []
    for (family, regime, label), sub in rows_df.groupby(["family", "regime", "label"], dropna=False):
        rw_m, rw_ci = _ci(sub["early_rw"].to_numpy())
        gap_m, gap_ci = _ci(sub["early_gap"].to_numpy())
        tail_m, tail_ci = _ci(sub["tail_delta_vs_erm"].to_numpy())
        proxy_m, proxy_ci = _ci(sub["proxy_delta_vs_erm"].to_numpy())
        summary_rows.append(
            {
                "family": family,
                "regime": regime,
                "label": label,
                "n": int(sub.shape[0]),
                "early_rw_mean": rw_m,
                "early_rw_ci": rw_ci,
                "early_gap_mean": gap_m,
                "early_gap_ci": gap_ci,
                "tail_delta_mean": tail_m,
                "tail_delta_ci": tail_ci,
                "proxy_delta_mean": proxy_m,
                "proxy_delta_ci": proxy_ci,
                "sign_accuracy": float(sub["sign_correct"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["family", "label"]).reset_index(drop=True)
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    eval_rows = []
    rows_eval = rows_df[rows_df["family"] != "other"].copy()
    summary_eval = summary_df[summary_df["family"] != "other"].rename(columns={"tail_delta_mean": "tail_delta", "early_rw_mean": "early_rw"}).copy()
    for scope, sub in [("seed", rows_eval), ("regime", summary_eval)]:
        x = sub["early_rw"].to_numpy(dtype=float)
        y = (sub["tail_delta_vs_erm"] if scope == "seed" else sub["tail_delta"]).to_numpy(dtype=float)
        acc = float(np.mean((x < 1.0) == (y > 0.0)))
        eval_rows.append(
            {
                "scope": scope,
                "n": int(len(sub)),
                "sign_accuracy_rw_lt1": acc,
                "pearson_corr": float(np.corrcoef(x, y)[0, 1]) if len(sub) > 1 else np.nan,
                "spearman_corr": _rank_corr(x, y) if len(sub) > 1 else np.nan,
            }
        )
    eval_df = pd.DataFrame(eval_rows)
    eval_csv = out_prefix.with_name(out_prefix.name + "_eval.csv")
    eval_df.to_csv(eval_csv, index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    colors = {"softclip": "#b35a1f", "labelsmooth": "#2f7d4a", "focal": "#2864c7"}
    markers = {"softclip": "o", "labelsmooth": "s", "focal": "^"}
    plot_df = summary_df[summary_df["family"] != "other"].copy()
    for _, row in plot_df.iterrows():
        fam = str(row["family"])
        ax.errorbar(
            row["early_rw_mean"],
            row["tail_delta_mean"],
            xerr=row["early_rw_ci"],
            yerr=row["tail_delta_ci"],
            fmt=markers.get(fam, "o"),
            ms=7,
            lw=1.4,
            elinewidth=1.2,
            capsize=2.5,
            color=colors.get(fam, "black"),
            mec="black",
            mew=0.5,
        )
        ax.annotate(
            str(row["label"]),
            (row["early_rw_mean"], row["tail_delta_mean"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color=colors.get(fam, "black"),
        )
    ax.axvline(1.0, color="gray", ls="--", lw=1.2)
    ax.axhline(0.0, color="gray", ls="--", lw=1.2)
    ax.set_xlabel(r"Early $R_w$ (epochs %d-%d)" % (int(args.early_lo), int(args.early_hi)))
    ax.set_ylabel(r"Selected $\Delta$Tail vs ERM")
    ax.set_title(r"Early $R_w$ predicts eventual tail direction on Camelyon17")
    fig.tight_layout()
    png_path = out_prefix.with_name(out_prefix.name + ".png")
    pdf_path = out_prefix.with_name(out_prefix.name + ".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote", rows_csv)
    print("[ok] wrote", summary_csv)
    print("[ok] wrote", eval_csv)
    print("[ok] wrote", png_path)
    print("[ok] wrote", pdf_path)


if __name__ == "__main__":
    main()
