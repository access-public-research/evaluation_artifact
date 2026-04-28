import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


@dataclass
class Case:
    dataset: str
    regime: str
    tag_filter: str
    embeds_dir: Path
    teacher_bins_path: Path
    label: str


def _discover_run_dirs(runs_root: Path, dataset: str, regime: str, tag_filter: str) -> list[tuple[int, Path]]:
    reg_dir = runs_root / dataset / regime
    out: list[tuple[int, Path]] = []
    if not reg_dir.exists():
        raise FileNotFoundError(f"Missing regime dir: {reg_dir}")
    for seed_dir in sorted(reg_dir.glob("seed*")):
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except Exception:
            continue
        cands = [d for d in seed_dir.iterdir() if d.is_dir() and tag_filter in d.name]
        if not cands:
            continue
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        out.append((seed, cands[0]))
    if not out:
        raise FileNotFoundError(f"No matching runs for {dataset}/{regime} with tag '{tag_filter}'")
    return out


def _bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.logaddexp(0.0, z) - yy * z


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(ci95_mean(vals))


def _rw_from_logits(logits: np.ndarray, y: np.ndarray, tail_mask: np.ndarray, clip_loss: float, clip_alpha: float) -> tuple[float, float, float]:
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    ce_grad = np.abs(probs - np.asarray(y, dtype=np.float64))
    ce_loss = _bce_from_logits(logits, y)
    weights = np.where(ce_loss <= float(clip_loss), 1.0, float(clip_alpha))
    obj_grad = ce_grad * weights
    core_mask = ~tail_mask
    eps = 1e-8
    mean_ce_tail = float(np.mean(ce_grad[tail_mask]))
    mean_ce_core = float(np.mean(ce_grad[core_mask]))
    mean_obj_tail = float(np.mean(obj_grad[tail_mask]))
    mean_obj_core = float(np.mean(obj_grad[core_mask]))
    rw = (mean_obj_tail / max(mean_ce_tail, eps)) / max((mean_obj_core / max(mean_ce_core, eps)), eps)
    frac_tail_clipped = float(np.mean(ce_loss[tail_mask] > float(clip_loss)))
    frac_core_clipped = float(np.mean(ce_loss[core_mask] > float(clip_loss)))
    return float(rw), frac_tail_clipped, frac_core_clipped


def _build_cases(repo_root: Path) -> list[Case]:
    return [
        Case(
            dataset="celeba",
            regime="erm_softclip_p95_a10",
            tag_filter="v25ermsoftclip_celeba_10s",
            embeds_dir=repo_root / "artifacts" / "embeds" / "celeba_resnet50",
            teacher_bins_path=repo_root / "artifacts" / "partitions_eval" / "celeba_resnet50" / "v5_confmlp_k64" / "teacher_difficulty" / "bankA" / "val_skew" / "diff_m00_K64.npy",
            label="CelebA ERM P95",
        ),
        Case(
            dataset="camelyon17",
            regime="erm_softclip_p95_a10_cam",
            tag_filter="v11ermsoftclipfix_cam_10s",
            embeds_dir=repo_root / "artifacts" / "embeds" / "camelyon17_resnet50",
            teacher_bins_path=repo_root / "artifacts" / "partitions_eval" / "camelyon17_resnet50" / "teacher_difficulty" / "bankA" / "val_skew" / "diff_m00_K64.npy",
            label="Camelyon17 ERM P95",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default="replication_rcg")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/rw_persistence_timecourse_20260327")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    runs_root = repo_root / "runs"
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case in _build_cases(repo_root):
        y_val = np.load(case.embeds_dir / "y_val_skew.npy").astype(np.int64)
        bins = np.load(case.teacher_bins_path)
        k = int(np.nanmax(bins)) + 1
        tail_start = int(np.floor(0.9 * k))
        tail_mask = bins >= tail_start
        run_dirs = _discover_run_dirs(runs_root, case.dataset, case.regime, case.tag_filter)

        for seed, run_dir in run_dirs:
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            clip_loss = float(cfg.get("clip_loss", cfg.get("training", {}).get("clip_loss", 0.0)))
            clip_alpha = float(cfg.get("clip_alpha", cfg.get("training", {}).get("clip_alpha", 1.0)))
            logits_all = np.load(run_dir / "val_logits_by_epoch.npy", mmap_mode="r")
            num_epochs = int(logits_all.shape[0])
            for epoch in range(1, num_epochs + 1):
                logits = np.asarray(logits_all[epoch - 1], dtype=np.float32)
                rw, frac_tail, frac_core = _rw_from_logits(logits, y_val, tail_mask, clip_loss, clip_alpha)
                rows.append(
                    {
                        "dataset": case.dataset,
                        "label": case.label,
                        "regime": case.regime,
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "R_w": rw,
                        "frac_tail_clipped": frac_tail,
                        "frac_core_clipped": frac_core,
                        "clip_loss": clip_loss,
                        "clip_alpha": clip_alpha,
                    }
                )

    rows_df = pd.DataFrame(rows).sort_values(["dataset", "seed", "epoch"])
    rows_csv = out_prefix.with_name(out_prefix.name + "_rows.csv")
    rows_df.to_csv(rows_csv, index=False)

    summary_rows: list[dict[str, object]] = []
    for (dataset, label, epoch), sub in rows_df.groupby(["dataset", "label", "epoch"], dropna=False):
        rw_m, rw_ci = _mean_ci(sub["R_w"].to_numpy())
        ft_m, ft_ci = _mean_ci(sub["frac_tail_clipped"].to_numpy())
        fc_m, fc_ci = _mean_ci(sub["frac_core_clipped"].to_numpy())
        summary_rows.append(
            {
                "dataset": dataset,
                "label": label,
                "epoch": int(epoch),
                "R_w_mean": rw_m,
                "R_w_ci": rw_ci,
                "frac_tail_clipped_mean": ft_m,
                "frac_tail_clipped_ci": ft_ci,
                "frac_core_clipped_mean": fc_m,
                "frac_core_clipped_ci": fc_ci,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "epoch"])
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    window_specs = [("early", 1, 5), ("late", 25, 30)]
    window_rows: list[dict[str, object]] = []
    for (dataset, label, seed), sub in rows_df.groupby(["dataset", "label", "seed"], dropna=False):
        rec: dict[str, object] = {"dataset": dataset, "label": label, "seed": int(seed)}
        for name, lo, hi in window_specs:
            win = sub[(sub["epoch"] >= lo) & (sub["epoch"] <= hi)]
            rec[f"{name}_R_w"] = float(np.mean(win["R_w"])) if not win.empty else np.nan
        rec["delta_late_minus_early"] = float(rec["late_R_w"] - rec["early_R_w"]) if np.isfinite(rec["late_R_w"]) and np.isfinite(rec["early_R_w"]) else np.nan
        window_rows.append(rec)
    windows_df = pd.DataFrame(window_rows)

    window_summary_rows: list[dict[str, object]] = []
    for (dataset, label), sub in windows_df.groupby(["dataset", "label"], dropna=False):
        early_m, early_ci = _mean_ci(sub["early_R_w"].to_numpy())
        late_m, late_ci = _mean_ci(sub["late_R_w"].to_numpy())
        delta_m, delta_ci = _mean_ci(sub["delta_late_minus_early"].to_numpy())
        window_summary_rows.append(
            {
                "dataset": dataset,
                "label": label,
                "n": int(sub.shape[0]),
                "early_R_w_mean": early_m,
                "early_R_w_ci": early_ci,
                "late_R_w_mean": late_m,
                "late_R_w_ci": late_ci,
                "delta_late_minus_early_mean": delta_m,
                "delta_late_minus_early_ci": delta_ci,
            }
        )
    window_summary_df = pd.DataFrame(window_summary_rows).sort_values("dataset")
    window_csv = out_prefix.with_name(out_prefix.name + "_window_summary.csv")
    window_summary_df.to_csv(window_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180, sharex=True)
    colors = {"CelebA ERM P95": "#c06c2b", "Camelyon17 ERM P95": "#0f6cbd"}

    for dataset_label in summary_df["label"].drop_duplicates():
        sub = summary_df[summary_df["label"] == dataset_label].sort_values("epoch")
        color = colors.get(dataset_label, None)
        axes[0].plot(sub["epoch"], sub["R_w_mean"], label=dataset_label, color=color, lw=2)
        axes[0].fill_between(
            sub["epoch"].to_numpy(),
            (sub["R_w_mean"] - sub["R_w_ci"]).to_numpy(),
            (sub["R_w_mean"] + sub["R_w_ci"]).to_numpy(),
            color=color,
            alpha=0.18,
        )
        axes[1].plot(
            sub["epoch"],
            sub["frac_tail_clipped_mean"] - sub["frac_core_clipped_mean"],
            label=dataset_label,
            color=color,
            lw=2,
        )

    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set_title(r"Tail/Core Orientation Over Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$R_w$")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="gray", ls="--", lw=1)
    axes[1].set_title("Tail-Core Clip Exposure Gap")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Frac clipped (tail - core)")

    fig.tight_layout()
    png_path = out_prefix.with_name(out_prefix.name + ".png")
    pdf_path = out_prefix.with_name(out_prefix.name + ".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote", rows_csv)
    print("[ok] wrote", summary_csv)
    print("[ok] wrote", window_csv)
    print("[ok] wrote", png_path)
    print("[ok] wrote", pdf_path)


if __name__ == "__main__":
    main()
