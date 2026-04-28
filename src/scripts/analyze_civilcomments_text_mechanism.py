import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.stats import ci95_mean
from .evaluate_civilcomments_text_e2e import CIVILCOMMENTS_IDENTITY_FIELDS
from .finetune_civilcomments_text import DistilBertBinaryClassifier, build_text_transform


def _mean_ci(values: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(ci95_mean(vals))


def _run_dir(cfg: dict, dataset_name: str, regime_name: str, seed: int) -> Path:
    runs_root = Path(cfg["project"]["runs_dir"])
    tag = str(cfg["training"]["tag_suffix"])
    return runs_root / f"{dataset_name}_e2e" / regime_name / f"seed{int(seed)}" / tag


def _load_ckpt_epoch(run_dir: Path, epoch: int) -> Dict:
    ckpt_path = run_dir / f"ckpt_epoch{int(epoch):03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def _bce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.logaddexp(0.0, z) - yy * z


def _ece_binary(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    conf = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(np.int64)
    correct = (preds == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, int(num_bins) + 1)
    ece = 0.0
    n = max(1, conf.shape[0])
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf <= hi) if hi >= 1.0 else ((conf >= lo) & (conf < hi))
        if not np.any(mask):
            continue
        ece += float(mask.sum() / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _resolve_val_surface_dir(cfg: dict, dataset_name: str, backbone_name: str) -> Path:
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    skew_cfg = cfg.get("validation", {}).get("skewed_val", {})
    override_name = str(skew_cfg.get("artifact_name_override", "")).strip()
    if override_name:
        return artifacts_dir / "embeds" / override_name
    return artifacts_dir / "embeds" / f"{dataset_name}_{backbone_name}"


def _epoch_mechanism_rows(cfg: dict, dataset_name: str, distorted_regime: str) -> pd.DataFrame:
    backbone_name = str(cfg["embeddings"]["backbone"])
    feat_dir = _resolve_val_surface_dir(cfg, dataset_name, backbone_name)
    y_val = np.load(feat_dir / "y_val_skew.npy").astype(np.int64)
    g_val = np.load(feat_dir / "g_val_skew.npy").astype(np.int64)
    tail_mask = g_val == int(np.max(g_val))
    if not np.any(tail_mask):
        raise ValueError("Resolved tail mask is empty.")

    rows: List[Dict[str, float]] = []
    for seed in [int(s) for s in cfg["training"]["seeds"]]:
        run_dir = _run_dir(cfg, dataset_name, distorted_regime, int(seed))
        if not (run_dir / "done.json").exists():
            continue
        run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        clip_loss = float(run_cfg.get("clip_loss", 0.0) or 0.0)
        clip_alpha = float(run_cfg.get("clip_alpha", 1.0) or 1.0)
        logits_all = np.load(run_dir / "val_logits_by_epoch.npy")
        standard_all = np.load(run_dir / "val_standard_loss_by_epoch.npy")
        proxy_all = np.load(run_dir / "val_proxy_loss_by_epoch.npy")
        correct_all = np.load(run_dir / "val_correct_by_epoch.npy")
        for epoch in range(1, int(logits_all.shape[0]) + 1):
            logits = np.asarray(logits_all[epoch - 1], dtype=np.float64)
            probs = 1.0 / (1.0 + np.exp(-logits))
            ce_grad = np.abs(probs - y_val.astype(np.float64))
            ce_loss = _bce_from_logits(logits, y_val)
            weights = np.where(ce_loss <= clip_loss, 1.0, clip_alpha)
            obj_grad = ce_grad * weights
            core_mask = ~tail_mask
            mt_ce = float(np.mean(ce_grad[tail_mask]))
            mc_ce = float(np.mean(ce_grad[core_mask]))
            mt_obj = float(np.mean(obj_grad[tail_mask]))
            mc_obj = float(np.mean(obj_grad[core_mask]))
            rw = (mt_obj / max(mt_ce, 1e-8)) / max((mc_obj / max(mc_ce, 1e-8)), 1e-8)
            rows.append(
                {
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "R_w": float(rw),
                    "frac_tail_clipped": float(np.mean(ce_loss[tail_mask] > clip_loss)),
                    "frac_core_clipped": float(np.mean(ce_loss[core_mask] > clip_loss)),
                    "tail_core_clip_gap": float(np.mean(ce_loss[tail_mask] > clip_loss) - np.mean(ce_loss[core_mask] > clip_loss)),
                    "val_proxy_loss": float(np.mean(proxy_all[epoch - 1])),
                    "val_standard_loss": float(np.mean(standard_all[epoch - 1])),
                    "val_acc": float(np.mean(correct_all[epoch - 1])),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "epoch"]).reset_index(drop=True)


@torch.no_grad()
def _eval_checkpoint_groupwise(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    num_bins: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    logits_all = []
    loss_all = []
    corr_all = []
    y_all = []
    meta_all = []
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")):
            logits = model(xb)
            losses = bce(logits, yb.float())
        preds = (logits >= 0).long()
        corr = (preds == yb).long()
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        loss_all.append(losses.detach().cpu().numpy().astype(np.float32))
        corr_all.append(corr.detach().cpu().numpy().astype(np.uint8))
        y_all.append(yb.detach().cpu().numpy().astype(np.int64))
        meta_all.append(mb.detach().cpu().numpy().astype(np.int64))

    logits = np.concatenate(logits_all)
    losses = np.concatenate(loss_all)
    corr = np.concatenate(corr_all)
    y = np.concatenate(y_all)
    meta = np.concatenate(meta_all)
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))

    out_rows: List[Dict[str, float]] = []
    for identity_idx, identity in enumerate(CIVILCOMMENTS_IDENTITY_FIELDS):
        for label in (0, 1):
            mask = (meta[:, identity_idx] == 1) & (y == label)
            if not np.any(mask):
                continue
            out_rows.append(
                {
                    "group": f"{identity}={label}",
                    "n": int(mask.sum()),
                    "group_acc": float(np.mean(corr[mask])),
                    "group_loss": float(np.mean(losses[mask])),
                    "group_brier": float(np.mean(np.square(probs[mask] - y[mask].astype(np.float64)))),
                    "group_ece": _ece_binary(probs[mask], y[mask], num_bins=num_bins),
                }
            )

    overall = {
        "test_overall_acc": float(np.mean(corr)),
        "test_overall_loss": float(np.mean(losses)),
        "test_brier": float(np.mean(np.square(probs - y.astype(np.float64)))),
        "test_ece": _ece_binary(probs, y, num_bins=num_bins),
    }
    return pd.DataFrame(out_rows), overall


def _groupwise_selector_rows(
    cfg: dict,
    dataset_name: str,
    rows_csv: Path,
    selectors: List[str],
    num_bins: int,
) -> pd.DataFrame:
    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    use_amp = bool(cfg["compute"].get("amp", True))

    transform = build_text_transform(str(cfg["embeddings"]["backbone"]), int(cfg["dataset"].get("max_token_length", 300)))
    ds = load_wilds_dataset(
        cfg["dataset"]["wilds_dataset"],
        cfg["dataset"].get("data_dir") or cfg["paths"]["wilds_data_dir"],
        download=False,
    )
    test_subset = ds.get_subset("test", frac=1.0, transform=transform)
    loader = DataLoader(
        test_subset,
        batch_size=int(cfg["finetune"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["compute"].get("num_workers", 0)),
        pin_memory=True,
    )

    rows_df = pd.read_csv(rows_csv)
    rows_df = rows_df[rows_df["selector"].isin(selectors)].copy()
    eval_cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    out: List[Dict[str, object]] = []
    for rec in rows_df.to_dict("records"):
        key = (str(rec["selected_regime"]), int(rec["seed"]), int(rec["selected_epoch"]))
        if key not in eval_cache:
            run_dir = _run_dir(cfg, dataset_name, key[0], key[1])
            ckpt = _load_ckpt_epoch(run_dir, key[2])
            model = DistilBertBinaryClassifier(str(cfg["embeddings"]["backbone"]))
            model.load_state_dict(ckpt["model_state"], strict=True)
            model = model.to(device)
            groups_df, overall = _eval_checkpoint_groupwise(model, loader, device=device, use_amp=use_amp, num_bins=num_bins)
            groups_df["selected_regime"] = key[0]
            groups_df["seed"] = int(key[1])
            groups_df["selected_epoch"] = int(key[2])
            for k, v in overall.items():
                groups_df[k] = float(v)
            eval_cache[key] = groups_df
        sub = eval_cache[key].copy()
        sub["selector"] = str(rec["selector"])
        sub["fallback_to_baseline"] = int(rec.get("fallback_to_baseline", 0))
        out.extend(sub.to_dict("records"))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--distorted_regime", default="erm_softclip_p95_a10_cc")
    ap.add_argument("--rows_csv", required=True)
    ap.add_argument("--selectors", default="baseline,proxy_only,val_loss_only,guardrail_1.25")
    ap.add_argument("--num_bins", type=int, default=15)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = str(cfg["dataset"]["name"])
    selectors = [s.strip() for s in str(args.selectors).split(",") if s.strip()]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    mech_df = _epoch_mechanism_rows(cfg, dataset_name, str(args.distorted_regime))
    mech_rows_csv = out_prefix.with_name(out_prefix.name + "_epoch_rows.csv")
    mech_df.to_csv(mech_rows_csv, index=False)

    epoch_summary = []
    for epoch, sub in mech_df.groupby("epoch", dropna=False):
        rw_m, rw_ci = _mean_ci(sub["R_w"].to_numpy())
        gap_m, gap_ci = _mean_ci(sub["tail_core_clip_gap"].to_numpy())
        tp_m, tp_ci = _mean_ci(sub["frac_tail_clipped"].to_numpy())
        cp_m, cp_ci = _mean_ci(sub["frac_core_clipped"].to_numpy())
        epoch_summary.append(
            {
                "epoch": int(epoch),
                "R_w_mean": rw_m,
                "R_w_ci": rw_ci,
                "tail_core_clip_gap_mean": gap_m,
                "tail_core_clip_gap_ci": gap_ci,
                "frac_tail_clipped_mean": tp_m,
                "frac_tail_clipped_ci": tp_ci,
                "frac_core_clipped_mean": cp_m,
                "frac_core_clipped_ci": cp_ci,
            }
        )
    epoch_summary_df = pd.DataFrame(epoch_summary).sort_values("epoch")
    mech_summary_csv = out_prefix.with_name(out_prefix.name + "_epoch_summary.csv")
    epoch_summary_df.to_csv(mech_summary_csv, index=False)

    group_df = _groupwise_selector_rows(cfg, dataset_name, Path(args.rows_csv), selectors, int(args.num_bins))
    group_rows_csv = out_prefix.with_name(out_prefix.name + "_group_rows.csv")
    group_df.to_csv(group_rows_csv, index=False)

    base_df = group_df[group_df["selector"] == "baseline"][["seed", "group", "group_acc", "group_loss", "group_brier", "group_ece"]].copy()
    base_df = base_df.rename(
        columns={
            "group_acc": "baseline_group_acc",
            "group_loss": "baseline_group_loss",
            "group_brier": "baseline_group_brier",
            "group_ece": "baseline_group_ece",
        }
    )
    merged = group_df.merge(base_df, on=["seed", "group"], how="left")
    merged["group_acc_delta_vs_baseline"] = merged["group_acc"] - merged["baseline_group_acc"]
    merged["group_loss_delta_vs_baseline"] = merged["group_loss"] - merged["baseline_group_loss"]
    merged["group_brier_delta_vs_baseline"] = merged["group_brier"] - merged["baseline_group_brier"]
    merged["group_ece_delta_vs_baseline"] = merged["group_ece"] - merged["baseline_group_ece"]

    summary_rows = []
    for selector, sub in merged.groupby("selector", dropna=False):
        count_loss = []
        count_acc = []
        for seed, seed_sub in sub.groupby("seed", dropna=False):
            count_loss.append(int(np.sum(seed_sub["group_loss_delta_vs_baseline"].to_numpy() > 0)))
            count_acc.append(int(np.sum(seed_sub["group_acc_delta_vs_baseline"].to_numpy() < 0)))
        loss_mean, loss_ci = _mean_ci(sub["group_loss_delta_vs_baseline"].to_numpy())
        acc_mean, acc_ci = _mean_ci(sub["group_acc_delta_vs_baseline"].to_numpy())
        brier_mean, brier_ci = _mean_ci(sub["group_brier_delta_vs_baseline"].to_numpy())
        ece_mean, ece_ci = _mean_ci(sub["group_ece_delta_vs_baseline"].to_numpy())
        count_loss_mean, count_loss_ci = _mean_ci(np.asarray(count_loss, dtype=np.float64))
        count_acc_mean, count_acc_ci = _mean_ci(np.asarray(count_acc, dtype=np.float64))
        summary_rows.append(
            {
                "selector": selector,
                "n_seeds": int(sub["seed"].nunique()),
                "groups_per_seed": int(sub["group"].nunique()),
                "mean_group_loss_delta_vs_baseline": loss_mean,
                "ci_group_loss_delta_vs_baseline": loss_ci,
                "mean_group_acc_delta_vs_baseline": acc_mean,
                "ci_group_acc_delta_vs_baseline": acc_ci,
                "mean_group_brier_delta_vs_baseline": brier_mean,
                "ci_group_brier_delta_vs_baseline": brier_ci,
                "mean_group_ece_delta_vs_baseline": ece_mean,
                "ci_group_ece_delta_vs_baseline": ece_ci,
                "mean_num_groups_worse_loss_vs_baseline": count_loss_mean,
                "ci_num_groups_worse_loss_vs_baseline": count_loss_ci,
                "mean_num_groups_worse_acc_vs_baseline": count_acc_mean,
                "ci_num_groups_worse_acc_vs_baseline": count_acc_ci,
            }
        )
    selector_summary_df = pd.DataFrame(summary_rows).sort_values("selector")
    selector_summary_csv = out_prefix.with_name(out_prefix.name + "_selector_group_summary.csv")
    selector_summary_df.to_csv(selector_summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=180)
    axes[0].plot(epoch_summary_df["epoch"], epoch_summary_df["R_w_mean"], color="#0f6cbd", lw=2)
    axes[0].fill_between(
        epoch_summary_df["epoch"].to_numpy(),
        (epoch_summary_df["R_w_mean"] - epoch_summary_df["R_w_ci"]).to_numpy(),
        (epoch_summary_df["R_w_mean"] + epoch_summary_df["R_w_ci"]).to_numpy(),
        color="#0f6cbd",
        alpha=0.18,
    )
    axes[0].axhline(1.0, color="gray", lw=1, ls="--")
    axes[0].set_title(r"Text $R_w$ Over Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$R_w$")

    axes[1].plot(epoch_summary_df["epoch"], epoch_summary_df["tail_core_clip_gap_mean"], color="#a64222", lw=2)
    axes[1].fill_between(
        epoch_summary_df["epoch"].to_numpy(),
        (epoch_summary_df["tail_core_clip_gap_mean"] - epoch_summary_df["tail_core_clip_gap_ci"]).to_numpy(),
        (epoch_summary_df["tail_core_clip_gap_mean"] + epoch_summary_df["tail_core_clip_gap_ci"]).to_numpy(),
        color="#a64222",
        alpha=0.18,
    )
    axes[1].axhline(0.0, color="gray", lw=1, ls="--")
    axes[1].set_title("Tail-Core Clip Exposure Gap")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Frac clipped (tail - core)")
    fig.tight_layout()
    fig_pdf = out_prefix.with_name(out_prefix.name + ".pdf")
    fig_png = out_prefix.with_name(out_prefix.name + ".png")
    fig.savefig(fig_pdf, bbox_inches="tight")
    fig.savefig(fig_png, bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote", mech_rows_csv)
    print("[ok] wrote", mech_summary_csv)
    print("[ok] wrote", group_rows_csv)
    print("[ok] wrote", selector_summary_csv)
    print("[ok] wrote", fig_pdf)
    print("[ok] wrote", fig_png)


if __name__ == "__main__":
    main()
