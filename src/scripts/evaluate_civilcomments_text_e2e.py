import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.stats import ci95_mean
from .finetune_civilcomments_text import DistilBertBinaryClassifier, build_text_transform


CIVILCOMMENTS_IDENTITY_FIELDS = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]


def _mean_ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(ci95_mean(x))


def _run_dir(cfg: dict, dataset_name: str, regime_name: str, seed: int) -> Path:
    runs_root = Path(cfg["project"]["runs_dir"])
    tag = str(cfg["training"]["tag_suffix"])
    return runs_root / f"{dataset_name}_e2e" / regime_name / f"seed{int(seed)}" / tag


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    rows = []
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No metrics rows in {run_dir}")
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def _load_ckpt_epoch(run_dir: Path, epoch: int) -> Dict:
    ckpt_path = run_dir / f"ckpt_epoch{int(epoch):03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


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
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += float(mask.sum() / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _worst_group_metrics(
    correct: np.ndarray,
    losses: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    meta: np.ndarray,
    num_bins: int,
) -> Dict[str, float]:
    worst_acc = 1.0
    worst_loss = -np.inf
    worst_brier = -np.inf
    worst_ece = -np.inf
    valid_groups = 0
    labels = np.asarray(labels, dtype=np.int64)
    meta = np.asarray(meta)
    for identity_idx in range(len(CIVILCOMMENTS_IDENTITY_FIELDS)):
        identity = meta[:, identity_idx].astype(np.int64)
        for label in (0, 1):
            mask = (identity == 1) & (labels == label)
            if mask.sum() == 0:
                continue
            valid_groups += 1
            worst_acc = min(worst_acc, float(np.mean(correct[mask])))
            worst_loss = max(worst_loss, float(np.mean(losses[mask])))
            worst_brier = max(worst_brier, float(np.mean(np.square(probs[mask] - labels[mask].astype(np.float64)))))
            worst_ece = max(worst_ece, _ece_binary(probs[mask], labels[mask], num_bins=num_bins))
    if valid_groups == 0:
        return {
            "test_wilds_wg_acc": np.nan,
            "test_wilds_wg_loss": np.nan,
            "test_wilds_wg_brier": np.nan,
            "test_wilds_wg_ece": np.nan,
            "test_wilds_wg_groups": 0,
        }
    return {
        "test_wilds_wg_acc": float(worst_acc),
        "test_wilds_wg_loss": float(worst_loss),
        "test_wilds_wg_brier": float(worst_brier),
        "test_wilds_wg_ece": float(worst_ece),
        "test_wilds_wg_groups": int(valid_groups),
    }


@torch.no_grad()
def _eval_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    num_bins: int,
) -> Dict[str, float]:
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

    out = {
        "test_overall_acc": float(np.mean(corr)),
        "test_overall_loss": float(np.mean(losses)),
        "test_brier": float(np.mean(np.square(probs - y.astype(np.float64)))),
        "test_ece": _ece_binary(probs, y, num_bins=num_bins),
    }
    out.update(_worst_group_metrics(corr, losses, probs, y, meta, num_bins=num_bins))
    return out


def _select_epoch(metrics: pd.DataFrame, column: str) -> Tuple[int, float, float]:
    idx = int(metrics[column].astype(float).argmin())
    row = metrics.iloc[idx]
    return int(row["epoch"]), float(row["val_proxy_loss"]), float(row["val_standard_loss"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--baseline_regime", default="erm")
    ap.add_argument("--distorted_regime", default="erm_softclip_p95_a10_cc")
    ap.add_argument("--guardrail_rho", type=float, default=1.25)
    ap.add_argument("--fixed_epoch", type=int, default=10)
    ap.add_argument("--num_bins", type=int, default=15)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    if dataset_name != "civilcomments":
        raise ValueError("This evaluator is only for civilcomments.")

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
    eval_batch_size = int(cfg["finetune"]["eval_batch_size"])
    test_loader = DataLoader(
        test_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(cfg["compute"].get("num_workers", 0)),
        pin_memory=True,
    )

    seeds = [int(s) for s in cfg["training"]["seeds"]]
    eval_cache: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        base_dir = _run_dir(cfg, dataset_name, str(args.baseline_regime), int(seed))
        dist_dir = _run_dir(cfg, dataset_name, str(args.distorted_regime), int(seed))
        if not (base_dir / "done.json").exists() or not (dist_dir / "done.json").exists():
            continue

        base_metrics = _load_metrics(base_dir)
        dist_metrics = _load_metrics(dist_dir)
        base_epoch, base_proxy_val, base_std_val = _select_epoch(base_metrics, "val_standard_loss")
        proxy_epoch, proxy_proxy_val, proxy_std_val = _select_epoch(dist_metrics, "val_proxy_loss")
        valloss_epoch, valloss_proxy_val, valloss_std_val = _select_epoch(dist_metrics, "val_standard_loss")

        feasible = dist_metrics[dist_metrics["val_standard_loss"].astype(float) <= float(args.guardrail_rho) * base_std_val].copy()
        if feasible.empty:
            guard_row = {
                "selector": f"guardrail_{args.guardrail_rho:.2f}",
                "selected_regime": str(args.baseline_regime),
                "epoch": int(base_epoch),
                "val_proxy_loss": float(base_proxy_val),
                "val_standard_loss": float(base_std_val),
                "fallback_to_baseline": 1,
            }
        else:
            feasible = feasible.sort_values(["val_proxy_loss", "epoch"]).reset_index(drop=True)
            fr = feasible.iloc[0]
            guard_row = {
                "selector": f"guardrail_{args.guardrail_rho:.2f}",
                "selected_regime": str(args.distorted_regime),
                "epoch": int(fr["epoch"]),
                "val_proxy_loss": float(fr["val_proxy_loss"]),
                "val_standard_loss": float(fr["val_standard_loss"]),
                "fallback_to_baseline": 0,
            }

        fixed_base_epoch = int(min(int(args.fixed_epoch), int(base_metrics["epoch"].max())))
        fixed_dist_epoch = int(min(int(args.fixed_epoch), int(dist_metrics["epoch"].max())))
        selector_rows = [
            {
                "selector": "baseline",
                "selected_regime": str(args.baseline_regime),
                "epoch": int(base_epoch),
                "val_proxy_loss": float(base_proxy_val),
                "val_standard_loss": float(base_std_val),
                "fallback_to_baseline": 0,
            },
            {
                "selector": "proxy_only",
                "selected_regime": str(args.distorted_regime),
                "epoch": int(proxy_epoch),
                "val_proxy_loss": float(proxy_proxy_val),
                "val_standard_loss": float(proxy_std_val),
                "fallback_to_baseline": 0,
            },
            {
                "selector": "val_loss_only",
                "selected_regime": str(args.distorted_regime),
                "epoch": int(valloss_epoch),
                "val_proxy_loss": float(valloss_proxy_val),
                "val_standard_loss": float(valloss_std_val),
                "fallback_to_baseline": 0,
            },
            guard_row,
            {
                "selector": f"fixed{int(args.fixed_epoch)}_baseline",
                "selected_regime": str(args.baseline_regime),
                "epoch": int(fixed_base_epoch),
                "val_proxy_loss": float(base_metrics.loc[base_metrics["epoch"] == fixed_base_epoch, "val_proxy_loss"].iloc[0]),
                "val_standard_loss": float(base_metrics.loc[base_metrics["epoch"] == fixed_base_epoch, "val_standard_loss"].iloc[0]),
                "fallback_to_baseline": 0,
            },
            {
                "selector": f"fixed{int(args.fixed_epoch)}_softclip",
                "selected_regime": str(args.distorted_regime),
                "epoch": int(fixed_dist_epoch),
                "val_proxy_loss": float(dist_metrics.loc[dist_metrics["epoch"] == fixed_dist_epoch, "val_proxy_loss"].iloc[0]),
                "val_standard_loss": float(dist_metrics.loc[dist_metrics["epoch"] == fixed_dist_epoch, "val_standard_loss"].iloc[0]),
                "fallback_to_baseline": 0,
            },
        ]

        for sel in selector_rows:
            sel_regime = str(sel["selected_regime"])
            epoch = int(sel["epoch"])
            cache_key = (sel_regime, int(seed), epoch)
            if cache_key not in eval_cache:
                run_dir = _run_dir(cfg, dataset_name, sel_regime, int(seed))
                ckpt = _load_ckpt_epoch(run_dir, epoch)
                model = DistilBertBinaryClassifier(str(cfg["embeddings"]["backbone"])).to(device)
                model.load_state_dict(ckpt["model_state"])
                eval_cache[cache_key] = _eval_checkpoint(model, test_loader, device=device, use_amp=use_amp, num_bins=int(args.num_bins))
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            row = {
                "seed": int(seed),
                "selector": str(sel["selector"]),
                "selected_regime": sel_regime,
                "selected_epoch": epoch,
                "fallback_to_baseline": int(sel["fallback_to_baseline"]),
                "val_proxy_loss": float(sel["val_proxy_loss"]),
                "val_standard_loss": float(sel["val_standard_loss"]),
            }
            row.update(eval_cache[cache_key])
            rows.append(row)

    df = pd.DataFrame(rows)
    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_rows, index=False)

    summary_rows: List[Dict[str, object]] = []
    if not df.empty:
        baseline_df = df[df["selector"] == "baseline"].set_index("seed")
        for selector, sub in df.groupby("selector", dropna=False):
            rec: Dict[str, object] = {
                "selector": selector,
                "n": int(sub.shape[0]),
            }
            for col in [
                "test_overall_acc",
                "test_overall_loss",
                "test_wilds_wg_acc",
                "test_wilds_wg_loss",
                "test_ece",
                "test_wilds_wg_ece",
                "test_brier",
                "test_wilds_wg_brier",
            ]:
                mean, ci = _mean_ci(sub[col].to_numpy())
                rec[f"{col}_mean"] = mean
                rec[f"{col}_ci"] = ci
                if selector != "baseline":
                    matched = []
                    for _, rr in sub.iterrows():
                        seed = int(rr["seed"])
                        if seed in baseline_df.index:
                            matched.append(float(rr[col]) - float(baseline_df.loc[seed, col]))
                    if matched:
                        dm, dci = _mean_ci(np.asarray(matched, dtype=np.float64))
                        rec[f"{col}_delta_vs_baseline_mean"] = dm
                        rec[f"{col}_delta_vs_baseline_ci"] = dci
            summary_rows.append(rec)

    out_summary = Path(args.out_summary)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
