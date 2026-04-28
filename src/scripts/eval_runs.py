import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import load_config
from ..metrics.group_eval import group_accuracy_from_logits
from ..metrics.proxy_eval import aggregate_proxy_metrics, snr_between_total_multi
from ..utils.io import ensure_dir


@dataclass
class RunInfo:
    dataset: str
    regime: str
    seed: int
    tag: str
    run_dir: Path


def discover_runs(runs_root: Path, dataset: str, regimes: Iterable[str]) -> List[RunInfo]:
    out: List[RunInfo] = []
    for regime in regimes:
        regime_dir = runs_root / dataset / regime
        if not regime_dir.exists():
            continue
        for seed_dir in sorted(regime_dir.glob("seed*")):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except Exception:
                continue
            for tag_dir in sorted(seed_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                cfg_path = tag_dir / "config.json"
                if not cfg_path.exists():
                    continue
                out.append(
                    RunInfo(
                        dataset=dataset,
                        regime=regime,
                        seed=seed,
                        tag=tag_dir.name,
                        run_dir=tag_dir,
                    )
                )
    return out


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
    X = np.asarray(X, dtype=np.float32)
    logits_all = []
    use_amp = device.startswith("cuda")
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb).squeeze(1)
        else:
            logits = model(xb).squeeze(1)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits_all, axis=0)


def _load_partitions(split_dir: Path, prefix: str, num_parts: int, K: int) -> List[np.ndarray]:
    parts: List[np.ndarray] = []
    for m in range(int(num_parts)):
        p = split_dir / f"{prefix}_m{m:02d}_K{int(K)}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing partition file: {p}")
        parts.append(np.load(p))
    return parts


def _selection_summary(df_run: pd.DataFrame, tail_lambda: float, hybrid_topk_frac: float, snr_threshold: float):
    # Expect one row per epoch.
    overall_acc = df_run["val_overall_acc"].to_numpy()
    overall_loss = df_run["val_overall_loss"].to_numpy()
    proxy_acc = df_run["val_proxy_worst_acc"].to_numpy()
    proxy_loss = df_run["val_proxy_worst_loss"].to_numpy()
    snr = df_run["val_snr_proxy_correct"].to_numpy()

    def argmax(x):
        return int(np.nanargmax(x))

    def argmin(x):
        return int(np.nanargmin(x))

    E = int(len(df_run))
    k = max(1, int(np.ceil(float(hybrid_topk_frac) * E)))
    topk = np.argsort(overall_acc)[-k:]
    hybrid_idx = int(topk[np.argmax(proxy_acc[topk])])

    tail_score = overall_acc - float(tail_lambda) * proxy_loss

    overall_idx = argmax(overall_acc)
    loss_idx = argmin(overall_loss)
    proxy_idx = argmax(proxy_acc)
    proxy_loss_idx = argmin(proxy_loss)
    tail_idx = argmax(tail_score)

    # Router gate: trust aggressive proxy critic only when SNR is high at that candidate.
    router_idx = proxy_idx if snr[proxy_idx] >= float(snr_threshold) else tail_idx

    methods = {
        "overall": overall_idx,
        "loss": loss_idx,
        "proxy_acc": proxy_idx,
        "proxy_loss": proxy_loss_idx,
        "tailmoderated": tail_idx,
        "hybrid": hybrid_idx,
        "router": router_idx,
    }

    oracle_wg = float(df_run["test_worst_group_acc"].max())
    oracle_overall = float(df_run["test_overall_acc"].max())

    rows = []
    for method, idx in methods.items():
        row = df_run.iloc[int(idx)]
        rows.append(
            {
                "method": method,
                "selected_epoch": int(row["epoch"]),
                "test_worst_group_acc": float(row["test_worst_group_acc"]),
                "test_overall_acc": float(row["test_overall_acc"]),
                "val_overall_acc": float(row["val_overall_acc"]),
                "val_proxy_worst_acc": float(row["val_proxy_worst_acc"]),
                "val_snr_proxy_correct": float(row["val_snr_proxy_correct"]),
                "oracle_best_wg_acc": oracle_wg,
                "oracle_best_overall_acc": oracle_overall,
            }
        )
    return rows


def eval_run(cfg, run: RunInfo, overwrite: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    backbone = cfg["embeddings"]["backbone"]
    feat_dir = artifacts_dir / "embeds" / f"{run.dataset}_{backbone}"

    run_metrics_path = run.run_dir / "metrics_by_epoch.csv"
    sel_path = run.run_dir / "selection_summary.csv"
    if run_metrics_path.exists() and sel_path.exists() and not overwrite:
        return pd.read_csv(run_metrics_path), pd.read_csv(sel_path)

    run_cfg = json.loads((run.run_dir / "config.json").read_text())
    hidden_dim = int(run_cfg.get("training", {}).get("hidden_dim", 0))
    dropout = float(run_cfg.get("training", {}).get("dropout", 0.0))
    d_in = int(run_cfg.get("d_in"))

    part_base = artifacts_dir / "partitions" / f"{run.dataset}_{backbone}"
    part_root_cfg = run_cfg.get("partition_root")
    part_version = cfg.get("partitions", {}).get("version")
    part_version_dir = part_base / str(part_version) if part_version else None
    if part_root_cfg:
        part_root = Path(part_root_cfg)
    elif part_base.exists():
        part_root = part_base
    elif part_version_dir and part_version_dir.exists():
        part_root = part_version_dir
    else:
        raise FileNotFoundError(f"Could not resolve partition root for {run.run_dir}")
    part_version_run = run_cfg.get("partition_version") or part_version

    # Val arrays (already cached during training).
    val_loss = np.load(run.run_dir / "val_loss_by_epoch.npy")  # (E, N)
    val_correct = np.load(run.run_dir / "val_correct_by_epoch.npy")  # (E, N)
    E, N = val_loss.shape

    # Labels and groups.
    y_val = np.load(feat_dir / "y_val_skew.npy")
    a_val = np.load(feat_dir / "a_val_skew.npy")
    g_val = np.load(feat_dir / "g_val_skew.npy")
    if int(y_val.shape[0]) != int(N):
        raise ValueError(f"Val size mismatch for {run.run_dir}: val arrays={N} labels={y_val.shape[0]}")

    # Test arrays.
    X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(feat_dir / "y_test.npy")
    g_test = np.load(feat_dir / "g_test.npy")

    # Partitions on val_skew.
    proxy_cfg = cfg["partitions"]["proxy"]
    dec_cfg = cfg["partitions"]["decoupled"]
    proxy_family = str(proxy_cfg.get("family", "random_hash"))
    if proxy_family == "random_hash":
        proxy_prefix = "hash"
    elif proxy_family == "random_proj_bins":
        proxy_prefix = "proj"
    else:
        raise ValueError(f"Unsupported proxy family: {proxy_family}")
    dec_family = str(dec_cfg.get("family", "random_proj_bins"))
    if dec_family == "random_hash":
        dec_prefix = "hash"
    elif dec_family == "random_proj_bins":
        dec_prefix = "proj"
    else:
        raise ValueError(f"Unsupported decoupled family: {dec_family}")

    proxy_parts = _load_partitions(
        part_root / "proxy" / "val_skew",
        proxy_prefix,
        proxy_cfg["num_partitions"],
        proxy_cfg["num_cells"],
    )
    dec_parts = _load_partitions(
        part_root / "decoupled" / "val_skew",
        dec_prefix,
        dec_cfg["num_partitions"],
        dec_cfg["num_cells"],
    )

    snr_trials = int(cfg.get("analysis", {}).get("snr_null_trials", 25))
    min_cell = int(cfg.get("analysis", {}).get("min_cell", 20))

    # Compute val-side metrics per epoch.
    rows: List[Dict] = []
    device = cfg["compute"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_head(d_in=d_in, hidden_dim=hidden_dim, dropout=dropout).to(device)
    eval_bs = int(cfg["training"].get("eval_batch_size", 2048))

    for e in range(E):
        losses_e = val_loss[e].astype(np.float64, copy=False)
        correct_e = val_correct[e].astype(np.float64, copy=False)
        overall_loss = float(losses_e.mean())
        overall_acc = float(correct_e.mean())

        proxy_worst_acc, proxy_worst_loss, proxy_between_loss, proxy_between_correct = aggregate_proxy_metrics(
            losses=losses_e,
            correct=correct_e,
            partitions=proxy_parts,
            num_cells=int(proxy_cfg["num_cells"]),
            min_cell=min_cell,
        )
        dec_worst_acc, dec_worst_loss, dec_between_loss, dec_between_correct = aggregate_proxy_metrics(
            losses=losses_e,
            correct=correct_e,
            partitions=dec_parts,
            num_cells=int(dec_cfg["num_cells"]),
            min_cell=min_cell,
        )
        snr = snr_between_total_multi(
            correct=correct_e,
            partitions=proxy_parts,
            num_cells=int(proxy_cfg["num_cells"]),
            null_trials=snr_trials,
            seed=int(run.seed) * 1000 + e,
        )

        # Test metrics for this epoch via checkpoint.
        ep = int(e + 1)
        ckpt_path = run.run_dir / f"ckpt_epoch{ep:03d}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logits_test = eval_logits(model, X_test, batch_size=eval_bs, device=device)
        test_overall, test_worst, _acc_map, _cnt_map = group_accuracy_from_logits(logits_test, y_test, g_test)

        rows.append(
            {
                "dataset": run.dataset,
                "regime": run.regime,
                "seed": int(run.seed),
                "tag": run.tag,
                "partition_version": part_version_run,
                "partition_root": str(part_root),
                "epoch": ep,
                "val_overall_acc": overall_acc,
                "val_overall_loss": overall_loss,
                "val_proxy_worst_acc": proxy_worst_acc,
                "val_proxy_worst_loss": proxy_worst_loss,
                "val_proxy_between_loss": proxy_between_loss,
                "val_proxy_between_correct": proxy_between_correct,
                "val_dec_worst_acc": dec_worst_acc,
                "val_dec_worst_loss": dec_worst_loss,
                "val_dec_between_loss": dec_between_loss,
                "val_dec_between_correct": dec_between_correct,
                "val_snr_proxy_correct": snr,
                "test_overall_acc": test_overall,
                "test_worst_group_acc": test_worst,
            }
        )

    df_run = pd.DataFrame(rows)
    df_run.to_csv(run_metrics_path, index=False)

    sel_cfg = cfg["selectors"]
    sel_rows = _selection_summary(
        df_run,
        tail_lambda=float(sel_cfg.get("tailmoderated_lambda", 0.5)),
        hybrid_topk_frac=float(sel_cfg.get("hybrid_topk_frac", 0.1)),
        snr_threshold=float(sel_cfg.get("router", {}).get("snr_threshold", 1.5)),
    )
    for r in sel_rows:
        r.update(
            {
                "dataset": run.dataset,
                "regime": run.regime,
                "seed": run.seed,
                "tag": run.tag,
                "partition_version": part_version_run,
                "partition_root": str(part_root),
            }
        )
    df_sel = pd.DataFrame(sel_rows)
    df_sel.to_csv(sel_path, index=False)
    return df_run, df_sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regimes", default="erm,rcgdro")
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    runs_root = Path(cfg["project"]["runs_dir"])
    runs = discover_runs(runs_root, dataset_name, regimes)
    if not runs:
        raise FileNotFoundError(f"No runs found under {runs_root / dataset_name} for regimes={regimes}")

    metrics_dir = Path(cfg["project"]["artifacts_dir"]) / "metrics"
    ensure_dir(metrics_dir)

    all_runs: List[pd.DataFrame] = []
    all_sel: List[pd.DataFrame] = []
    for run in runs:
        df_run, df_sel = eval_run(cfg, run, overwrite=bool(int(args.overwrite)))
        all_runs.append(df_run)
        all_sel.append(df_sel)

    backbone = cfg["embeddings"]["backbone"]
    out_runs = metrics_dir / f"{dataset_name}_{backbone}_run_metrics.csv"
    out_sel = metrics_dir / f"{dataset_name}_{backbone}_selection_summary.csv"
    pd.concat(all_runs, ignore_index=True).to_csv(out_runs, index=False)
    pd.concat(all_sel, ignore_index=True).to_csv(out_sel, index=False)
    print(f"[eval_runs] wrote: {out_runs}")
    print(f"[eval_runs] wrote: {out_sel}")


if __name__ == "__main__":
    main()
