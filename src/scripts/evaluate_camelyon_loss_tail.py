import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.wilds_loader import load_wilds_dataset
from ..utils.stats import ci95_mean
from .camelyon_domain_eval import (
    WildsWithDomain,
    _bce_losses_from_logits,
    build_backbone,
    build_head,
    eval_logits,
    eval_logits_full,
)


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(ci95_mean(x))


def _tail_cvar(losses: np.ndarray, q: float) -> float:
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    if losses.size == 0:
        return np.nan
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    k = max(1, int(np.ceil(q * losses.size)))
    idx = np.argpartition(losses, -k)[-k:]
    return float(np.mean(losses[idx]))


def _resolve_tag_dir(reg_dir: Path, requested_tag: str | None, tag_filter: str | None) -> Path:
    tag_dirs = [d for d in reg_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not tag_dirs:
        raise FileNotFoundError(f"No run tags found under {reg_dir}")

    req = (requested_tag or "").strip()
    if req:
        exact = reg_dir / req
        if exact.exists() and (exact / "config.json").exists():
            return exact
        raise FileNotFoundError(f"Requested tag '{req}' not found under {reg_dir}")

    candidates = tag_dirs
    if tag_filter:
        candidates = [d for d in candidates if tag_filter in d.name]
        if not candidates:
            raise FileNotFoundError(f"No tags matched tag_filter='{tag_filter}' under {reg_dir}")

    if len(candidates) != 1:
        names = sorted([d.name for d in candidates])
        raise ValueError(f"Ambiguous tags under {reg_dir}; provide tag in selector_rows_csv. Candidates: {names}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="camelyon17")
    ap.add_argument("--selector_rows_csv", required=True)
    ap.add_argument("--suite", default="")
    ap.add_argument("--selection_policies", default="baseline,proxy_only,val_loss_only,guardrail")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--heldout_domain", type=int, default=2)
    ap.add_argument("--tail_q", type=float, default=0.1)
    ap.add_argument("--tag_filter", default="")
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    if dataset_name != "camelyon17":
        raise ValueError("This evaluator is only for camelyon17.")

    backbone = cfg["embeddings"]["backbone"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    runs_dir = Path(cfg["project"]["runs_dir"])
    feat_dir = artifacts_dir / "embeds" / f"{dataset_name}_{backbone}"

    device = str(cfg["compute"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    eval_batch = int(cfg.get("finetune", {}).get("eval_batch_size", cfg["training"].get("eval_batch_size", 2048)))
    hidden_dim = int(cfg["training"].get("hidden_dim", 0))
    dropout = float(cfg["training"].get("dropout", 0.0))

    selector_df = pd.read_csv(args.selector_rows_csv)
    required = {"seed", "epoch", "regime", "selection_policy"}
    missing = required - set(selector_df.columns)
    if missing:
        raise ValueError(f"selector_rows_csv missing required columns: {sorted(missing)}")

    if str(args.suite).strip():
        if "suite" not in selector_df.columns:
            raise ValueError("selector_rows_csv has no suite column, but --suite was provided.")
        selector_df = selector_df[selector_df["suite"] == str(args.suite).strip()].copy()

    policies = [p.strip() for p in str(args.selection_policies).split(",") if p.strip()]
    selector_df = selector_df[selector_df["selection_policy"].isin(policies)].copy()

    if str(args.seeds).strip():
        seeds = {int(s.strip()) for s in str(args.seeds).split(",") if s.strip()}
        selector_df = selector_df[selector_df["seed"].astype(int).isin(seeds)].copy()

    if selector_df.empty:
        raise ValueError("No selector rows remain after filtering.")

    has_suite = "suite" in selector_df.columns
    has_tag = "tag" in selector_df.columns

    X_test = y_test = a_test = None
    full_test_loader = None
    data_dir = cfg["dataset"].get("data_dir") or cfg["paths"]["wilds_data_dir"]
    wilds_name = cfg["dataset"]["wilds_dataset"]
    domain_field = str(cfg["dataset"].get("spurious_metadata_field", "hospital"))

    out_rows: List[Dict[str, object]] = []
    for _, row in selector_df.iterrows():
        suite = str(row["suite"]) if has_suite else ""
        selection_policy = str(row["selection_policy"])
        regime = str(row["regime"])
        seed = int(row["seed"])
        epoch = int(row["epoch"])
        requested_tag = str(row["tag"]).strip() if has_tag and pd.notna(row["tag"]) else None

        reg_dir = runs_dir / dataset_name / regime / f"seed{seed}"
        tag_dir = _resolve_tag_dir(reg_dir, requested_tag=requested_tag, tag_filter=str(args.tag_filter))
        ckpt = tag_dir / f"ckpt_epoch{epoch:03d}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        state = torch.load(ckpt, map_location=device)
        model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        run_cfg = json.loads((tag_dir / "config.json").read_text(encoding="utf-8"))
        keys = list(model_state.keys())
        is_full_model = any(k.startswith("conv1.") or k.startswith("layer1.") or k.startswith("fc.") for k in keys)

        if is_full_model:
            if full_test_loader is None:
                ds = load_wilds_dataset(wilds_name, data_dir, download=False)
                fields = list(getattr(ds, "metadata_fields", []))
                if domain_field not in fields:
                    raise ValueError(f"Domain field '{domain_field}' not found in metadata fields: {fields}")
                domain_col = int(fields.index(domain_field))
                model_tmp, tfm = build_backbone(backbone)
                del model_tmp
                test_base = ds.get_subset("test", frac=1.0, transform=tfm)
                test_ds = WildsWithDomain(test_base, domain_col=domain_col)
                full_test_loader = DataLoader(
                    test_ds,
                    batch_size=eval_batch,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
            model, _ = build_backbone(backbone)
            model = model.to(device)
            model.load_state_dict(model_state, strict=True)
            logits_test, y_test_run, a_test_run = eval_logits_full(model, full_test_loader, device)
            mask = np.asarray(a_test_run, dtype=np.int64) == int(args.heldout_domain)
        else:
            if X_test is None:
                X_test = np.load(feat_dir / "X_test.npy", mmap_mode="r")
                y_test = np.load(feat_dir / "y_test.npy")
                a_test = np.load(feat_dir / "a_test.npy")
            d_in = int(run_cfg.get("d_in", int(X_test.shape[1])))
            hidden_dim_run = int(run_cfg.get("training", {}).get("hidden_dim", hidden_dim))
            dropout_run = float(run_cfg.get("training", {}).get("dropout", dropout))
            eval_batch_run = int(run_cfg.get("training", {}).get("eval_batch_size", eval_batch))
            model = build_head(d_in, hidden_dim_run, dropout_run).to(device)
            model.load_state_dict(model_state, strict=True)
            logits_test = eval_logits(model, X_test, y_test, eval_batch_run, device)
            y_test_run = np.asarray(y_test, dtype=np.int64)
            a_test_run = np.asarray(a_test, dtype=np.int64)
            mask = a_test_run == int(args.heldout_domain)

        if not np.any(mask):
            raise ValueError(
                f"Held-out domain {args.heldout_domain} not present for regime={regime}, seed={seed}, epoch={epoch}"
            )

        losses_test = _bce_losses_from_logits(logits_test[mask], y_test_run[mask])
        preds_test = (np.asarray(logits_test[mask]) >= 0).astype(np.int64)
        row_out = {
            "suite": suite,
            "selection_policy": selection_policy,
            "regime": regime,
            "seed": int(seed),
            "epoch": int(epoch),
            "tag": tag_dir.name,
            "n_eval": int(mask.sum()),
            "test_acc": float(np.mean(preds_test == y_test_run[mask])),
            "test_loss_mean": float(np.mean(losses_test)),
            "test_loss_tail_cvar": _tail_cvar(losses_test, float(args.tail_q)),
        }
        out_rows.append(row_out)

    rows_df = pd.DataFrame(out_rows).sort_values(["suite", "selection_policy", "seed"]).reset_index(drop=True)
    out_rows_path = Path(args.out_rows)
    out_rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(out_rows_path, index=False)

    summary_rows: List[Dict[str, object]] = []
    group_keys = ["suite"] if has_suite else []
    for key, sub in rows_df.groupby(group_keys or [lambda _: True], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        suite_name = key[0] if has_suite else ""
        base = sub[sub["selection_policy"] == "baseline"][["seed", "test_loss_mean", "test_loss_tail_cvar", "test_acc"]]
        base = base.rename(
            columns={
                "test_loss_mean": "baseline_test_loss_mean",
                "test_loss_tail_cvar": "baseline_test_loss_tail_cvar",
                "test_acc": "baseline_test_acc",
            }
        )
        for policy, psub in sub.groupby("selection_policy", dropna=False):
            rec: Dict[str, object] = {
                "suite": suite_name,
                "selection_policy": str(policy),
                "n": int(psub.shape[0]),
            }
            for col in ["test_loss_mean", "test_loss_tail_cvar", "test_acc"]:
                mean, ci = _mean_ci(psub[col].to_numpy())
                rec[f"{col}_mean"] = mean
                rec[f"{col}_ci"] = ci
            if str(policy) != "baseline" and not base.empty:
                merged = psub.merge(base, on="seed", how="inner")
                for col, base_col in [
                    ("test_loss_mean", "baseline_test_loss_mean"),
                    ("test_loss_tail_cvar", "baseline_test_loss_tail_cvar"),
                    ("test_acc", "baseline_test_acc"),
                ]:
                    deltas = merged[col].to_numpy(dtype=np.float64) - merged[base_col].to_numpy(dtype=np.float64)
                    mean, ci = _mean_ci(deltas)
                    rec[f"delta_{col}_vs_baseline_mean"] = mean
                    rec[f"delta_{col}_vs_baseline_ci"] = ci
                    if col != "test_acc":
                        rec[f"worse_seed_count_{col}"] = int(np.sum(deltas > 0))
                    else:
                        rec[f"worse_seed_count_{col}"] = int(np.sum(deltas < 0))
            summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows).sort_values(["suite", "selection_policy"]).reset_index(drop=True)
    out_summary_path = Path(args.out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_summary_path, index=False)

    print(f"[camelyon_loss_tail] wrote rows to {out_rows_path}")
    print(f"[camelyon_loss_tail] wrote summary to {out_summary_path}")


if __name__ == "__main__":
    main()
