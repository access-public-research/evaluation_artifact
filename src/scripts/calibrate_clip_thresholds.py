import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_SPECS = [
    ("celeba", "erm", "v7confclip_p60_10s", ""),
    ("waterbirds", "erm", "h256_do0.0_lr0.001_wd0.0001_bs512_ep30", "_wb_h256cal"),
    ("camelyon17", "erm", "h0_do0.0_lr0.001_wd0.0001_bs512_ep30", "_cam"),
]


def _latest_tag_dir(seed_dir: Path, tag_hint: str = "") -> Path:
    tags = [d for d in seed_dir.iterdir() if d.is_dir()]
    if tag_hint:
        exact = [d for d in tags if d.name == tag_hint]
        if exact:
            tags = exact
        else:
            hinted = [d for d in tags if tag_hint in d.name]
            if hinted:
                tags = hinted
    tags = sorted(tags, key=lambda p: p.stat().st_mtime, reverse=True)
    if not tags:
        raise FileNotFoundError(f"No tag directories under {seed_dir}")
    return tags[0]


def _load_epoch1_train_losses(run_dir: Path) -> np.ndarray:
    arr = np.load(run_dir / "train_loss_by_epoch.npy", mmap_mode="r")
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(f"Unexpected train_loss_by_epoch shape for {run_dir}: {arr.shape}")
    return np.asarray(arr[0], dtype=np.float64)


def _config_thresholds(configs_root: Path, suffix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in [95, 97, 99]:
        cfg = configs_root / "regimes" / f"rcgdro_softclip_p{p}_a10{suffix}.yaml"
        text = cfg.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("clip_loss:"):
                out[f"config_q{p}"] = float(line.split(":", 1)[1].strip())
                break
        else:
            raise KeyError(f"clip_loss not found in {cfg}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="replication_rcg/runs")
    ap.add_argument("--quantiles", default="0.95,0.97,0.99")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs_root", default="replication_rcg/configs")
    ap.add_argument("--out_csv", default="replication_rcg/artifacts/metrics/clip_threshold_calibration_epoch1_20260307.csv")
    args = ap.parse_args()

    qs = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    rows: List[Dict[str, object]] = []
    runs_root = Path(args.runs_root)
    configs_root = Path(args.configs_root)
    for dataset, regime, tag_hint, suffix in DEFAULT_SPECS:
        seed_dir = runs_root / dataset / regime / f"seed{int(args.seed)}"
        run_dir = _latest_tag_dir(seed_dir, tag_hint=tag_hint)
        losses = _load_epoch1_train_losses(run_dir)
        cfg_thresholds = _config_thresholds(configs_root, suffix=suffix)
        row: Dict[str, object] = {
            "dataset": dataset,
            "regime": regime,
            "seed": int(args.seed),
            "tag_hint": tag_hint,
            "run_dir": str(run_dir),
        }
        for q in qs:
            qp = int(round(100 * q))
            row[f"q{qp}"] = float(np.quantile(losses, q))
            row[f"config_q{qp}"] = float(cfg_thresholds[f"config_q{qp}"])
            row[f"delta_q{qp}_vs_config"] = float(row[f"q{qp}"] - row[f"config_q{qp}"])
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
