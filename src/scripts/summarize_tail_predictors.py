import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(x.size)}
    pearson = float(np.corrcoef(x, y)[0, 1])
    sx = pd.Series(x).rank(method="average").to_numpy()
    sy = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(sx, sy)[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_csvs", required=True, help="Comma-separated tail-distortion rows csv files.")
    ap.add_argument("--dataset_order", default="celeba,waterbirds,camelyon17")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in str(args.rows_csvs).split(",") if p.strip()]
    frames: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)

    # Include baseline + clipped regimes so correlations reflect the full
    # properness sweep rather than only within-softclip variation.
    dfx = df.copy()

    predictors = [
        "frac_clipped_selected",
        "distortion_mass_selected",
        "anchor_rho_cvar_clip",
    ]
    target = "tail_delta_vs_baseline"

    rows: List[Dict] = []
    for dataset, grp in dfx.groupby("dataset"):
        for p in predictors:
            cc = _corr(grp[p].to_numpy(), grp[target].to_numpy())
            rows.append(
                {
                    "scope": "per_dataset",
                    "dataset": dataset,
                    "predictor": p,
                    "target": target,
                    "n": cc["n"],
                    "pearson": cc["pearson"],
                    "spearman": cc["spearman"],
                }
            )

    # pooled
    for p in predictors:
        cc = _corr(dfx[p].to_numpy(), dfx[target].to_numpy())
        rows.append(
            {
                "scope": "pooled",
                "dataset": "all",
                "predictor": p,
                "target": target,
                "n": cc["n"],
                "pearson": cc["pearson"],
                "spearman": cc["spearman"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["scope", "dataset", "predictor"]).reset_index(drop=True)

    out_dir = paths[0].parent
    ensure_dir(out_dir)
    suf = str(args.out_suffix).strip()
    suf = f"_{suf}" if suf else ""
    out_path = out_dir / f"tail_predictor_correlations{suf}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[tail-predictors] wrote {out_path}")


if __name__ == "__main__":
    main()
