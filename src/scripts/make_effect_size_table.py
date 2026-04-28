import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _mean_ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    mean = float(x.mean())
    if x.size == 1:
        return mean, 0.0
    return mean, ci95_mean(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--metrics", default="proxy_worst_loss_clip,proxy_worst_loss,tail_worst_cvar,oracle_wg_acc,val_overall_acc,frac_clipped_val")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = []
    for regime, sub in df.groupby("regime"):
        row = {"regime": regime, "n": int(sub.shape[0])}
        for metric in metrics:
            if metric not in sub.columns:
                continue
            mean, ci = _mean_ci(sub[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci"] = ci
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("regime")
    out.to_csv(Path(args.out_csv), index=False)


if __name__ == "__main__":
    main()
