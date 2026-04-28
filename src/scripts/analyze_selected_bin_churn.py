import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _mean_ci(vals: pd.Series) -> tuple[float, float]:
    arr = vals.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(ci95_mean(arr))


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xs = x.to_numpy(dtype=np.float64)
    ys = y.to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(mask.sum()) < 3:
        return float("nan")
    xs = xs[mask]
    ys = ys[mask]
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_label", required=True)
    ap.add_argument("--selected_rows_csv", required=True)
    ap.add_argument("--bin_churn_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_note_txt", required=True)
    args = ap.parse_args()

    sel = pd.read_csv(args.selected_rows_csv)
    sel = sel[sel["selection_mode"] == "selected_best_proxy"].copy()
    churn = pd.read_csv(args.bin_churn_csv)

    merged = sel.merge(
        churn[["regime", "seed", "tag", "epoch", "train_bin_churn"]],
        on=["regime", "seed", "tag", "epoch"],
        how="left",
        validate="one_to_one",
    )
    if merged.empty:
        raise FileNotFoundError("No selected rows matched bin-churn file.")

    base = merged[merged["regime"] == "rcgdro"][["seed", "tail_worst_cvar"]].rename(columns={"tail_worst_cvar": "base_tail"})
    merged = merged.merge(base, on="seed", how="left", validate="many_to_one")
    merged["delta_tail"] = merged["tail_worst_cvar"] - merged["base_tail"]

    rows: List[Dict[str, object]] = []
    for regime, sub in merged.groupby("regime", dropna=False):
        churn_m, churn_ci = _mean_ci(sub["train_bin_churn"])
        rows.append(
            {
                "dataset": args.dataset_label,
                "regime": regime,
                "n": int(sub.shape[0]),
                "selected_epoch_churn_mean": churn_m,
                "selected_epoch_churn_ci": churn_ci,
                "tail_delta_mean": float(sub["delta_tail"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)

    out_csv = Path(args.out_summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    corr_tail = _pearson(merged["train_bin_churn"], merged["delta_tail"])
    corr_frac = _pearson(merged["train_bin_churn"], merged["frac_clipped_val"])
    note = (
        f"{args.dataset_label}: corr(selected_epoch_churn, tail_delta)={corr_tail:.3f}; "
        f"corr(selected_epoch_churn, frac_clipped_val)={corr_frac:.3f}\n"
    )
    out_note = Path(args.out_note_txt)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(note, encoding="utf-8")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_note}")


if __name__ == "__main__":
    main()
