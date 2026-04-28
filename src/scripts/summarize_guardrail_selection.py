import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


def _merge_on_common_keys(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("regime", "seed", "epoch", "tag") if c in left.columns and c in right.columns]
    if not keys:
        raise ValueError("No common merge keys found.")
    return left.merge(right.drop_duplicates(subset=keys).copy(), on=keys, how="left")


def _aggregate_tail(phase1_csv: Path, tail_family: str) -> pd.DataFrame:
    df = pd.read_csv(phase1_csv)
    req = {"regime", "seed", "epoch", "family", "worst_cell_cvar", "worst_cell_mean_loss"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{phase1_csv} missing required columns: {sorted(miss)}")
    df = df[(df["family"] == tail_family) & (df["split"] == "val")].copy() if "split" in df.columns else df[df["family"] == tail_family].copy()
    group_cols = [c for c in ("regime", "seed", "tag", "epoch") if c in df.columns]
    return (
        df.groupby(group_cols, as_index=False)
        .agg({"worst_cell_cvar": "mean", "worst_cell_mean_loss": "mean"})
        .rename(columns={"worst_cell_cvar": "selected_tail_worst_cvar", "worst_cell_mean_loss": "selected_tail_worst_loss"})
    )


def _mean_ci(x: pd.Series) -> tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(ci95_mean(arr))


def _write_latex(df: pd.DataFrame, out_tex: Path, heldout_loss_col: str, heldout_acc_col: str) -> None:
    lines = [
        "\\begin{tabular}{llcccc}",
        "  \\toprule",
        "  Suite & Policy & Accept rate & Held-out loss & Held-out acc & Tail CVaR \\\\",
        "  \\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            "  {suite} & {policy} & {acc_rate:.2f} & {loss:.3f} & {acc:.3f} & {tail:.2f} \\\\".format(
                suite=str(r["suite"]).replace("_", " "),
                policy=str(r["selection_policy"]).replace("_", " "),
                acc_rate=float(r["accept_rate"]),
                loss=float(r[f"{heldout_loss_col}_mean"]),
                acc=float(r[f"{heldout_acc_col}_mean"]),
                tail=float(r["selected_tail_worst_cvar_mean"]),
            )
        )
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection_rows_csv", required=True)
    ap.add_argument("--heldout_rows_csv", required=True)
    ap.add_argument("--phase1_csv", required=True)
    ap.add_argument("--tail_family", default="teacher_difficulty")
    ap.add_argument("--heldout_loss_col", required=True)
    ap.add_argument("--heldout_acc_col", required=True)
    ap.add_argument("--out_rows_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_tex", default="")
    args = ap.parse_args()

    selected = pd.read_csv(args.selection_rows_csv)
    heldout = pd.read_csv(args.heldout_rows_csv)
    tail = _aggregate_tail(Path(args.phase1_csv), tail_family=args.tail_family)

    merged = _merge_on_common_keys(selected, heldout)
    merged = _merge_on_common_keys(merged, tail)

    out_rows = Path(args.out_rows_csv)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_rows, index=False)

    summary_rows: List[dict] = []
    group_cols = ["suite", "target_regime", "selection_policy"]
    for keys, sub in merged.groupby(group_cols, dropna=False):
        suite, target_regime, selection_policy = keys
        record = {
            "suite": suite,
            "target_regime": target_regime,
            "selection_policy": selection_policy,
            "n": int(sub.shape[0]),
            "accept_rate": float(1.0 - sub["fallback_to_baseline"].astype(float).mean()) if "fallback_to_baseline" in sub.columns else np.nan,
            "fallback_rate": float(sub["fallback_to_baseline"].astype(float).mean()) if "fallback_to_baseline" in sub.columns else np.nan,
        }
        for col in [
            "chosen_val_overall_loss",
            "chosen_val_overall_acc",
            "selected_tail_worst_cvar",
            "selected_tail_worst_loss",
            args.heldout_loss_col,
            args.heldout_acc_col,
        ]:
            mean, ci = _mean_ci(sub[col]) if col in sub.columns else (np.nan, np.nan)
            record[f"{col}_mean"] = mean
            record[f"{col}_ci"] = ci
        summary_rows.append(record)

    summary = pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)

    # Add policy deltas relative to baseline within each suite/target.
    delta_rows = []
    for (suite, target_regime), sub in summary.groupby(["suite", "target_regime"], dropna=False):
        if "baseline" not in set(sub["selection_policy"]):
            continue
        base = sub[sub["selection_policy"] == "baseline"].iloc[0]
        for _, row in sub.iterrows():
            rec = row.to_dict()
            for col in [
                "chosen_val_overall_loss",
                "chosen_val_overall_acc",
                "selected_tail_worst_cvar",
                "selected_tail_worst_loss",
                args.heldout_loss_col,
                args.heldout_acc_col,
            ]:
                rec[f"{col}_delta_vs_baseline"] = float(rec.get(f"{col}_mean", np.nan) - base.get(f"{col}_mean", np.nan))
            delta_rows.append(rec)
    summary = pd.DataFrame(delta_rows).sort_values(group_cols).reset_index(drop=True)

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    if str(args.out_tex).strip():
        _write_latex(summary, Path(args.out_tex), heldout_loss_col=args.heldout_loss_col, heldout_acc_col=args.heldout_acc_col)
        print(f"[ok] wrote {args.out_tex}")
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")


if __name__ == "__main__":
    main()
