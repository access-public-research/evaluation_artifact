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


def _proxy_value(df: pd.DataFrame, regime: str) -> pd.Series:
    if "clip" in str(regime).lower():
        vals = pd.to_numeric(df["proxy_worst_loss_clip_min"], errors="coerce")
        if np.isfinite(vals.to_numpy(dtype=np.float64)).any():
            return vals
    return pd.to_numeric(df["proxy_worst_loss_min"], errors="coerce")


def _short_regime(regime: str) -> str:
    rr = str(regime).lower()
    if regime == "rcgdro":
        return "rcgdro"
    for tag in ("p95", "p96", "p97", "p98", "p99"):
        if tag in rr:
            return tag.upper()
    return regime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_label", required=True)
    ap.add_argument("--selected_rows_csv", required=True)
    ap.add_argument("--phase0_csv", required=True)
    ap.add_argument("--phase1_csv", required=True)
    ap.add_argument("--baseline_regime", default="rcgdro")
    ap.add_argument("--proxy_family", default="conf_init_wpl")
    ap.add_argument("--tail_family", default="teacher_difficulty")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    sel = pd.read_csv(args.selected_rows_csv)
    phase0 = pd.read_csv(args.phase0_csv)
    phase1 = pd.read_csv(args.phase1_csv)
    phase1 = phase1[phase1["split"] == "val"].copy()

    if "selection_mode" in sel.columns:
        sel = sel[sel["selection_mode"] == "selected_best_proxy"].copy()
    else:
        sel = sel.copy()

    # Some paper-facing selected-row exports only keep (regime, seed, epoch).
    # Recover the unique tag from phase0 so downstream joins still work.
    if "tag" not in sel.columns and "tag" in phase0.columns:
        tag_map = phase0[["regime", "seed", "epoch", "tag"]].drop_duplicates()
        sel = sel.merge(tag_map, on=["regime", "seed", "epoch"], how="left")

    rows: List[Dict[str, object]] = []
    for _, srow in sel.iterrows():
        regime = str(srow["regime"])
        seed = int(srow["seed"])
        epoch = int(srow["epoch"])
        tag = str(srow["tag"])

        p0 = phase0[
            (phase0["regime"] == regime)
            & (phase0["seed"] == seed)
            & (phase0["tag"] == tag)
            & (phase0["family"] == args.proxy_family)
            & (phase0["epoch"] == epoch)
        ].copy()
        if p0.empty:
            continue
        p0["proxy_selected"] = _proxy_value(p0, regime)

        p1 = phase1[
            (phase1["regime"] == regime)
            & (phase1["seed"] == seed)
            & (phase1["tag"] == tag)
            & (phase1["family"] == args.tail_family)
            & (phase1["epoch"] == epoch)
        ][["bank", "worst_cell_cvar"]].copy()

        merged = p0.merge(p1, on="bank", how="inner", validate="one_to_one")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "dataset": args.dataset_label,
                    "regime": regime,
                    "regime_label": _short_regime(regime),
                    "seed": seed,
                    "epoch": epoch,
                    "bank": str(row["bank"]),
                    "proxy_selected": float(row["proxy_selected"]),
                    "tail_worst_cvar": float(row["worst_cell_cvar"]),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No per-bank selected rows found.")

    base = df[df["regime"] == args.baseline_regime][["seed", "bank", "proxy_selected", "tail_worst_cvar"]].rename(
        columns={"proxy_selected": "base_proxy", "tail_worst_cvar": "base_tail"}
    )
    comp = df.merge(base, on=["seed", "bank"], how="left", validate="many_to_one")
    comp["delta_proxy"] = comp["proxy_selected"] - comp["base_proxy"]
    comp["delta_tail"] = comp["tail_worst_cvar"] - comp["base_tail"]
    comp["proxy_improves"] = comp["delta_proxy"] < 0
    comp["tail_worsens"] = comp["delta_tail"] > 0
    comp["decoupling"] = comp["proxy_improves"] & comp["tail_worsens"]

    summary_rows: List[Dict[str, object]] = []
    for (bank, regime_label), sub in comp.groupby(["bank", "regime_label"], dropna=False):
        if regime_label == _short_regime(args.baseline_regime):
            continue
        proxy_m, proxy_ci = _mean_ci(sub["delta_proxy"])
        tail_m, tail_ci = _mean_ci(sub["delta_tail"])
        summary_rows.append(
            {
                "dataset": args.dataset_label,
                "bank": bank,
                "regime": regime_label,
                "n": int(sub.shape[0]),
                "delta_proxy_mean": proxy_m,
                "delta_proxy_ci": proxy_ci,
                "delta_tail_mean": tail_m,
                "delta_tail_ci": tail_ci,
                "proxy_improves_frac": float(sub["proxy_improves"].mean()),
                "tail_worsens_frac": float(sub["tail_worsens"].mean()),
                "decoupling_frac": float(sub["decoupling"].mean()),
            }
        )
    out = pd.DataFrame(summary_rows).sort_values(["bank", "regime"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    lines = [
        "\\begin{tabular}{llccccc}",
        "  \\toprule",
        "  Dataset & Bank & Regime & $\\Delta$Proxy & $\\Delta$Tail & Proxy improve frac & Decoupling frac \\\\",
        "  \\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            "  {dataset} & {bank} & {regime} & {dproxy:+.3f} & {dtail:+.3f} & {pfrac:.2f} & {dfrac:.2f} \\\\".format(
                dataset=r["dataset"],
                bank=r["bank"],
                regime=r["regime"],
                dproxy=r["delta_proxy_mean"],
                dtail=r["delta_tail_mean"],
                pfrac=r["proxy_improves_frac"],
                dfrac=r["decoupling_frac"],
            )
        )
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
