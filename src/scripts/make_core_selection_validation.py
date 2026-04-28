import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_EFFECT_CSVS = [
    "replication_rcg/artifacts/metrics/celeba_effect_size_v7confclip_p60_p95_p97_p99_10s.csv",
    "replication_rcg/artifacts/metrics/camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv",
]

DATASET_LABELS = {
    "celeba": "CelebA",
    "waterbirds": "Waterbirds",
    "camelyon17": "Camelyon17",
}


def _load_effect_csv(path: Path, selection_metric_mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"regime", "tail_worst_cvar_mean"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing required columns: {sorted(miss)}")

    dataset_key = None
    for key in DATASET_LABELS:
        if key in path.name.lower():
            dataset_key = key
            break
    if dataset_key is None:
        raise ValueError(f"Could not infer dataset from filename: {path.name}")

    perf_col = "oracle_wg_acc_mean"
    if dataset_key == "camelyon17":
        perf_col = "test_hosp2_acc_mean" if "test_hosp2_acc_mean" in df.columns else "test_hosp_2_acc_mean"
    if perf_col not in df.columns and dataset_key != "camelyon17":
        raise ValueError(f"{path} missing performance column: {perf_col}")

    proxy_col = "proxy_worst_loss_mean"
    mode = str(selection_metric_mode).strip().lower()
    if mode == "clip_aware" and "proxy_worst_loss_clip_mean" in df.columns:
        clipped = df["regime"].astype(str).str.contains("clip", case=False, regex=False)
        df["proxy_selected_mean"] = df[proxy_col]
        df.loc[clipped, "proxy_selected_mean"] = df.loc[clipped, "proxy_worst_loss_clip_mean"]
    elif mode == "stationary_unclipped":
        df["proxy_selected_mean"] = df[proxy_col]
    else:
        raise ValueError(f"Unknown selection_metric_mode={selection_metric_mode}")

    cols = ["regime", "proxy_selected_mean", "tail_worst_cvar_mean", "frac_clipped_val_mean"]
    if perf_col in df.columns:
        cols.append(perf_col)
    out = df[cols].copy()
    rename_map = {
        "tail_worst_cvar_mean": "tail_mean",
        "frac_clipped_val_mean": "fracclip_mean",
    }
    if perf_col in out.columns:
        rename_map[perf_col] = "perf_mean"
    out = out.rename(columns=rename_map)
    out["dataset"] = DATASET_LABELS[dataset_key]
    return out


def _select_proxy_only(g: pd.DataFrame) -> pd.Series:
    return g.sort_values(["proxy_selected_mean", "tail_mean", "perf_mean"], ascending=[True, True, False]).iloc[0]


def _select_constrained(g: pd.DataFrame, proxy_tol: float, tail_budget: float) -> pd.Series:
    baseline = g[g["regime"] == "rcgdro"].iloc[0]
    proxy_thr = baseline["proxy_selected_mean"] * (1.0 + proxy_tol)
    tail_thr = baseline["tail_mean"] + tail_budget

    feasible = g[(g["proxy_selected_mean"] <= proxy_thr) & (g["tail_mean"] <= tail_thr)].copy()
    if feasible.empty:
        return baseline
    feasible = feasible.sort_values(["perf_mean", "tail_mean", "fracclip_mean"], ascending=[False, True, True])
    return feasible.iloc[0]


def _short_regime(r: str) -> str:
    if r == "rcgdro":
        return r
    rr = str(r).lower()
    if "p95" in rr:
        return "P95"
    if "p97" in rr:
        return "P97"
    if "p99" in rr:
        return "P99"
    return str(r)


def _write_latex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "  \\toprule",
        "  Dataset & Proxy-only pick & Constrained pick & $\\Delta$Tail (proxy-only) & $\\Delta$Tail (constrained) \\\\",
        "  \\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            "  {dataset} & {proxy} & {cons} & {dproxy:+.2f} & {dcons:+.2f} \\\\".format(
                dataset=r["dataset"],
                proxy=r["proxy_only_pick"],
                cons=r["constrained_pick"],
                dproxy=r["proxy_only_tail_delta"],
                dcons=r["constrained_tail_delta"],
            )
        )
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effect_csvs", nargs="+", default=DEFAULT_EFFECT_CSVS)
    ap.add_argument(
        "--camelyon_domain_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_resnet50_domain_acc_cam_softclip_a10_p99_20260207.csv",
    )
    ap.add_argument("--proxy_tol", type=float, default=0.05)
    ap.add_argument("--tail_budget", type=float, default=1.0)
    ap.add_argument(
        "--selection_metric_mode",
        default="clip_aware",
        choices=["clip_aware", "stationary_unclipped"],
    )
    ap.add_argument("--out_csv", default="replication_rcg/artifacts/metrics/core_selection_policy_validation_20260305.csv")
    ap.add_argument("--out_tex", default="paper/neurips2026_selection_risk/tables/table_selection_policy_core3.tex")
    args = ap.parse_args()

    frames: List[pd.DataFrame] = []
    for raw in args.effect_csvs:
        p = Path(raw)
        df = _load_effect_csv(p, selection_metric_mode=args.selection_metric_mode)
        if "camelyon17" in p.name.lower():
            dom = pd.read_csv(args.camelyon_domain_csv)
            perf_col = "test_hosp_2_acc"
            perf_mean = dom.groupby("regime")[perf_col].mean().to_dict()
            df["perf_mean"] = df["regime"].map(perf_mean)
        frames.append(df)
    core = pd.concat(frames, ignore_index=True)

    records: List[Dict[str, object]] = []
    for dataset, g in core.groupby("dataset", dropna=False):
        baseline = g[g["regime"] == "rcgdro"].iloc[0]
        proxy_pick = _select_proxy_only(g)
        constrained_pick = _select_constrained(g, args.proxy_tol, args.tail_budget)
        records.append(
            {
                "dataset": dataset,
                "proxy_only_pick": _short_regime(str(proxy_pick["regime"])),
                "constrained_pick": _short_regime(str(constrained_pick["regime"])),
                "proxy_only_tail_delta": float(proxy_pick["tail_mean"] - baseline["tail_mean"]),
                "constrained_tail_delta": float(constrained_pick["tail_mean"] - baseline["tail_mean"]),
            }
        )

    out = pd.DataFrame(records).sort_values("dataset").reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_latex(out, Path(args.out_tex))
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
