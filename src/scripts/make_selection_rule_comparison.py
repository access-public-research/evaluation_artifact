import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASET_LABELS = {
    "celeba": "CelebA",
    "waterbirds": "Waterbirds",
    "camelyon17": "Camelyon17",
}


def _infer_dataset(path: Path) -> str:
    lower = path.name.lower()
    for key, label in DATASET_LABELS.items():
        if key in lower:
            return label
    raise ValueError(f"Could not infer dataset from filename: {path}")


def _load_effect(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["dataset"] = _infer_dataset(path)
    return out


def _load_perf_override(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dataset = _infer_dataset(path)
    if "test_hosp_2_acc" in df.columns:
        out = df.groupby("regime", as_index=False)["test_hosp_2_acc"].mean().rename(columns={"test_hosp_2_acc": "perf"})
    elif "test_oracle_wg_acc_mean" in df.columns:
        out = df[["regime", "test_oracle_wg_acc_mean"]].rename(columns={"test_oracle_wg_acc_mean": "perf"})
    elif "oracle_wg_acc_mean" in df.columns:
        out = df[["regime", "oracle_wg_acc_mean"]].rename(columns={"oracle_wg_acc_mean": "perf"})
    else:
        raise ValueError(f"Could not infer performance column from {path}")
    out["dataset"] = dataset
    return out


def _perf_column(df: pd.DataFrame, dataset: str) -> str:
    if dataset == "Camelyon17":
        for cand in ("test_hosp2_acc_mean", "test_hosp_2_acc_mean"):
            if cand in df.columns:
                return cand
        raise ValueError("Camelyon effect table missing held-out performance column.")
    return "oracle_wg_acc_mean"


def _proxy_value(row: pd.Series, mode: str) -> float:
    mode = str(mode).strip().lower()
    if mode == "clip_aware":
        v = row.get("proxy_worst_loss_clip_mean")
        if pd.notna(v):
            return float(v)
        return float(row["proxy_worst_loss_mean"])
    if mode == "stationary_unclipped":
        return float(row["proxy_worst_loss_mean"])
    raise ValueError(f"Unknown proxy mode: {mode}")


def _short_regime(regime: str) -> str:
    r = str(regime)
    rr = r.lower()
    if r == "rcgdro":
        return "rcgdro"
    if "p95" in rr:
        return "P95"
    if "p96" in rr:
        return "P96"
    if "p97" in rr:
        return "P97"
    if "p98" in rr:
        return "P98"
    if "p99" in rr:
        return "P99"
    return r


def _select_proxy_only(g: pd.DataFrame, mode: str) -> pd.Series:
    return g.assign(proxy_sel=g.apply(lambda r: _proxy_value(r, mode), axis=1)).sort_values(
        ["proxy_sel", "tail_worst_cvar_mean"], ascending=[True, True]
    ).iloc[0]


def _select_tail_only(g: pd.DataFrame, perf_col: str) -> pd.Series:
    return g.sort_values(["tail_worst_cvar_mean", perf_col], ascending=[True, False]).iloc[0]


def _select_constrained(g: pd.DataFrame, proxy_tol: float, tail_budget: float, mode: str, perf_col: str) -> pd.Series:
    base = g[g["regime"] == "rcgdro"].iloc[0]
    proxy_thr = _proxy_value(base, mode) * (1.0 + float(proxy_tol))
    tail_thr = float(base["tail_worst_cvar_mean"]) + float(tail_budget)
    gg = g.assign(proxy_sel=g.apply(lambda r: _proxy_value(r, mode), axis=1))
    feasible = gg[(gg["proxy_sel"] <= proxy_thr) & (gg["tail_worst_cvar_mean"] <= tail_thr)].copy()
    if feasible.empty:
        return base
    return feasible.sort_values([perf_col, "tail_worst_cvar_mean", "proxy_sel"], ascending=[False, True, True]).iloc[0]


def _write_tex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{tabular}{llccc}",
        "  \\toprule",
        "  Dataset & Rule & Pick & $\\Delta$Tail & Perf \\\\",
        "  \\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            "  {dataset} & {rule} & {pick} & {dtail:+.2f} & {perf:.3f} \\\\".format(
                dataset=r["dataset"],
                rule=r["rule"],
                pick=r["pick"],
                dtail=r["tail_delta"],
                perf=r["perf"],
            )
        )
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effect_csvs", nargs="+", required=True)
    ap.add_argument("--perf_csvs", nargs="*", default=[])
    ap.add_argument("--proxy_tol", type=float, default=0.05)
    ap.add_argument("--tail_budget", type=float, default=1.0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    perf_overrides: Dict[str, pd.DataFrame] = {}
    for raw in args.perf_csvs:
        pdf = _load_perf_override(Path(raw))
        perf_overrides[str(pdf["dataset"].iloc[0])] = pdf

    rows: List[Dict[str, object]] = []
    for raw in args.effect_csvs:
        df = _load_effect(Path(raw))
        dataset = str(df["dataset"].iloc[0])
        if dataset in perf_overrides:
            df = df.merge(perf_overrides[dataset][["regime", "perf"]], on="regime", how="left", validate="one_to_one")
            perf_col = "perf"
        else:
            perf_col = _perf_column(df, dataset)
        base = df[df["regime"] == "rcgdro"].iloc[0]
        rules = {
            "Proxy-only (clip-aware)": _select_proxy_only(df, "clip_aware"),
            "Proxy-only (stationary)": _select_proxy_only(df, "stationary_unclipped"),
            "Tail-only": _select_tail_only(df, perf_col),
            "Constrained": _select_constrained(df, args.proxy_tol, args.tail_budget, "clip_aware", perf_col),
        }
        for rule, row in rules.items():
            rows.append(
                {
                    "dataset": dataset,
                    "rule": rule,
                    "pick": _short_regime(str(row["regime"])),
                    "tail_delta": float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
                    "perf": float(row[perf_col]),
                }
            )

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_tex(out, Path(args.out_tex))
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
