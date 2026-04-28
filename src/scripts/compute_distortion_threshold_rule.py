import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 3:
        return {"n": n, "intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    X = np.column_stack([np.ones(n), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"n": n, "intercept": float(beta[0]), "slope": float(beta[1]), "r2": r2}


def _row_to_tex_label(scope: str) -> str:
    if scope.lower() == "pooled":
        return "Pooled"
    if scope.lower() == "celeba":
        return "CelebA"
    if scope.lower() == "waterbirds":
        return "Waterbirds"
    if scope.lower() == "camelyon17":
        return "Camelyon17"
    return scope


def _write_table(path: Path, rows: List[Dict[str, float]], eps_tail: float) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "  \\toprule",
        "  Scope & $n$ & Fit ($\\Delta$Tail $= a + b\\cdot$DistMass) & $R^2$ & DistMass budget for $\\Delta$Tail$\\leq$"
        + f"{eps_tail:.1f}"
        + " \\\\",
        "  \\midrule",
    ]
    for r in rows:
        a = r["intercept"]
        b = r["slope"]
        if not math.isfinite(a) or not math.isfinite(b):
            fit_txt = "N/A"
            r2_txt = "N/A"
            budget_txt = "N/A"
        else:
            fit_txt = f"$a={a:.2f},\\ b={b:.2f}$"
            r2_txt = f"{r['r2']:.2f}"
            if b > 0:
                budget = (eps_tail - a) / b
                budget_txt = f"{budget:.3f}"
            else:
                budget_txt = "N/A"
        lines.append(
            f"  {_row_to_tex_label(str(r['scope']))} & {int(r['n'])} & {fit_txt} & {r2_txt} & {budget_txt} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics")
    ap.add_argument("--tables_dir", default="paper/neurips2026_selection_risk/tables")
    ap.add_argument("--eps_tail", type=float, default=1.0)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    row_paths = [
        metrics_dir / "celeba_tail_distortion_rows_v7confclip_head_20260227.csv",
        metrics_dir / "camelyon17_tail_distortion_rows_cam_softclip_a10_p99_20260227.csv",
    ]
    frames = [pd.read_csv(p) for p in row_paths]
    all_df = pd.concat(frames, ignore_index=True)

    summaries: List[Dict[str, float]] = []
    for ds_name, grp in all_df.groupby("dataset"):
        fit = _fit_linear(grp["distortion_mass_selected"].to_numpy(), grp["tail_delta_vs_baseline"].to_numpy())
        fit["scope"] = ds_name
        summaries.append(fit)
    pooled = _fit_linear(all_df["distortion_mass_selected"].to_numpy(), all_df["tail_delta_vs_baseline"].to_numpy())
    pooled["scope"] = "pooled"
    summaries.append(pooled)

    out_df = pd.DataFrame(summaries)
    out_df["eps_tail_budget"] = args.eps_tail
    out_df["distortion_mass_budget"] = np.where(
        out_df["slope"] > 0,
        (args.eps_tail - out_df["intercept"]) / out_df["slope"],
        np.nan,
    )
    out_csv = metrics_dir / "distortion_mass_threshold_rule_head_20260302.csv"
    out_df.to_csv(out_csv, index=False)

    order = {"celeba": 0, "camelyon17": 1, "pooled": 2}
    tex_rows = out_df.copy()
    tex_rows["__ord"] = tex_rows["scope"].map(order).fillna(999)
    tex_rows = tex_rows.sort_values("__ord").drop(columns="__ord")
    _write_table(tables_dir / "table_distortion_threshold_rule.tex", tex_rows.to_dict(orient="records"), args.eps_tail)

    print(f"[distortion-rule] wrote {out_csv}")
    print(f"[distortion-rule] wrote {tables_dir / 'table_distortion_threshold_rule.tex'}")


if __name__ == "__main__":
    main()
