import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def _fmt_pm(mean: float, ci: float, nd: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(ci):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} $\\pm$ {ci:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    order: Dict[str, int] = {"rcgdro": 0, "p95": 1, "p97": 2, "p99": 3}
    dfx = df.copy()
    dfx["ord"] = dfx["regime"].map(order).fillna(99)
    dfx = dfx.sort_values("ord")

    label = {"rcgdro": "Baseline", "p95": "P95", "p97": "P97", "p99": "P99"}

    lines: List[str] = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Regime & Proxy & Tail CVaR & Hard Acc & FracClip \\\\")
    lines.append("\\midrule")
    for _, r in dfx.iterrows():
        lines.append(
            " & ".join(
                [
                    label.get(str(r["regime"]), str(r["regime"])),
                    _fmt_pm(r["proxy_metric_mean"], r["proxy_metric_ci95"]),
                    _fmt_pm(r["tail_hard_cvar_mean"], r["tail_hard_cvar_ci95"]),
                    _fmt_pm(r["hard_acc_mean"], r["hard_acc_ci95"]),
                    _fmt_pm(r["frac_clipped_val_mean"], r["frac_clipped_val_ci95"]),
                ]
            )
            + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = Path(args.out_tex)
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[synthetic-toy-table] wrote {out}")


if __name__ == "__main__":
    main()

