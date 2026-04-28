import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BudgetSpec:
    budget: float
    path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _specs(root: Path) -> list[BudgetSpec]:
    art = root / "artifacts" / "metrics"
    return [
        BudgetSpec(1.10, art / "guardrail_merged_rows_civilcomments_objfam_p95_ratio110_20260328.csv"),
        BudgetSpec(1.25, art / "guardrail_merged_rows_civilcomments_objfam_p95_ratio125_20260328.csv"),
        BudgetSpec(1.50, art / "guardrail_merged_rows_civilcomments_objfam_p95_ratio150_20260328.csv"),
    ]


def _n_of_k(n: int, k: int) -> str:
    return f"{int(n)}/{int(k)}"


def _write_tex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{lccc}",
        r"  \toprule",
        r"  Budget & Accepted target run & Fallback to baseline & Matches proxy-only \\",
        r"  \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "  {budget:.2f}x & {accept} & {fallback} & {proxy} \\\\".format(
                budget=row["budget"],
                accept=row["accept_n_of_k"],
                fallback=row["fallback_n_of_k"],
                proxy=row["match_proxy_n_of_k"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="artifacts/metrics/civilcomments_guardrail_objfam_summary_20260329.csv")
    ap.add_argument("--out_tex", default="paper/neurips2026_selection_risk/tables/table_civilcomments_guardrail.tex")
    args = ap.parse_args()

    root = _repo_root()
    rows = []
    for spec in _specs(root):
        df = pd.read_csv(spec.path)
        base = df[df["selection_policy"] == "baseline"].drop_duplicates(subset=["seed"]).set_index("seed")
        proxy = df[df["selection_policy"] == "proxy_only"].drop_duplicates(subset=["seed"]).set_index("seed")
        guard = df[df["selection_policy"] == "guardrail"].drop_duplicates(subset=["seed"]).set_index("seed")
        seeds = sorted(set(base.index) & set(proxy.index) & set(guard.index))
        base = base.loc[seeds]
        proxy = proxy.loc[seeds]
        guard = guard.loc[seeds]

        fallback = guard["fallback_to_baseline"].astype(bool)
        match_proxy = (guard["regime"] == proxy["regime"]) & (guard["epoch"] == proxy["epoch"]) & (guard["tag"] == proxy["tag"])
        match_base = (guard["regime"] == base["regime"]) & (guard["epoch"] == base["epoch"]) & (guard["tag"] == base["tag"])

        rows.append(
            {
                "budget": spec.budget,
                "n": int(len(seeds)),
                "accept_rate": float((~fallback).mean()),
                "fallback_rate": float(fallback.mean()),
                "match_proxy_rate": float(match_proxy.mean()),
                "match_base_rate": float(match_base.mean()),
                "accept_n_of_k": _n_of_k((~fallback).sum(), len(seeds)),
                "fallback_n_of_k": _n_of_k(fallback.sum(), len(seeds)),
                "match_proxy_n_of_k": _n_of_k(match_proxy.sum(), len(seeds)),
                "match_base_n_of_k": _n_of_k(match_base.sum(), len(seeds)),
            }
        )

    out = pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)
    out_csv = root / args.out_csv
    out_tex = root / args.out_tex
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_tex(out, out_tex)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
