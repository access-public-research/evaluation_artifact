import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepSpec:
    suite: str
    label: str
    heldout_loss_col: str
    heldout_acc_col: str
    paths: dict[float, Path]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _specs(root: Path) -> list[SweepSpec]:
    art = root / "artifacts" / "metrics"
    return [
        SweepSpec(
            suite="camelyon_erm_p95",
            label="Camelyon17 ERM",
            heldout_loss_col="test_hosp_2_loss",
            heldout_acc_col="test_hosp_2_acc",
            paths={
                1.10: art / "guardrail_merged_rows_camelyon_erm_p95_ratio110_20260327.csv",
                1.25: art / "guardrail_merged_rows_camelyon_erm_p95_ratio125_20260326.csv",
                1.50: art / "guardrail_merged_rows_camelyon_erm_p95_ratio150_20260326.csv",
            },
        ),
        SweepSpec(
            suite="camelyon_finetune_p95",
            label="Camelyon17 Finetune",
            heldout_loss_col="test_hosp_2_loss",
            heldout_acc_col="test_hosp_2_acc",
            paths={
                1.10: art / "guardrail_merged_rows_camelyon_finetune_p95_ratio110_20260327.csv",
                1.25: art / "guardrail_merged_rows_camelyon_finetune_p95_ratio125_20260326.csv",
                1.50: art / "guardrail_merged_rows_camelyon_finetune_p95_ratio150_20260326.csv",
            },
        ),
    ]


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    subset = [c for c in ["suite", "target_regime", "selection_policy", "seed", "regime", "epoch", "tag"] if c in df.columns]
    if not subset:
        return df.copy()
    return df.drop_duplicates(subset=subset).copy()


def _bootstrap_ci(arr: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    rows = []
    for spec in _specs(_repo_root()):
        for budget, path in spec.paths.items():
            df = _dedup(pd.read_csv(path))
            sub = df[df["suite"] == spec.suite].copy()
            piv = {}
            for policy in ["baseline", "proxy_only", "guardrail"]:
                tmp = sub[sub["selection_policy"] == policy].drop_duplicates(subset=["seed"]).set_index("seed")
                piv[policy] = tmp
            seeds = sorted(set(piv["baseline"].index) & set(piv["proxy_only"].index) & set(piv["guardrail"].index))
            if not seeds:
                continue
            base = piv["baseline"].loc[seeds]
            proxy = piv["proxy_only"].loc[seeds]
            guard = piv["guardrail"].loc[seeds]
            metrics = {
                "heldout_loss_reduction": pd.to_numeric(proxy[spec.heldout_loss_col] - guard[spec.heldout_loss_col], errors="coerce").to_numpy(),
                "tail_reduction": pd.to_numeric(proxy["selected_tail_worst_cvar"] - guard["selected_tail_worst_cvar"], errors="coerce").to_numpy(),
                "accuracy_change": pd.to_numeric(guard[spec.heldout_acc_col] - proxy[spec.heldout_acc_col], errors="coerce").to_numpy(),
                "proxy_metric_change": pd.to_numeric(guard["chosen_proxy_metric"] - proxy["chosen_proxy_metric"], errors="coerce").to_numpy(),
                "accuracy_vs_baseline": pd.to_numeric(guard[spec.heldout_acc_col] - base[spec.heldout_acc_col], errors="coerce").to_numpy(),
            }
            for metric_name, values in metrics.items():
                mean, lo, hi = _bootstrap_ci(values, seed=int(round(budget * 1000)))
                rows.append(
                    {
                        "suite_label": spec.label,
                        "budget_ratio": budget,
                        "metric": metric_name,
                        "n": int(np.isfinite(values).sum()),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    lines = [
        r"\begin{tabular}{llccc}",
        r"  \toprule",
        r"  Suite & Budget & Loss reduction & Tail reduction & Acc. change \\",
        r"  \midrule",
    ]
    for suite in ["Camelyon17 ERM", "Camelyon17 Finetune"]:
        for budget in [1.10, 1.25, 1.50]:
            sub = out[(out["suite_label"] == suite) & (out["budget_ratio"] == budget)]
            if sub.empty:
                continue
            loss = sub[sub["metric"] == "heldout_loss_reduction"].iloc[0]
            tail = sub[sub["metric"] == "tail_reduction"].iloc[0]
            acc = sub[sub["metric"] == "accuracy_change"].iloc[0]
            lines.append(
                "  {suite} & {budget:.2f}x & {lmean:+.3f} [{llo:+.3f},{lhi:+.3f}] & {tmean:+.3f} [{tlo:+.3f},{thi:+.3f}] & {amean:+.3f} [{alo:+.3f},{ahi:+.3f}] \\\\".format(
                    suite=suite,
                    budget=budget,
                    lmean=loss["mean"],
                    llo=loss["ci_low"],
                    lhi=loss["ci_high"],
                    tmean=tail["mean"],
                    tlo=tail["ci_low"],
                    thi=tail["ci_high"],
                    amean=acc["mean"],
                    alo=acc["ci_low"],
                    ahi=acc["ci_high"],
                )
            )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
