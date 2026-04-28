from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"

SELECTED_ROWS = ART / "camelyon_loo_selector_standard_metrics_selected_rows_20260424.csv"
EVAL_ROWS = ART / "camelyon_loo_selector_standard_metrics_eval_rows_20260424.csv"

SEED_DELTAS_OUT = ART / "camelyon_loo_selector_standard_metrics_seed_deltas_20260424.csv"
SEEDFOLD_SUMMARY_OUT = ART / "camelyon_loo_selector_standard_metrics_seedfold_summary_20260424.csv"
FOLD_SUMMARY_OUT = ART / "camelyon_loo_selector_standard_metrics_fold_summary_20260424.csv"
FOLD_POOLED_SUMMARY_OUT = ART / "camelyon_loo_selector_standard_metrics_fold_pooled_summary_20260424.csv"
TABLE_OUT = TABLES / "table_camelyon_loo_selector_standard_metrics.tex"

LEVELS = ["P95", "P97", "P99"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_seed_deltas() -> pd.DataFrame:
    selected = _read(SELECTED_ROWS)
    eval_rows = _read(EVAL_ROWS)
    keys = ["fold", "holdout_hospital", "regime", "seed", "tag", "epoch"]
    merged = selected.merge(eval_rows.drop_duplicates(keys), on=keys, how="left")
    missing = [c for c in ["loss", "acc", "brier", "ece"] if c not in merged.columns or merged[c].isna().any()]
    if missing:
        raise RuntimeError(f"Missing eval metrics after merge: {missing}")

    rows: list[dict[str, object]] = []
    for (fold, h, seed), sub in merged.groupby(["fold", "holdout_hospital", "seed"]):
        base = sub[sub["selection_policy"] == "baseline"]
        if base.empty:
            continue
        base = base.iloc[0]
        for level in LEVELS:
            level_sub = sub[sub["level"] == level]
            piv = level_sub.set_index("selection_policy")
            if "proxy_only" not in piv.index:
                continue
            proxy = piv.loc["proxy_only"]
            for contrast, other_policy in [
                ("workflow: proxy - baseline", "baseline"),
                ("same trajectory: proxy - val-loss", "val_loss_only"),
            ]:
                if other_policy == "baseline":
                    other = base
                elif other_policy in piv.index:
                    other = piv.loc[other_policy]
                else:
                    continue
                rows.append(
                    {
                        "fold": fold,
                        "holdout_hospital": int(h),
                        "seed": int(seed),
                        "level": level,
                        "contrast": contrast,
                        "proxy_epoch": int(proxy["epoch"]),
                        "other_epoch": int(other["epoch"]),
                        "delta_proxy_metric": float(proxy["chosen_proxy_metric"] - other["chosen_proxy_metric"]),
                        "delta_val_overall_loss": float(
                            proxy["chosen_val_overall_loss"] - other["chosen_val_overall_loss"]
                        ),
                        "delta_loss": float(proxy["loss"] - other["loss"]),
                        "delta_acc": float(proxy["acc"] - other["acc"]),
                        "delta_brier": float(proxy["brier"] - other["brier"]),
                        "delta_ece": float(proxy["ece"] - other["ece"]),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["contrast", "level", "holdout_hospital", "seed"]).reset_index(drop=True)
    expected = 5 * 10 * len(LEVELS) * 2
    if out.shape[0] != expected:
        raise RuntimeError(f"Expected {expected} seed-delta rows, found {out.shape[0]}")
    return out


def summarize(seed_deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_specs = [
        ("Loss", "delta_loss", "positive"),
        ("Acc", "delta_acc", "negative"),
        ("Brier", "delta_brier", "positive"),
        ("ECE", "delta_ece", "positive"),
    ]
    seed_summary_rows: list[dict[str, object]] = []
    for (contrast, level), sub in seed_deltas.groupby(["contrast", "level"]):
        for m_idx, (metric, col, harmful) in enumerate(metric_specs):
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            mean, lo, hi = _bootstrap_ci(vals, seed=8100 + 17 * m_idx)
            harm = int(np.sum(vals > 0)) if harmful == "positive" else int(np.sum(vals < 0))
            seed_summary_rows.append(
                {
                    "contrast": contrast,
                    "level": level,
                    "metric": metric,
                    "unit": "seed-fold",
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": int(np.isfinite(vals).sum()),
                    "harm_count": harm,
                    "harmful_direction": harmful,
                }
            )
    seed_summary = pd.DataFrame(seed_summary_rows)

    fold_rows: list[dict[str, object]] = []
    for (contrast, level, fold, h), sub in seed_deltas.groupby(["contrast", "level", "fold", "holdout_hospital"]):
        rec: dict[str, object] = {
            "contrast": contrast,
            "level": level,
            "fold": fold,
            "holdout_hospital": int(h),
            "n_seeds": int(sub.shape[0]),
        }
        for _, col, _harmful in metric_specs:
            rec[col] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        fold_rows.append(rec)
    fold_summary = pd.DataFrame(fold_rows).sort_values(["contrast", "level", "holdout_hospital"]).reset_index(drop=True)

    fold_pooled_rows: list[dict[str, object]] = []
    for (contrast, level), sub in fold_summary.groupby(["contrast", "level"]):
        for m_idx, (metric, col, harmful) in enumerate(metric_specs):
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            mean, lo, hi = _bootstrap_ci(vals, seed=9100 + 17 * m_idx, n_boot=20_000)
            harm = int(np.sum(vals > 0)) if harmful == "positive" else int(np.sum(vals < 0))
            fold_pooled_rows.append(
                {
                    "contrast": contrast,
                    "level": level,
                    "metric": metric,
                    "unit": "fold",
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": int(np.isfinite(vals).sum()),
                    "harm_count": harm,
                    "harmful_direction": harmful,
                }
            )
    fold_pooled = pd.DataFrame(fold_pooled_rows)
    return seed_summary, fold_summary, fold_pooled


def _fmt_mean(metric: str, mean: float) -> str:
    if metric in {"ECE", "Brier", "Acc"}:
        return f"{mean:+.4f}"
    return f"{mean:+.3f}"


def _cell(summary: pd.DataFrame, level: str, metric: str) -> tuple[str, int, int]:
    row = summary[
        (summary["contrast"] == "same trajectory: proxy - val-loss")
        & (summary["level"] == level)
        & (summary["metric"] == metric)
        & (summary["unit"] == "fold")
    ]
    if row.empty:
        raise ValueError(f"Missing fold summary for {level}/{metric}")
    r = row.iloc[0]
    return _fmt_mean(metric, float(r["mean"])), int(r["harm_count"]), int(r["n"])


def write_tex(fold_pooled: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrrrl@{}}",
        r"  \toprule",
        r"  Regime & $\Delta$Loss & $\Delta$Acc & $\Delta$Brier & $\Delta$ECE & Harmful folds \\",
        r"  \midrule",
    ]
    for level in LEVELS:
        loss, loss_bad, n = _cell(fold_pooled, level, "Loss")
        acc, acc_bad, _ = _cell(fold_pooled, level, "Acc")
        brier, brier_bad, _ = _cell(fold_pooled, level, "Brier")
        ece, ece_bad, _ = _cell(fold_pooled, level, "ECE")
        counts = f"loss/Brier {loss_bad}/{n},{brier_bad}/{n}; acc/ECE {acc_bad}/{n},{ece_bad}/{n}"
        lines.append(f"  {level} & {loss} & {acc} & {brier} & {ece} & {counts} \\\\")
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    seed_deltas = build_seed_deltas()
    seed_summary, fold_summary, fold_pooled = summarize(seed_deltas)
    seed_deltas.to_csv(SEED_DELTAS_OUT, index=False)
    seed_summary.to_csv(SEEDFOLD_SUMMARY_OUT, index=False)
    fold_summary.to_csv(FOLD_SUMMARY_OUT, index=False)
    fold_pooled.to_csv(FOLD_POOLED_SUMMARY_OUT, index=False)
    write_tex(fold_pooled)
    print(f"[ok] wrote {SEED_DELTAS_OUT}")
    print(f"[ok] wrote {SEEDFOLD_SUMMARY_OUT}")
    print(f"[ok] wrote {FOLD_SUMMARY_OUT}")
    print(f"[ok] wrote {FOLD_POOLED_SUMMARY_OUT}")
    print(f"[ok] wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
