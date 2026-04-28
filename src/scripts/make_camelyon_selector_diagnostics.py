from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"


@dataclass(frozen=True)
class Suite:
    key: str
    label: str
    seed_table: Path
    selector_rows: Path
    selector_eval_rows: Path
    phase1_csv: Path
    selector_calibration_rows: Path
    calibration_prefix: str


SUITES = [
    Suite(
        key="camelyon_erm",
        label="Camelyon17 ERM",
        seed_table=ART / "camelyon_erm_stepup_seed_table_20260328.csv",
        selector_rows=ART / "camelyon_erm_p95_selector_rows_trueval_20260329.csv",
        selector_eval_rows=ART / "camelyon_erm_p95_selector_eval_rows_trueval_20260329.csv",
        phase1_csv=ART / "camelyon17_resnet50_phase1_pockets_v11erm_softclip_cam_10s_fix_20260228.csv",
        selector_calibration_rows=ART / "camelyon_erm_selector_calibration_rows_20260424.csv",
        calibration_prefix="test",
    ),
    Suite(
        key="camelyon_finetune",
        label="Camelyon17 Finetune",
        seed_table=ART / "camelyon_finetune_stepup_seed_table_20260328.csv",
        selector_rows=ART / "camelyon_finetune_p95_selector_rows_trueval_20260329.csv",
        selector_eval_rows=ART / "camelyon_finetune_p95_selector_eval_rows_trueval_20260329.csv",
        phase1_csv=ART / "camelyon17_resnet50_phase1_pockets_finetune_cam_scivalid10s_20260326.csv",
        selector_calibration_rows=ART / "camelyon_finetune_selector_calibration_rows_20260424.csv",
        calibration_prefix="val",
    ),
]


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
    boot = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _metric_record(
    *,
    suite: str,
    contrast: str,
    metric: str,
    values: np.ndarray,
    seed: int,
    worse_when: str,
) -> dict[str, object]:
    mean, lo, hi = _bootstrap_ci(values, seed=seed)
    if worse_when == "positive":
        worse_count = int(np.sum(np.asarray(values, dtype=np.float64) > 0))
    elif worse_when == "negative":
        worse_count = int(np.sum(np.asarray(values, dtype=np.float64) < 0))
    else:
        raise ValueError(f"unknown worse_when={worse_when}")
    return {
        "suite": suite,
        "contrast": contrast,
        "metric": metric,
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(np.isfinite(values).sum()),
        "worse_count": worse_count,
    }


def build_standard_triangulation() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs_by_suite = {
        "camelyon_erm": [
            ("Loss", "delta_test_hosp_2_loss_soft_minus_base", "positive"),
            ("Acc", "delta_test_hosp_2_acc_soft_minus_base", "negative"),
            ("ECE", "delta_test_ece_soft_minus_base", "positive"),
            ("Brier", "delta_test_brier_soft_minus_base", "positive"),
            ("Teacher-tail CVaR", "delta_worst_cell_cvar_soft_minus_base", "positive"),
        ],
        "camelyon_finetune": [
            ("Loss", "delta_test_hosp_2_loss_soft_minus_base", "positive"),
            ("Acc", "delta_test_hosp_2_acc_soft_minus_base", "negative"),
            ("ECE", "delta_val_ece_soft_minus_base", "positive"),
            ("Brier", "delta_val_brier_soft_minus_base", "positive"),
            ("Teacher-tail CVaR", "delta_worst_cell_cvar_soft_minus_base", "positive"),
        ],
    }
    for s_idx, suite in enumerate(SUITES):
        df = _read(suite.seed_table)
        for m_idx, (metric, col, worse_when) in enumerate(specs_by_suite[suite.key]):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            rows.append(
                _metric_record(
                    suite=suite.label,
                    contrast="workflow: proxy - baseline",
                    metric=metric,
                    values=vals,
                    seed=5100 + 10 * s_idx + m_idx,
                    worse_when=worse_when,
                )
            )
    return pd.DataFrame(rows)


def _attach_tail_and_calibration(suite: Suite, selector_rows: pd.DataFrame) -> pd.DataFrame:
    phase1 = _read(suite.phase1_csv)
    phase1 = phase1[(phase1["split"] == "val") & (phase1["family"] == "teacher_difficulty")].copy()
    keys = [c for c in ["regime", "seed", "tag", "epoch"] if c in phase1.columns and c in selector_rows.columns]
    tail = (
        phase1.groupby(keys, as_index=False)[["worst_cell_cvar"]]
        .mean(numeric_only=True)
        .rename(columns={"worst_cell_cvar": "teacher_tail_cvar"})
    )
    merged = selector_rows.merge(tail, on=keys, how="left")

    cal = _read(suite.selector_calibration_rows)
    cal_cols = [f"{suite.calibration_prefix}_ece", f"{suite.calibration_prefix}_brier"]
    cal_cols = [c for c in cal_cols if c in cal.columns]
    cal = cal[keys + cal_cols].drop_duplicates(keys)
    return merged.merge(cal, on=keys, how="left")


def build_selector_only() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for s_idx, suite in enumerate(SUITES):
        selector_rows = _read(suite.selector_rows)
        eval_rows = _read(suite.selector_eval_rows)
        selector_rows = selector_rows[selector_rows["selection_policy"].isin(["proxy_only", "val_loss_only"])].copy()
        selector_rows = selector_rows[selector_rows["fallback_to_baseline"].astype(str).str.lower() != "true"].copy()
        keys = [c for c in ["regime", "seed", "tag", "epoch"] if c in selector_rows.columns and c in eval_rows.columns]
        merged = selector_rows.merge(eval_rows.drop_duplicates(keys), on=keys, how="left", suffixes=("", "_eval"))
        merged = _attach_tail_and_calibration(suite, merged)

        ece_col = f"{suite.calibration_prefix}_ece"
        brier_col = f"{suite.calibration_prefix}_brier"
        per_seed: list[dict[str, object]] = []
        for seed, sub in merged.groupby("seed"):
            piv = sub.set_index("selection_policy")
            if "proxy_only" not in piv.index or "val_loss_only" not in piv.index:
                continue
            proxy = piv.loc["proxy_only"]
            val_loss = piv.loc["val_loss_only"]
            per_seed.append(
                {
                    "suite": suite.label,
                    "seed": int(seed),
                    "proxy_epoch": int(proxy["epoch"]),
                    "val_loss_epoch": int(val_loss["epoch"]),
                    "delta_loss": float(proxy["test_hosp_2_loss"] - val_loss["test_hosp_2_loss"]),
                    "delta_acc": float(proxy["test_hosp_2_acc"] - val_loss["test_hosp_2_acc"]),
                    "delta_ece": float(proxy[ece_col] - val_loss[ece_col]),
                    "delta_brier": float(proxy[brier_col] - val_loss[brier_col]),
                    "delta_teacher_tail_cvar": float(proxy["teacher_tail_cvar"] - val_loss["teacher_tail_cvar"]),
                }
            )
        seed_df = pd.DataFrame(per_seed)
        seed_rows.extend(per_seed)
        metric_specs = [
            ("Loss", "delta_loss", "positive"),
            ("Acc", "delta_acc", "negative"),
            ("ECE", "delta_ece", "positive"),
            ("Brier", "delta_brier", "positive"),
            ("Teacher-tail CVaR", "delta_teacher_tail_cvar", "positive"),
        ]
        for m_idx, (metric, col, worse_when) in enumerate(metric_specs):
            vals = pd.to_numeric(seed_df[col], errors="coerce").to_numpy(dtype=np.float64)
            summary_rows.append(
                _metric_record(
                    suite=suite.label,
                    contrast="same trajectory: proxy - val-loss",
                    metric=metric,
                    values=vals,
                    seed=6100 + 10 * s_idx + m_idx,
                    worse_when=worse_when,
                )
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(seed_rows)


def _fmt(metric: str, mean: float, lo: float, hi: float) -> str:
    if metric in {"ECE", "Brier"}:
        return f"{mean:+.4f} [{lo:+.4f},{hi:+.4f}]"
    if metric == "Teacher-tail CVaR":
        return f"{mean:+.2f} [{lo:+.2f},{hi:+.2f}]"
    return f"{mean:+.3f} [{lo:+.3f},{hi:+.3f}]"


def _cell(summary: pd.DataFrame, suite: str, contrast: str, metric: str) -> tuple[str, int, int]:
    row = summary[(summary["suite"] == suite) & (summary["contrast"] == contrast) & (summary["metric"] == metric)]
    if row.empty:
        raise ValueError(f"missing row: {suite}/{contrast}/{metric}")
    r = row.iloc[0]
    return _fmt(metric, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])), int(r["worse_count"]), int(r["n"])


def _write_tex(summary: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}llrrrrrl@{}}",
        r"  \toprule",
        r"  Suite & Contrast & $\Delta$Loss & $\Delta$Acc & $\Delta$ECE & $\Delta$Brier & $\Delta$Tail & Harm counts \\",
        r"  \midrule",
    ]
    contrasts = ["workflow: proxy - baseline", "same trajectory: proxy - val-loss"]
    for contrast_idx, contrast in enumerate(contrasts):
        if contrast_idx == 1:
            lines.append(r"  \midrule")
        for suite in [s.label for s in SUITES]:
            loss, loss_bad, n = _cell(summary, suite, contrast, "Loss")
            acc, _, _ = _cell(summary, suite, contrast, "Acc")
            ece, ece_bad, _ = _cell(summary, suite, contrast, "ECE")
            brier, brier_bad, _ = _cell(summary, suite, contrast, "Brier")
            tail, tail_bad, _ = _cell(summary, suite, contrast, "Teacher-tail CVaR")
            counts = f"loss/ECE/tail {loss_bad}/{n},{ece_bad}/{n},{tail_bad}/{n}; Brier {brier_bad}/{n}"
            lines.append(
                f"  {suite} & {contrast} & {loss} & {acc} & {ece} & {brier} & {tail} & {counts} \\\\"
            )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    standard = build_standard_triangulation()
    selector, selector_seeds = build_selector_only()
    combined = pd.concat([standard, selector], ignore_index=True)

    standard_out = ART / "camelyon_standard_metric_triangulation_20260424.csv"
    selector_out = ART / "camelyon_same_trajectory_selector_contrasts_20260424.csv"
    selector_seed_out = ART / "camelyon_same_trajectory_selector_seed_deltas_20260424.csv"
    table_out = TABLES / "table_camelyon_selector_diagnostics.tex"

    standard.to_csv(standard_out, index=False)
    selector.to_csv(selector_out, index=False)
    selector_seeds.to_csv(selector_seed_out, index=False)
    _write_tex(combined, table_out)

    print(f"[ok] wrote {standard_out}")
    print(f"[ok] wrote {selector_out}")
    print(f"[ok] wrote {selector_seed_out}")
    print(f"[ok] wrote {table_out}")


if __name__ == "__main__":
    main()
