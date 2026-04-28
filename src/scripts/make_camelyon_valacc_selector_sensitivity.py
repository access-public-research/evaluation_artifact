from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"

SELECTED_ROWS = ART / "camelyon_valacc_selector_sensitivity_selected_rows_20260425.csv"
DOMAIN_ROWS = ART / "camelyon_valacc_selector_sensitivity_domain_rows_20260425.csv"
CALIBRATION_ROWS = ART / "camelyon_valacc_selector_sensitivity_calibration_rows_20260425.csv"
PHASE1_ROWS = ART / "camelyon17_resnet50_phase1_pockets_v11erm_softclip_cam_10s_fix_20260228.csv"

SEED_DELTAS_OUT = ART / "camelyon_valacc_selector_sensitivity_seed_deltas_20260425.csv"
SUMMARY_OUT = ART / "camelyon_valacc_selector_sensitivity_summary_20260425.csv"
TABLE_OUT = TABLES / "table_camelyon_valacc_selector_sensitivity.tex"

LEVELS = ["P95", "P97", "P99"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _tail_rows() -> pd.DataFrame:
    phase1 = _read(PHASE1_ROWS)
    phase1 = phase1[(phase1["split"] == "val") & (phase1["family"] == "teacher_difficulty")].copy()
    keys = ["regime", "seed", "tag", "epoch"]
    return (
        phase1.groupby(keys, as_index=False)[["worst_cell_cvar"]]
        .mean(numeric_only=True)
        .rename(columns={"worst_cell_cvar": "teacher_tail_cvar"})
    )


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
    domain = _read(DOMAIN_ROWS)
    calibration = _read(CALIBRATION_ROWS)
    tail = _tail_rows()
    keys = ["regime", "seed", "tag", "epoch"]

    merged = selected.merge(domain.drop_duplicates(keys), on=keys, how="left")
    merged = merged.merge(calibration.drop_duplicates(keys), on=keys, how="left", suffixes=("", "_cal"))
    merged = merged.merge(tail.drop_duplicates(keys), on=keys, how="left")

    required = ["test_hosp_2_loss", "test_hosp_2_acc", "test_ece", "test_brier", "teacher_tail_cvar"]
    missing = [c for c in required if c not in merged.columns or merged[c].isna().any()]
    if missing:
        raise RuntimeError(f"Missing merged metric values: {missing}")

    rows: list[dict[str, object]] = []
    for (level, seed), sub in merged.groupby(["level", "seed"]):
        piv = sub.set_index("selection_policy")
        if "val_loss_only" not in piv.index or "val_acc_only" not in piv.index:
            continue
        acc = piv.loc["val_acc_only"]
        loss = piv.loc["val_loss_only"]
        rows.append(
            {
                "level": level,
                "seed": int(seed),
                "contrast": "same trajectory: val-accuracy - val-loss",
                "val_accuracy_epoch": int(acc["epoch"]),
                "val_loss_epoch": int(loss["epoch"]),
                "delta_validation_accuracy": float(acc["chosen_val_overall_acc"] - loss["chosen_val_overall_acc"]),
                "delta_validation_loss": float(acc["chosen_val_overall_loss"] - loss["chosen_val_overall_loss"]),
                "delta_proxy_metric": float(acc["chosen_proxy_metric"] - loss["chosen_proxy_metric"]),
                "delta_loss": float(acc["test_hosp_2_loss"] - loss["test_hosp_2_loss"]),
                "delta_acc": float(acc["test_hosp_2_acc"] - loss["test_hosp_2_acc"]),
                "delta_ece": float(acc["test_ece"] - loss["test_ece"]),
                "delta_brier": float(acc["test_brier"] - loss["test_brier"]),
                "delta_teacher_tail_cvar": float(acc["teacher_tail_cvar"] - loss["teacher_tail_cvar"]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["level", "seed"]).reset_index(drop=True)
    expected = len(LEVELS) * 10
    if out.shape[0] != expected:
        raise RuntimeError(f"Expected {expected} seed-delta rows, found {out.shape[0]}")
    return out


def summarize(seed_deltas: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Validation accuracy", "delta_validation_accuracy", "positive"),
        ("Loss", "delta_loss", "positive"),
        ("Acc", "delta_acc", "negative"),
        ("ECE", "delta_ece", "positive"),
        ("Brier", "delta_brier", "positive"),
        ("Teacher-tail CVaR", "delta_teacher_tail_cvar", "positive"),
    ]
    rows: list[dict[str, object]] = []
    for level_idx, level in enumerate(LEVELS):
        sub = seed_deltas[seed_deltas["level"] == level]
        for metric_idx, (metric, col, harmful_direction) in enumerate(specs):
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            mean, lo, hi = _bootstrap_ci(vals, seed=11600 + 100 * level_idx + metric_idx)
            if harmful_direction == "positive":
                harm_count = int(np.sum(vals > 0))
            elif harmful_direction == "negative":
                harm_count = int(np.sum(vals < 0))
            else:
                raise ValueError(harmful_direction)
            rows.append(
                {
                    "level": level,
                    "contrast": "same trajectory: val-accuracy - val-loss",
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": int(np.isfinite(vals).sum()),
                    "harm_count": harm_count,
                    "harmful_direction": harmful_direction,
                }
            )
    return pd.DataFrame(rows)


def _fmt(metric: str, mean: float, lo: float, hi: float) -> str:
    if metric in {"ECE", "Brier", "Acc"}:
        return f"{mean:+.4f} [{lo:+.4f},{hi:+.4f}]"
    if metric == "Teacher-tail CVaR":
        return f"{mean:+.2f} [{lo:+.2f},{hi:+.2f}]"
    return f"{mean:+.3f} [{lo:+.3f},{hi:+.3f}]"


def _cell(summary: pd.DataFrame, level: str, metric: str) -> tuple[str, int, int]:
    row = summary[(summary["level"] == level) & (summary["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing summary row for {level}/{metric}")
    r = row.iloc[0]
    return _fmt(metric, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])), int(r["harm_count"]), int(r["n"])


def write_tex(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrrrll@{}}",
        r"  \toprule",
        r"  Regime & $\Delta$Loss & $\Delta$ECE & $\Delta$Brier & $\Delta$Tail & Harm counts & Note \\",
        r"  \midrule",
    ]
    for level in LEVELS:
        loss, loss_bad, n = _cell(summary, level, "Loss")
        ece, ece_bad, _ = _cell(summary, level, "ECE")
        brier, brier_bad, _ = _cell(summary, level, "Brier")
        tail, tail_bad, _ = _cell(summary, level, "Teacher-tail CVaR")
        counts = f"loss/ECE/tail {loss_bad}/{n},{ece_bad}/{n},{tail_bad}/{n}; Brier {brier_bad}/{n}"
        lines.append(f"  {level} & {loss} & {ece} & {brier} & {tail} & {counts} & clean loss/ECE/tail \\\\")
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    seed_deltas = build_seed_deltas()
    summary = summarize(seed_deltas)
    seed_deltas.to_csv(SEED_DELTAS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    write_tex(summary)
    print(f"[ok] wrote {SEED_DELTAS_OUT}")
    print(f"[ok] wrote {SUMMARY_OUT}")
    print(f"[ok] wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
