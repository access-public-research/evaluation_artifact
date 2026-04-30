from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROWS_CSV = ROOT / "artifacts/metrics/camelyon17_gce_q07_10s_selector_analysis_20260430.csv"
SUMMARY_CSV = ROOT / "artifacts/metrics/camelyon17_gce_q07_10s_selector_summary_20260430.csv"
OUT_TEX = ROOT / "paper/neurips2026_selection_risk/tables/table_gce_q07_extension.tex"


CONTRASTS = [
    (
        "GCE proxy-best $-$ ERM baseline-selected",
        "gce_proxy_selected",
        "baseline_selected",
        "14.4 / 3.8",
    ),
    (
        "GCE proxy-best $-$ GCE val-loss-best",
        "gce_proxy_selected",
        "gce_val_loss_selected",
        "14.4 / 3.3",
    ),
    (
        "GCE fixed epoch 30 $-$ ERM fixed epoch 30",
        "gce_fixed30",
        "baseline_fixed30",
        "30 / 30",
    ),
]


def _paired_ci(values: np.ndarray, *, seed: int = 42, draws: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.shape[0], size=(draws, values.shape[0]))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fmt_delta(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"


def _fmt_ci(mean: float, lo: float, hi: float, digits: int = 4) -> str:
    return f"{mean:+.{digits}f} [{lo:+.{digits}f},{hi:+.{digits}f}]"


def main() -> None:
    df = pd.read_csv(ROWS_CSV)
    rows = []
    for label, a_name, b_name, epochs in CONTRASTS:
        a = df[df["condition"].eq(a_name)].sort_values("seed").reset_index(drop=True)
        b = df[df["condition"].eq(b_name)].sort_values("seed").reset_index(drop=True)
        if a.empty or b.empty:
            raise ValueError(f"Missing contrast rows for {a_name} vs {b_name}")
        if list(a["seed"]) != list(b["seed"]):
            raise ValueError(f"Seed mismatch for {a_name} vs {b_name}")

        record = {
            "contrast": label,
            "condition_a": a_name,
            "condition_b": b_name,
            "n": int(a.shape[0]),
            "mean_epoch_a": float(a["epoch"].mean()),
            "mean_epoch_b": float(b["epoch"].mean()),
            "epochs_display": epochs,
            "mean_rw_a": float(a["rw_tail_core"].mean()),
        }
        for col in ["gce_proxy", "test_acc", "test_loss", "test_ece", "tail_worst_cvar"]:
            delta = pd.to_numeric(a[col], errors="raise").to_numpy(dtype=np.float64) - pd.to_numeric(
                b[col], errors="raise"
            ).to_numpy(dtype=np.float64)
            lo, hi = _paired_ci(delta)
            record[f"delta_{col}_mean"] = float(delta.mean())
            record[f"delta_{col}_ci_low"] = lo
            record[f"delta_{col}_ci_high"] = hi
            record[f"delta_{col}_positive_count"] = int((delta > 0.0).sum())
            record[f"delta_{col}_negative_count"] = int((delta < 0.0).sum())
        rows.append(record)

    summary = pd.DataFrame(rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    tex_lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Contrast & Epochs & $\Delta$GCE proxy & $\Delta$Acc & $\Delta$Loss & $\Delta$ECE & $\Delta$Tail & $R_w$ \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        tex_lines.append(
            " & ".join(
                [
                    str(row["contrast"]),
                    str(row["epochs_display"]),
                    _fmt_delta(float(row["delta_gce_proxy_mean"]), 4),
                    _fmt_delta(float(row["delta_test_acc_mean"]), 4),
                    _fmt_ci(
                        float(row["delta_test_loss_mean"]),
                        float(row["delta_test_loss_ci_low"]),
                        float(row["delta_test_loss_ci_high"]),
                        4,
                    ),
                    _fmt_ci(
                        float(row["delta_test_ece_mean"]),
                        float(row["delta_test_ece_ci_low"]),
                        float(row["delta_test_ece_ci_high"]),
                        4,
                    ),
                    _fmt_ci(
                        float(row["delta_tail_worst_cvar_mean"]),
                        float(row["delta_tail_worst_cvar_ci_low"]),
                        float(row["delta_tail_worst_cvar_ci_high"]),
                        2,
                    ),
                    f"{float(row['mean_rw_a']):.3f}",
                ]
            )
            + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"[ok] wrote {SUMMARY_CSV}")
    print(f"[ok] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
