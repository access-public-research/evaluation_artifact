import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


CONFIG = "configs/base_v27_erm_softclip_civilcomments_10seeds.yaml"
DATASET = "civilcomments"
TARGET_REGIME = "erm_softclip_p95_a10"
PROXY_FAMILY = "global_hash"
SUITE = "civilcomments_erm_p95"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _aggregate_phase0(phase0_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(phase0_csv)
    req = {
        "regime",
        "seed",
        "tag",
        "epoch",
        "family",
        "proxy_worst_loss_min",
        "proxy_worst_loss_clip_min",
        "val_overall_loss",
        "val_overall_acc",
    }
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{phase0_csv} missing required columns: {sorted(miss)}")
    df = df[(df["regime"] == TARGET_REGIME) & (df["family"] == PROXY_FAMILY)].copy()
    if df.empty:
        raise ValueError(f"No phase0 rows for regime={TARGET_REGIME} and family={PROXY_FAMILY}")
    return (
        df.groupby(["regime", "seed", "tag", "epoch"], as_index=False)
        .agg(
            {
                "proxy_worst_loss_min": "mean",
                "proxy_worst_loss_clip_min": "mean",
                "val_overall_loss": "mean",
                "val_overall_acc": "mean",
            }
        )
        .sort_values(["seed", "epoch"])
        .reset_index(drop=True)
    )


def _build_val_loss_rows(phase0_csv: Path, guardrail_csv: Path) -> pd.DataFrame:
    phase0 = _aggregate_phase0(phase0_csv)
    guard = pd.read_csv(guardrail_csv)
    baseline_rows = (
        guard[guard["selection_policy"] == "baseline"]
        .drop_duplicates(subset=["seed"])
        .set_index("seed")
    )
    rows = []
    for seed in sorted(baseline_rows.index.astype(int).tolist()):
        sub = phase0[phase0["seed"] == seed].copy()
        if sub.empty:
            continue
        pick = sub.sort_values(
            ["val_overall_loss", "proxy_worst_loss_clip_min", "epoch"],
            ascending=[True, True, True],
        ).iloc[0]
        base = baseline_rows.loc[seed]
        rows.append(
            {
                "suite": SUITE,
                "target_regime": TARGET_REGIME,
                "baseline_regime": str(base["baseline_regime"]),
                "seed": int(seed),
                "baseline_epoch": int(base["baseline_epoch"]),
                "baseline_tag": str(base["baseline_tag"]),
                "baseline_val_overall_loss": float(base["baseline_val_overall_loss"]),
                "baseline_proxy_metric": float(base["baseline_proxy_metric"]),
                "budget_mode": "none",
                "budget_value": float("nan"),
                "budget_threshold": float("nan"),
                "selection_policy": "val_loss_only",
                "regime": TARGET_REGIME,
                "tag": str(pick["tag"]),
                "epoch": int(pick["epoch"]),
                "chosen_val_overall_loss": float(pick["val_overall_loss"]),
                "chosen_val_overall_acc": float(pick["val_overall_acc"]),
                "chosen_proxy_metric": float(pick["proxy_worst_loss_clip_min"]),
                "chosen_proxy_metric_col": "proxy_worst_loss_clip_min",
                "fallback_to_baseline": False,
                "feasible_target_epochs": int(sub.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_eval_outputs(root: Path, selected_rows_csv: Path, out_test_rows: Path, out_test_summary: Path, out_cal_rows: Path, out_cal_summary: Path) -> None:
    if not out_test_rows.exists() or not out_test_summary.exists():
        _run(
            [
                sys.executable,
                "-m",
                "src.scripts.evaluate_selected_civilcomments_test",
                "--config",
                CONFIG,
                "--dataset",
                DATASET,
                "--regimes",
                TARGET_REGIME,
                "--metrics_suffix",
                "civilcomments_erm_softclip_10s_20260328",
                "--selected_rows_csv",
                str(selected_rows_csv),
                "--out_rows",
                str(out_test_rows),
                "--out_summary",
                str(out_test_summary),
            ],
            cwd=root,
        )
    if not out_cal_rows.exists() or not out_cal_summary.exists():
        _run(
            [
                sys.executable,
                "-m",
                "src.scripts.evaluate_selected_civilcomments_calibration",
                "--config",
                CONFIG,
                "--dataset",
                DATASET,
                "--regimes",
                TARGET_REGIME,
                "--metrics_suffix",
                "civilcomments_erm_softclip_10s_20260328",
                "--selected_rows_csv",
                str(selected_rows_csv),
                "--out_rows",
                str(out_cal_rows),
                "--out_summary",
                str(out_cal_summary),
            ],
            cwd=root,
        )


def _policy_label(policy: str) -> str:
    return {
        "baseline": "Baseline",
        "proxy_only": "Proxy-only",
        "val_loss_only": "Validation-loss",
        "guardrail": "1.25x guardrail",
    }[policy]


def _write_tex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"  \toprule",
        r"  Selector & Admit rate & $\Delta$Overall loss$\downarrow$ & $\Delta$Overall acc$\uparrow$ & $\Delta$WG loss$\downarrow$ & $\Delta$ECE$\downarrow$ \\",
        r"  \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "  {selector} & {rate:.2f} & {dloss:+.3f} & {dacc:+.3f} & {dwg:+.3f} & {dece:+.4f} \\\\".format(
                selector=row["selector_label"],
                rate=row["accept_rate"],
                dloss=row["delta_overall_loss"],
                dacc=row["delta_overall_acc"],
                dwg=row["delta_wg_loss"],
                dece=row["delta_ece"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrail_csv", default="artifacts/metrics/guardrail_merged_rows_civilcomments_erm_p95_ratio125_20260328.csv")
    ap.add_argument("--phase0_csv", default="artifacts/metrics/civilcomments_distilbert-base-uncased_phase0_val_metrics_civilcomments_erm_softclip_10s_20260328.csv")
    ap.add_argument("--existing_cal_rows", default="artifacts/metrics/civilcomments_selected_calibration_p95_20260328_rows.csv")
    ap.add_argument("--out_selected_rows", default="artifacts/metrics/civilcomments_selector_val_loss_rows_20260331.csv")
    ap.add_argument("--out_test_rows", default="artifacts/metrics/civilcomments_selector_val_loss_test_rows_20260331.csv")
    ap.add_argument("--out_test_summary", default="artifacts/metrics/civilcomments_selector_val_loss_test_summary_20260331.csv")
    ap.add_argument("--out_cal_rows", default="artifacts/metrics/civilcomments_selector_val_loss_calibration_rows_20260331.csv")
    ap.add_argument("--out_cal_summary", default="artifacts/metrics/civilcomments_selector_val_loss_calibration_summary_20260331.csv")
    ap.add_argument("--out_rows", default="artifacts/metrics/civilcomments_selector_comparison_rows_20260331.csv")
    ap.add_argument("--out_summary", default="artifacts/metrics/civilcomments_selector_comparison_summary_20260331.csv")
    ap.add_argument("--out_tex", default="paper/neurips2026_selection_risk/tables/table_civilcomments_selector_comparison.tex")
    args = ap.parse_args()

    root = _repo_root()
    guardrail_csv = root / args.guardrail_csv
    phase0_csv = root / args.phase0_csv
    existing_cal_rows = root / args.existing_cal_rows
    out_selected_rows = root / args.out_selected_rows
    out_test_rows = root / args.out_test_rows
    out_test_summary = root / args.out_test_summary
    out_cal_rows = root / args.out_cal_rows
    out_cal_summary = root / args.out_cal_summary
    out_rows = root / args.out_rows
    out_summary = root / args.out_summary
    out_tex = root / args.out_tex

    if out_summary.exists():
        summary = pd.read_csv(out_summary)
        _write_tex(summary, out_tex)
        print(f"[ok] wrote {out_tex} from {out_summary}")
        return

    val_rows = _build_val_loss_rows(phase0_csv=phase0_csv, guardrail_csv=guardrail_csv)
    out_selected_rows.parent.mkdir(parents=True, exist_ok=True)
    val_rows.to_csv(out_selected_rows, index=False)

    _ensure_eval_outputs(
        root=root,
        selected_rows_csv=out_selected_rows,
        out_test_rows=out_test_rows,
        out_test_summary=out_test_summary,
        out_cal_rows=out_cal_rows,
        out_cal_summary=out_cal_summary,
    )

    guard = pd.read_csv(guardrail_csv)
    guard = guard[guard["selection_policy"].isin(["baseline", "proxy_only", "guardrail"])].drop_duplicates(
        subset=["selection_policy", "seed", "regime", "epoch", "tag"]
    )
    existing_cal = pd.read_csv(existing_cal_rows)
    guard = guard.merge(
        existing_cal[
            [
                "regime",
                "seed",
                "tag",
                "epoch",
                "test_brier",
                "test_ece",
                "test_wilds_wg_brier",
                "test_wilds_wg_ece",
            ]
        ].drop_duplicates(),
        on=["regime", "seed", "tag", "epoch"],
        how="left",
    )

    val_test = pd.read_csv(out_test_rows)
    val_cal = pd.read_csv(out_cal_rows)
    val = val_rows.merge(val_test, on=["regime", "seed", "tag", "epoch"], how="left")
    val = val.merge(
        val_cal[
            [
                "regime",
                "seed",
                "tag",
                "epoch",
                "test_brier",
                "test_ece",
                "test_wilds_wg_brier",
                "test_wilds_wg_ece",
            ]
        ],
        on=["regime", "seed", "tag", "epoch"],
        how="left",
    )

    merged = pd.concat([guard, val], ignore_index=True, sort=False)
    merged.to_csv(out_rows, index=False)

    base = merged[merged["selection_policy"] == "baseline"].drop_duplicates(subset=["seed"]).set_index("seed")
    summary_rows = []
    for policy in ["baseline", "proxy_only", "val_loss_only", "guardrail"]:
        sub = merged[merged["selection_policy"] == policy].drop_duplicates(subset=["seed"]).set_index("seed")
        seeds = sorted(set(base.index) & set(sub.index))
        base_sub = base.loc[seeds]
        sub = sub.loc[seeds]
        summary_rows.append(
            {
                "selection_policy": policy,
                "selector_label": _policy_label(policy),
                "n": int(len(seeds)),
                "accept_rate": float(1.0 - sub["fallback_to_baseline"].astype(float).mean()),
                "delta_overall_loss": float((sub["test_overall_loss"] - base_sub["test_overall_loss"]).mean()),
                "delta_overall_acc": float((sub["test_overall_acc"] - base_sub["test_overall_acc"]).mean()),
                "delta_wg_loss": float((sub["test_wilds_wg_loss"] - base_sub["test_wilds_wg_loss"]).mean()),
                "delta_wg_acc": float((sub["test_wilds_wg_acc"] - base_sub["test_wilds_wg_acc"]).mean()),
                "delta_ece": float((sub["test_ece"] - base_sub["test_ece"]).mean()),
                "delta_brier": float((sub["test_brier"] - base_sub["test_brier"]).mean()),
                "delta_wg_ece": float((sub["test_wilds_wg_ece"] - base_sub["test_wilds_wg_ece"]).mean()),
                "delta_wg_brier": float((sub["test_wilds_wg_brier"] - base_sub["test_wilds_wg_brier"]).mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_summary, index=False)
    _write_tex(summary, out_tex)
    print(f"[ok] wrote {out_selected_rows}")
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()
