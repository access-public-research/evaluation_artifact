import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SuiteSpec:
    suite: str
    label: str
    baseline_regime: str
    soft_regime: str
    lure_csv: Path
    heldout_csv: Path
    calibration_csv: Path
    selected_rows_csv: Path
    phase1_csv: Path
    heldout_loss_col: str
    heldout_acc_col: str
    calibration_ece_col: str
    calibration_brier_col: str
    extra_cols: Dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bootstrap_ci(arr: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _suite_specs(root: Path, date_tag: str) -> List[SuiteSpec]:
    art = root / "artifacts" / "metrics"
    return [
        SuiteSpec(
            suite="camelyon_erm",
            label="Camelyon17 ERM",
            baseline_regime="erm",
            soft_regime="erm_softclip_p95_a10_cam",
            lure_csv=art / f"camelyon_erm_seedmatched_proxy_check_{date_tag}.csv",
            heldout_csv=art / "camelyon17_resnet50_domain_acc_with_loss_selected_v11erm_softclip_cam_10s_fix_20260325.csv",
            calibration_csv=art / "camelyon_erm_selected_calibration_p95_20260327_rows.csv",
            selected_rows_csv=art / "camelyon17_selected_rows_v11erm_softclip_cam_10s_fix_20260325.csv",
            phase1_csv=art / "camelyon17_resnet50_phase1_pockets_v11erm_softclip_cam_10s_fix_20260228.csv",
            heldout_loss_col="test_hosp_2_loss",
            heldout_acc_col="test_hosp_2_acc",
            calibration_ece_col="test_ece",
            calibration_brier_col="test_brier",
            extra_cols={},
        ),
        SuiteSpec(
            suite="camelyon_finetune",
            label="Camelyon17 Finetune",
            baseline_regime="rcgdro_finetune",
            soft_regime="rcgdro_softclip_p95_a10_cam_finetune",
            lure_csv=art / f"camelyon_finetune_seedmatched_proxy_check_{date_tag}.csv",
            heldout_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected.csv",
            calibration_csv=art / f"camelyon_finetune_selected_calibration_p95_{date_tag}_rows.csv",
            selected_rows_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected.csv",
            phase1_csv=art / "camelyon17_resnet50_phase1_pockets_finetune_cam_scivalid10s_20260326.csv",
            heldout_loss_col="test_hosp_2_loss",
            heldout_acc_col="test_hosp_2_acc",
            calibration_ece_col="val_ece",
            calibration_brier_col="val_brier",
            extra_cols={},
        ),
        SuiteSpec(
            suite="civilcomments",
            label="CivilComments",
            baseline_regime="erm",
            soft_regime="erm_softclip_p95_a10",
            lure_csv=art / "civilcomments_seedmatched_proxy_check_20260328.csv",
            heldout_csv=art / "civilcomments_test_wilds_selected_rows_civilcomments_erm_softclip_10s_20260328.csv",
            calibration_csv=art / "civilcomments_selected_calibration_p95_20260328_rows.csv",
            selected_rows_csv=art / "civilcomments_test_wilds_selected_rows_civilcomments_erm_softclip_10s_20260328.csv",
            phase1_csv=art / "civilcomments_distilbert-base-uncased_phase1_pockets_civilcomments_erm_softclip_10s_20260328.csv",
            heldout_loss_col="test_overall_loss",
            heldout_acc_col="test_overall_acc",
            calibration_ece_col="test_ece",
            calibration_brier_col="test_brier",
            extra_cols={
                "test_wilds_wg_loss": "test_wilds_wg_loss",
                "test_wilds_wg_acc": "test_wilds_wg_acc",
                "test_wilds_wg_ece": "test_wilds_wg_ece",
                "test_wilds_wg_brier": "test_wilds_wg_brier",
            },
        ),
    ]


def _load_tail_rows(phase1_csv: Path, selected_df: pd.DataFrame, tail_family: str = "teacher_difficulty") -> pd.DataFrame:
    phase1 = pd.read_csv(phase1_csv)
    phase1 = phase1[
        (phase1["split"] == "val")
        & (phase1["family"] == tail_family)
    ].copy()
    agg = (
        phase1.groupby([c for c in ["regime", "seed", "epoch", "tag"] if c in phase1.columns], as_index=False)[
            ["oracle_wg_acc", "within_cvar_mean", "worst_cell_cvar", "worst_cell_mean_loss"]
        ]
        .mean(numeric_only=True)
    )
    merge_keys = [c for c in ["regime", "seed", "epoch", "tag"] if c in selected_df.columns and c in agg.columns]
    return selected_df[merge_keys].drop_duplicates().merge(agg, on=merge_keys, how="left")


def _pivot_pair(df: pd.DataFrame, baseline_regime: str, soft_regime: str, value_cols: List[str]) -> pd.DataFrame:
    work = df[df["regime"].isin([baseline_regime, soft_regime])].copy()
    if work.empty:
        return pd.DataFrame()
    keep = ["seed", "regime"] + [c for c in value_cols if c in work.columns]
    work = work[keep].drop_duplicates()
    piv = work.pivot(index="seed", columns="regime", values=[c for c in keep if c not in {"seed", "regime"}])
    rows = []
    for seed in sorted(piv.index.tolist()):
        row = {"seed": int(seed)}
        ok = True
        for col in piv.columns.levels[0]:
            if (col, baseline_regime) not in piv.columns or (col, soft_regime) not in piv.columns:
                ok = False
                break
            row[f"{col}_baseline"] = piv.loc[seed, (col, baseline_regime)]
            row[f"{col}_softclip"] = piv.loc[seed, (col, soft_regime)]
            row[f"delta_{col}_soft_minus_base"] = pd.to_numeric(row[f"{col}_softclip"], errors="coerce") - pd.to_numeric(row[f"{col}_baseline"], errors="coerce")
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def _make_suite_seed_table(spec: SuiteSpec) -> pd.DataFrame:
    lure = pd.read_csv(spec.lure_csv)
    heldout = pd.read_csv(spec.heldout_csv)
    heldout_piv = _pivot_pair(
        heldout,
        spec.baseline_regime,
        spec.soft_regime,
        value_cols=[spec.heldout_loss_col, spec.heldout_acc_col] + list(spec.extra_cols.values()),
    )

    calib = pd.read_csv(spec.calibration_csv)
    calib_piv = _pivot_pair(
        calib,
        spec.baseline_regime,
        spec.soft_regime,
        value_cols=[
            spec.calibration_ece_col,
            spec.calibration_brier_col,
        ] + [c for c in spec.extra_cols.values() if c.endswith("_ece") or c.endswith("_brier")],
    )

    selected_rows = pd.read_csv(spec.selected_rows_csv)
    tail_rows = _load_tail_rows(spec.phase1_csv, selected_rows)
    tail_piv = _pivot_pair(
        tail_rows,
        spec.baseline_regime,
        spec.soft_regime,
        value_cols=["oracle_wg_acc", "worst_cell_cvar", "worst_cell_mean_loss"],
    )

    merged = lure.merge(heldout_piv, on="seed", how="left").merge(calib_piv, on="seed", how="left").merge(tail_piv, on="seed", how="left")
    merged.insert(0, "suite", spec.suite)
    merged.insert(1, "suite_label", spec.label)
    return merged.sort_values("seed").reset_index(drop=True)


def _paired_records(df: pd.DataFrame, label: str) -> List[Dict[str, object]]:
    metrics = [
        ("proxy_clip", ["delta_clip_proxy_soft_minus_base"]),
        ("val_loss", ["delta_val_loss_soft_minus_base"]),
        ("test_loss", ["delta_test_hosp_2_loss_soft_minus_base", "delta_test_overall_loss_soft_minus_base"]),
        ("tail_cvar", ["delta_worst_cell_cvar_soft_minus_base"]),
        ("ece", ["delta_test_ece_soft_minus_base", "delta_val_ece_soft_minus_base"]),
        ("brier", ["delta_test_brier_soft_minus_base", "delta_val_brier_soft_minus_base"]),
        ("test_acc", ["delta_test_hosp_2_acc_soft_minus_base", "delta_test_overall_acc_soft_minus_base"]),
        ("wilds_wg_loss", ["delta_test_wilds_wg_loss_soft_minus_base"]),
        ("wilds_wg_acc", ["delta_test_wilds_wg_acc_soft_minus_base"]),
        ("wilds_wg_ece", ["delta_test_wilds_wg_ece_soft_minus_base"]),
        ("wilds_wg_brier", ["delta_test_wilds_wg_brier_soft_minus_base"]),
    ]
    rows = []
    for idx, (metric, candidates) in enumerate(metrics):
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mean, lo, hi = _bootstrap_ci(vals, seed=101 + idx)
        rows.append(
            {
                "suite_label": label,
                "metric": metric,
                "n": int(np.isfinite(vals).sum()),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return rows


def _dominance_record(df: pd.DataFrame, label: str) -> Dict[str, object]:
    proxy = pd.to_numeric(df["delta_clip_proxy_soft_minus_base"], errors="coerce")
    val_loss = pd.to_numeric(df["delta_val_loss_soft_minus_base"], errors="coerce")

    test_loss = None
    for col in ["delta_test_hosp_2_loss_soft_minus_base", "delta_test_overall_loss_soft_minus_base"]:
        if col in df.columns:
            test_loss = pd.to_numeric(df[col], errors="coerce")
            break

    test_acc = None
    for col in ["delta_test_hosp_2_acc_soft_minus_base", "delta_test_overall_acc_soft_minus_base"]:
        if col in df.columns:
            test_acc = pd.to_numeric(df[col], errors="coerce")
            break

    ece = None
    for col in ["delta_test_ece_soft_minus_base", "delta_val_ece_soft_minus_base"]:
        if col in df.columns:
            ece = pd.to_numeric(df[col], errors="coerce")
            break
    tail = pd.to_numeric(df["delta_worst_cell_cvar_soft_minus_base"], errors="coerce") if "delta_worst_cell_cvar_soft_minus_base" in df.columns else None

    proxy_better = proxy < 0
    rec = {
        "suite_label": label,
        "n": int(df.shape[0]),
        "n_proxy_better": int(np.sum(proxy_better)),
        "n_proxy_better_and_val_loss_worse": int(np.sum(proxy_better & (val_loss > 0))),
        "n_proxy_better_and_test_loss_worse": int(np.sum(proxy_better & (test_loss > 0))) if test_loss is not None else np.nan,
        "n_proxy_better_and_tail_worse": int(np.sum(proxy_better & (tail > 0))) if tail is not None else np.nan,
        "n_proxy_better_and_ece_worse": int(np.sum(proxy_better & (ece > 0))) if ece is not None else np.nan,
        "mean_test_acc_delta": float(np.nanmean(test_acc.to_numpy(dtype=np.float64))) if test_acc is not None else np.nan,
    }
    if "delta_test_wilds_wg_loss_soft_minus_base" in df.columns:
        wg_loss = pd.to_numeric(df["delta_test_wilds_wg_loss_soft_minus_base"], errors="coerce")
        rec["n_proxy_better_and_wg_loss_worse"] = int(np.sum(proxy_better & (wg_loss > 0)))
    if "delta_test_wilds_wg_acc_soft_minus_base" in df.columns:
        wg_acc = pd.to_numeric(df["delta_test_wilds_wg_acc_soft_minus_base"], errors="coerce")
        rec["mean_wg_acc_delta"] = float(np.nanmean(wg_acc.to_numpy(dtype=np.float64)))
    return rec


def _write_tex_tables(paired: pd.DataFrame, dominance: pd.DataFrame, paired_tex: Path, dominance_tex: Path) -> None:
    metric_label = {
        "proxy_clip": "proxy clip",
        "val_loss": "validation loss",
        "test_loss": "held-out loss",
        "tail_cvar": "tail CVaR",
        "ece": "ECE",
        "brier": "Brier",
        "test_acc": "held-out acc",
        "wilds_wg_loss": "WILDS wg loss",
        "wilds_wg_acc": "WILDS wg acc",
        "wilds_wg_ece": "WILDS wg ECE",
        "wilds_wg_brier": "WILDS wg Brier",
    }
    paired_lines = [
        r"\begin{tabular}{llccc}",
        r"  \toprule",
        r"  Suite & Metric & Mean & 95\% CI low & 95\% CI high \\",
        r"  \midrule",
    ]
    metric_order = ["proxy_clip", "val_loss", "test_loss", "tail_cvar", "ece", "brier", "test_acc", "wilds_wg_loss", "wilds_wg_acc"]
    for suite in paired["suite_label"].drop_duplicates():
        sub = paired[paired["suite_label"] == suite].copy()
        sub["metric_order"] = sub["metric"].apply(lambda m: metric_order.index(m) if m in metric_order else 999)
        sub = sub.sort_values(["metric_order", "metric"])
        for _, row in sub.iterrows():
            if pd.isna(row["mean"]):
                continue
            paired_lines.append(
                f"  {suite} & {metric_label.get(row['metric'], row['metric'])} & {row['mean']:+.4f} & {row['ci_low']:+.4f} & {row['ci_high']:+.4f} \\\\"
            )
    paired_lines.extend([r"  \bottomrule", r"\end{tabular}"])
    paired_tex.parent.mkdir(parents=True, exist_ok=True)
    paired_tex.write_text("\n".join(paired_lines) + "\n", encoding="utf-8")

    dom_lines = [
        r"\begin{tabular}{lccccc}",
        r"  \toprule",
        r"  Suite & Proxy better & Proxy$\land$val loss worse & Proxy$\land$test loss worse & Proxy$\land$reliability worse & Proxy$\land$ECE worse \\",
        r"  \midrule",
    ]
    def fmt_count(v: object, n: int) -> str:
        if pd.isna(v):
            return f"--/{n}"
        s = f"{int(v)}/{n}"
        return rf"\textbf{{{s}}}" if int(v) == n else s

    for _, row in dominance.iterrows():
        reliability_count = row["n_proxy_better_and_tail_worse"]
        if "n_proxy_better_and_wg_loss_worse" in row.index and pd.notna(row["n_proxy_better_and_wg_loss_worse"]):
            reliability_count = row["n_proxy_better_and_wg_loss_worse"]
        dom_lines.append(
            "  {suite} & {pb} & {pvl} & {ptl} & {ptw} & {pew} \\\\".format(
                suite=row["suite_label"],
                pb=fmt_count(row["n_proxy_better"], int(row["n"])),
                pvl=fmt_count(row["n_proxy_better_and_val_loss_worse"], int(row["n"])),
                ptl=fmt_count(row["n_proxy_better_and_test_loss_worse"], int(row["n"])),
                ptw=fmt_count(reliability_count, int(row["n"])),
                pew=fmt_count(row["n_proxy_better_and_ece_worse"], int(row["n"])),
            )
        )
    dom_lines.extend([r"  \bottomrule", r"\end{tabular}"])
    dominance_tex.parent.mkdir(parents=True, exist_ok=True)
    dominance_tex.write_text("\n".join(dom_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date_tag", default="20260328")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--paper_tables_dir", default="")
    args = ap.parse_args()

    root = _repo_root()
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else root / "artifacts" / "metrics"
    paper_tables_dir = Path(args.paper_tables_dir) if str(args.paper_tables_dir).strip() else root / "paper" / "neurips2026_selection_risk" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = []
    dominance_rows = []
    for spec in _suite_specs(root, str(args.date_tag)):
        suite_seed_csv = out_dir / f"{spec.suite}_stepup_seed_table_{args.date_tag}.csv"
        if suite_seed_csv.exists():
            suite_df = pd.read_csv(suite_seed_csv)
        else:
            suite_df = _make_suite_seed_table(spec)
        suite_df.to_csv(out_dir / f"{spec.suite}_stepup_seed_table_{args.date_tag}.csv", index=False)
        paired_rows.extend(_paired_records(suite_df, spec.label))
        dominance_rows.append(_dominance_record(suite_df, spec.label))

    paired = pd.DataFrame(paired_rows)
    dominance = pd.DataFrame(dominance_rows)

    paired_csv = out_dir / f"acceptance_stepup_paired_effects_{args.date_tag}.csv"
    dom_csv = out_dir / f"acceptance_stepup_dominance_{args.date_tag}.csv"
    paired.to_csv(paired_csv, index=False)
    dominance.to_csv(dom_csv, index=False)

    _write_tex_tables(
        paired=paired,
        dominance=dominance,
        paired_tex=paper_tables_dir / "table_acceptance_stepup_paired.tex",
        dominance_tex=paper_tables_dir / "table_acceptance_stepup_dominance.tex",
    )

    print(f"[ok] wrote {paired_csv}")
    print(f"[ok] wrote {dom_csv}")


if __name__ == "__main__":
    main()
