import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SuiteSpec:
    suite: str
    suite_label: str
    rows_csv: Path
    eval_rows_csv: Path
    phase1_csv: Path
    tail_family: str
    heldout_loss_col: str = "test_hosp_2_loss"
    heldout_acc_col: str = "test_hosp_2_acc"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _specs(root: Path) -> list[SuiteSpec]:
    art = root / "artifacts" / "metrics"
    return [
        SuiteSpec(
            suite="camelyon_erm_p95_adaptive",
            suite_label="Camelyon17 ERM",
            rows_csv=art / "adaptive_guardrail_rows_camelyon_erm_p95_mu2sigma_20260327.csv",
            eval_rows_csv=art / "adaptive_guardrail_eval_rows_camelyon_erm_p95_mu2sigma_20260327.csv",
            phase1_csv=art / "camelyon17_resnet50_phase1_pockets_v11erm_softclip_cam_10s_fix_20260228.csv",
            tail_family="teacher_difficulty",
        ),
        SuiteSpec(
            suite="camelyon_finetune_p95_adaptive",
            suite_label="Camelyon17 Finetune",
            rows_csv=art / "adaptive_guardrail_rows_camelyon_finetune_p95_mu2sigma_20260327.csv",
            eval_rows_csv=art / "adaptive_guardrail_eval_rows_camelyon_finetune_p95_mu2sigma_20260327.csv",
            phase1_csv=art / "camelyon17_resnet50_phase1_pockets_finetune_cam_scivalid10s_20260326.csv",
            tail_family="teacher_difficulty",
        ),
    ]


def _aggregate_tail(phase1_csv: Path, tail_family: str) -> pd.DataFrame:
    df = pd.read_csv(phase1_csv)
    req = {"regime", "seed", "epoch", "family", "worst_cell_cvar"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{phase1_csv} missing required columns: {sorted(miss)}")
    if "split" in df.columns:
        df = df[(df["family"] == tail_family) & (df["split"] == "val")].copy()
    else:
        df = df[df["family"] == tail_family].copy()
    group_cols = [c for c in ("regime", "seed", "tag", "epoch") if c in df.columns]
    return (
        df.groupby(group_cols, as_index=False)
        .agg({"worst_cell_cvar": "mean"})
        .rename(columns={"worst_cell_cvar": "selected_tail_worst_cvar"})
    )


def _attach_missing_tags(eval_df: pd.DataFrame, selection_rows: pd.DataFrame) -> pd.DataFrame:
    if "tag" not in selection_rows.columns:
        return eval_df
    if "tag" in eval_df.columns and eval_df["tag"].notna().all():
        return eval_df
    keys = [c for c in ("regime", "seed", "epoch") if c in eval_df.columns and c in selection_rows.columns]
    tags = selection_rows[keys + ["tag"]].drop_duplicates().copy()
    if "tag" not in eval_df.columns:
        return eval_df.merge(tags, on=keys, how="left")
    filled = eval_df.merge(tags, on=keys, how="left", suffixes=("", "_from_sel"))
    filled["tag"] = filled["tag"].fillna(filled["tag_from_sel"])
    return filled.drop(columns=["tag_from_sel"])


def _merge_on_common_keys(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("regime", "seed", "epoch", "tag") if c in left.columns and c in right.columns]
    if not keys:
        raise ValueError("No common merge keys found.")
    return left.merge(right.drop_duplicates(subset=keys), on=keys, how="left")


def _policy_label(policy: str) -> str:
    return {
        "baseline": "Baseline",
        "proxy_only": "Proxy-only",
        "guardrail": r"Adaptive $\mu+2\sigma$",
    }.get(str(policy), str(policy))


def _summarize_suite(spec: SuiteSpec) -> pd.DataFrame:
    rows = pd.read_csv(spec.rows_csv)
    eval_rows = pd.read_csv(spec.eval_rows_csv)
    eval_rows = _attach_missing_tags(eval_rows, rows)
    tail = _aggregate_tail(spec.phase1_csv, tail_family=spec.tail_family)

    merged = _merge_on_common_keys(rows, eval_rows)
    merged = _merge_on_common_keys(merged, tail)
    merged = merged[merged["selection_policy"].isin(["baseline", "proxy_only", "guardrail"])].copy()
    merged = merged.drop_duplicates(subset=["selection_policy", "seed"]).reset_index(drop=True)

    base = merged[merged["selection_policy"] == "baseline"].set_index("seed")
    out = []
    for policy in ["baseline", "proxy_only", "guardrail"]:
        sub = merged[merged["selection_policy"] == policy].set_index("seed")
        seeds = sorted(set(base.index) & set(sub.index))
        if not seeds:
            continue
        base_sub = base.loc[seeds]
        sub = sub.loc[seeds]
        out.append(
            {
                "suite": spec.suite,
                "suite_label": spec.suite_label,
                "selection_policy": policy,
                "selector_label": _policy_label(policy),
                "accept_rate": float(1.0 - sub["fallback_to_baseline"].astype(float).mean()),
                "delta_loss": float((sub[spec.heldout_loss_col] - base_sub[spec.heldout_loss_col]).mean()),
                "delta_acc": float((sub[spec.heldout_acc_col] - base_sub[spec.heldout_acc_col]).mean()),
                "delta_tail": float((sub["selected_tail_worst_cvar"] - base_sub["selected_tail_worst_cvar"]).mean()),
                "baseline_window": int(sub["baseline_window"].dropna().iloc[0]),
            }
        )
    return pd.DataFrame(out)


def _write_tex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"  \toprule",
        r"  Suite & Selector & Admit rate & $\Delta$Held-out loss$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $\Delta$Tail$\downarrow$ \\",
        r"  \midrule",
    ]
    last_suite = None
    for _, row in df.iterrows():
        suite = row["suite_label"]
        suite_cell = suite if suite != last_suite else ""
        last_suite = suite
        lines.append(
            "  {suite} & {selector} & {rate:.2f} & {dloss:+.3f} & {dacc:+.3f} & {dtail:+.3f} \\\\".format(
                suite=suite_cell,
                selector=row["selector_label"],
                rate=row["accept_rate"],
                dloss=row["delta_loss"],
                dacc=row["delta_acc"],
                dtail=row["delta_tail"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="artifacts/metrics/adaptive_guardrail_selector_summary_20260331.csv")
    ap.add_argument(
        "--out_tex",
        default="paper/neurips2026_selection_risk/tables/table_adaptive_guardrail_summary.tex",
    )
    args = ap.parse_args()

    root = _repo_root()
    frames = [_summarize_suite(spec) for spec in _specs(root)]
    combined = pd.concat(frames, ignore_index=True)
    suite_order = {"Camelyon17 ERM": 0, "Camelyon17 Finetune": 1}
    policy_order = {"Baseline": 0, "Proxy-only": 1, r"Adaptive $\mu+2\sigma$": 2}
    combined["suite_order"] = combined["suite_label"].map(suite_order)
    combined["policy_order"] = combined["selector_label"].map(policy_order)
    combined = combined.sort_values(["suite_order", "policy_order"]).drop(columns=["suite_order", "policy_order"]).reset_index(drop=True)

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    _write_tex(combined, root / args.out_tex)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {root / args.out_tex}")


if __name__ == "__main__":
    main()
