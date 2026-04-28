import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SuiteSpec:
    suite: str
    suite_label: str
    config: str
    dataset: str
    phase0_csv: Path
    phase1_csv: Path
    guardrail_rows_csv: Path
    existing_eval_rows_csv: Path
    baseline_regime: str
    target_regime: str
    tag_filter: str
    proxy_family: str = "conf_teacher_wpl"
    tail_family: str = "teacher_difficulty"
    heldout_loss_col: str = "test_hosp_2_loss"
    heldout_acc_col: str = "test_hosp_2_acc"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _specs(root: Path) -> list[SuiteSpec]:
    art = root / "artifacts" / "metrics"
    return [
        SuiteSpec(
            suite="camelyon_erm_p95",
            suite_label="Camelyon17 ERM",
            config="configs/base_v11_erm_softclip_camelyon_10seeds_nw0_fix.yaml",
            dataset="camelyon17",
            phase0_csv=art / "camelyon17_resnet50_phase0_val_metrics_v11erm_softclip_cam_10s_fix_20260228.csv",
            phase1_csv=art / "camelyon17_resnet50_phase1_pockets_v11erm_softclip_cam_10s_fix_20260228.csv",
            guardrail_rows_csv=art / "guardrail_rows_camelyon_erm_p95_ratio125_20260326.csv",
            existing_eval_rows_csv=art / "guardrail_eval_rows_camelyon_erm_p95_ratio125_20260326.csv",
            baseline_regime="erm",
            target_regime="erm_softclip_p95_a10_cam",
            tag_filter="v11ermsoftclipfix_cam_10s",
        ),
        SuiteSpec(
            suite="camelyon_finetune_p95",
            suite_label="Camelyon17 Finetune",
            config="configs/camelyon_base_scivalid_10seeds.yaml",
            dataset="camelyon17",
            phase0_csv=art / "camelyon17_resnet50_phase0_val_metrics_finetune_cam_scivalid10s_20260326.csv",
            phase1_csv=art / "camelyon17_resnet50_phase1_pockets_finetune_cam_scivalid10s_20260326.csv",
            guardrail_rows_csv=art / "guardrail_rows_camelyon_finetune_p95_ratio125_20260326.csv",
            existing_eval_rows_csv=art / "guardrail_eval_rows_camelyon_finetune_p95_ratio125_20260326.csv",
            baseline_regime="rcgdro_finetune",
            target_regime="rcgdro_softclip_p95_a10_cam_finetune",
            tag_filter="scivalid10s",
        ),
    ]


def _aggregate_phase0(metrics_csv: Path, proxy_family: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
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
        raise ValueError(f"{metrics_csv} missing required columns: {sorted(miss)}")
    df = df[df["family"] == proxy_family].copy()
    if df.empty:
        raise ValueError(f"{metrics_csv} has no rows for family={proxy_family}")
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
        .sort_values(["regime", "seed", "tag", "epoch"])
        .reset_index(drop=True)
    )


def _metric_col(df: pd.DataFrame, regime: str) -> str:
    if "clip" in str(regime).lower() and df["proxy_worst_loss_clip_min"].notna().any():
        return "proxy_worst_loss_clip_min"
    return "proxy_worst_loss_min"


def _select_proxy(df: pd.DataFrame, regime: str) -> pd.Series:
    metric_col = _metric_col(df, regime)
    row = df.sort_values([metric_col, "val_overall_loss", "epoch"], ascending=[True, True, True]).iloc[0].copy()
    row["selected_proxy_metric"] = float(row[metric_col])
    row["selected_proxy_metric_col"] = metric_col
    return row


def _select_val_loss(df: pd.DataFrame, regime: str) -> pd.Series:
    metric_col = _metric_col(df, regime)
    row = df.sort_values(["val_overall_loss", metric_col, "epoch"], ascending=[True, True, True]).iloc[0].copy()
    row["selected_proxy_metric"] = float(row[metric_col])
    row["selected_proxy_metric_col"] = metric_col
    return row


def _aggregate_tail(phase1_csv: Path, tail_family: str) -> pd.DataFrame:
    df = pd.read_csv(phase1_csv)
    req = {"regime", "seed", "epoch", "family", "worst_cell_cvar", "worst_cell_mean_loss"}
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
        .agg({"worst_cell_cvar": "mean", "worst_cell_mean_loss": "mean"})
        .rename(
            columns={
                "worst_cell_cvar": "selected_tail_worst_cvar",
                "worst_cell_mean_loss": "selected_tail_worst_loss",
            }
        )
    )


def _merge_on_common_keys(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("regime", "seed", "epoch", "tag") if c in left.columns and c in right.columns]
    if not keys:
        raise ValueError("No common merge keys found.")
    return left.merge(right.drop_duplicates(subset=keys), on=keys, how="left")


def _build_selection_rows(spec: SuiteSpec) -> pd.DataFrame:
    phase0 = _aggregate_phase0(spec.phase0_csv, proxy_family=spec.proxy_family)
    base_rows = pd.read_csv(spec.guardrail_rows_csv)
    dedup_cols = [c for c in ("suite", "target_regime", "selection_policy", "seed", "regime", "epoch", "tag") if c in base_rows.columns]
    base_rows = base_rows.drop_duplicates(subset=dedup_cols).copy()
    keep = base_rows[base_rows["selection_policy"].isin(["baseline", "proxy_only", "guardrail"])].copy()

    baseline_common = (
        keep[keep["selection_policy"] == "baseline"]
        .drop_duplicates(subset=["seed"])
        .set_index("seed")
    )

    rows = []
    for seed in sorted(baseline_common.index.astype(int).tolist()):
        cand = phase0[(phase0["regime"] == spec.target_regime) & (phase0["seed"] == seed)].copy()
        if cand.empty:
            continue
        pick = _select_val_loss(cand, regime=spec.target_regime)
        base = baseline_common.loc[seed]
        rows.append(
            {
                "suite": spec.suite,
                "target_regime": spec.target_regime,
                "baseline_regime": spec.baseline_regime,
                "seed": int(seed),
                "baseline_epoch": int(base["baseline_epoch"]),
                "baseline_tag": str(base["baseline_tag"]),
                "baseline_val_overall_loss": float(base["baseline_val_overall_loss"]),
                "baseline_proxy_metric": float(base["baseline_proxy_metric"]),
                "budget_mode": "none",
                "budget_value": float("nan"),
                "budget_threshold": float("nan"),
                "selection_policy": "val_loss_only",
                "regime": spec.target_regime,
                "tag": str(pick["tag"]),
                "epoch": int(pick["epoch"]),
                "chosen_val_overall_loss": float(pick["val_overall_loss"]),
                "chosen_val_overall_acc": float(pick["val_overall_acc"]),
                "chosen_proxy_metric": float(pick["selected_proxy_metric"]),
                "chosen_proxy_metric_col": str(pick["selected_proxy_metric_col"]),
                "fallback_to_baseline": False,
                "feasible_target_epochs": int(cand.shape[0]),
            }
        )

    combined = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    order = {"baseline": 0, "proxy_only": 1, "val_loss_only": 2, "guardrail": 3}
    combined["policy_order"] = combined["selection_policy"].map(order).fillna(99)
    combined = combined.sort_values(["suite", "seed", "policy_order"]).drop(columns=["policy_order"]).reset_index(drop=True)
    return combined


def _run_domain_eval(root: Path, spec: SuiteSpec, selection_rows_csv: Path, heldout_rows_csv: Path, heldout_summary_csv: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.camelyon_domain_eval",
        "--config",
        spec.config,
        "--dataset",
        spec.dataset,
        "--summary_csv",
        str(selection_rows_csv),
        "--tag_filter",
        spec.tag_filter,
        "--out_csv",
        str(heldout_rows_csv),
        "--out_summary",
        str(heldout_summary_csv),
    ]
    subprocess.run(cmd, cwd=str(root), check=True)


def _attach_missing_tags(eval_df: pd.DataFrame, selection_rows: pd.DataFrame) -> pd.DataFrame:
    if "tag" not in selection_rows.columns:
        return eval_df
    if "tag" in eval_df.columns and eval_df["tag"].notna().all():
        return eval_df
    keys = [c for c in ("regime", "seed", "epoch") if c in eval_df.columns and c in selection_rows.columns]
    tags = selection_rows[keys + ["tag"]].drop_duplicates().copy()
    if "tag" not in eval_df.columns:
        return eval_df.merge(tags, on=keys, how="left")
    missing = eval_df["tag"].isna()
    if not missing.any():
        return eval_df
    filled = eval_df.merge(tags, on=keys, how="left", suffixes=("", "_from_sel"))
    filled["tag"] = filled["tag"].fillna(filled["tag_from_sel"])
    return filled.drop(columns=["tag_from_sel"])


def _missing_eval_rows(selection_rows: pd.DataFrame, existing_eval: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("regime", "seed", "epoch", "tag") if c in selection_rows.columns]
    sel = selection_rows[keys].drop_duplicates().copy()
    if existing_eval.empty:
        return sel
    merge_keys = [c for c in keys if c in existing_eval.columns]
    have = existing_eval[merge_keys].drop_duplicates().copy()
    merged = sel.merge(have.assign(_have=1), on=merge_keys, how="left")
    return merged[merged["_have"].isna()].drop(columns=["_have"]).reset_index(drop=True)


def _policy_label(policy: str) -> str:
    return {
        "baseline": "Baseline",
        "proxy_only": "Proxy-only",
        "val_loss_only": "Validation-loss",
        "guardrail": "1.25x guardrail",
    }.get(str(policy), str(policy))


def _summarize_suite(spec: SuiteSpec, selection_rows: pd.DataFrame, heldout_rows_csv: Path) -> pd.DataFrame:
    heldout = pd.read_csv(heldout_rows_csv)
    tail = _aggregate_tail(spec.phase1_csv, tail_family=spec.tail_family)

    merged = _merge_on_common_keys(selection_rows, heldout)
    merged = _merge_on_common_keys(merged, tail)

    rows = []
    base = merged[merged["selection_policy"] == "baseline"].drop_duplicates(subset=["seed"]).set_index("seed")
    for policy in ["baseline", "proxy_only", "val_loss_only", "guardrail"]:
        sub = merged[merged["selection_policy"] == policy].drop_duplicates(subset=["seed"]).set_index("seed")
        seeds = sorted(set(base.index) & set(sub.index))
        if not seeds:
            continue
        base_sub = base.loc[seeds]
        sub = sub.loc[seeds]
        rows.append(
            {
                "suite": spec.suite,
                "suite_label": spec.suite_label,
                "selection_policy": policy,
                "selector_label": _policy_label(policy),
                "accept_rate": float(1.0 - sub["fallback_to_baseline"].astype(float).mean()),
                "delta_loss": float((sub[spec.heldout_loss_col] - base_sub[spec.heldout_loss_col]).mean()),
                "delta_acc": float((sub[spec.heldout_acc_col] - base_sub[spec.heldout_acc_col]).mean()),
                "delta_tail": float((sub["selected_tail_worst_cvar"] - base_sub["selected_tail_worst_cvar"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _camelyon_baseline_values(root: Path, suite_label: str) -> dict[str, float]:
    art = root / "artifacts" / "metrics"
    if suite_label == "Camelyon17 ERM":
        domain = pd.read_csv(art / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv")
        effect = pd.read_csv(art / "camelyon17_effect_size_erm_softclip_v11_10s_fix_20260228.csv")
        regime = "erm"
    elif suite_label == "Camelyon17 Finetune":
        domain = pd.read_csv(art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv")
        effect = pd.read_csv(art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_selected.csv")
        regime = "rcgdro_finetune"
    else:
        raise ValueError(f"Unknown Camelyon suite label: {suite_label}")
    domain_row = domain[domain["regime"] == regime].iloc[0]
    effect_row = effect[effect["regime"] == regime].iloc[0]
    return {
        "acc": float(domain_row["test_hosp_2_acc_mean"]),
        "loss": float(domain_row["test_hosp_2_loss_mean"]),
        "tail": float(effect_row["tail_worst_cvar_mean"]),
    }


def _mean_epoch(root: Path, suite: str, policy: str) -> float:
    rows_csv = root / "artifacts" / "metrics" / f"{suite}_selector_rows_trueval_20260329.csv"
    if not rows_csv.exists():
        return float("nan")
    rows = pd.read_csv(rows_csv)
    sub = rows[rows["selection_policy"] == policy].copy()
    if sub.empty:
        return float("nan")
    return float(sub["epoch"].mean())


def _write_tex(df: pd.DataFrame, out_tex: Path, root: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"  \toprule",
        r"  Suite & Selector / rule & Reported policy & Supp.\ admit & Sel.\ epoch & Acc / $\Delta$Acc & Loss / $\Delta$Loss & Rel. / $\Delta$Rel \\",
        r"  \midrule",
        r"  \multicolumn{8}{@{}l}{\emph{Camelyon17 selector comparison (absolute held-out metrics across baseline and suppressive trajectories)}} \\",
    ]

    policy_order = ["baseline", "proxy_only", "val_loss_only", "guardrail"]
    report_policy = {
        "baseline": "baseline run",
        "proxy_only": "suppressive run",
        "val_loss_only": "suppressive run",
        "guardrail": r"\shortstack[l]{mixed report:\\admit or fallback}",
    }
    selector_label = {
        "baseline": "Baseline",
        "proxy_only": "Proxy-only",
        "val_loss_only": "Val-loss",
        "guardrail": "1.25x veto",
    }
    for suite_label, suite_tex in [
        ("Camelyon17 ERM", r"\shortstack[l]{Camelyon17\\ERM}"),
        ("Camelyon17 Finetune", r"\shortstack[l]{Camelyon17\\Finetune}"),
    ]:
        base = _camelyon_baseline_values(root, suite_label)
        sub = df[df["suite_label"] == suite_label].set_index("selection_policy")
        lines.append(rf"  \multirow{{4}}{{*}}{{{suite_tex}}}")
        for i, policy in enumerate(policy_order):
            row = sub.loc[policy]
            acc = base["acc"] + float(row["delta_acc"])
            loss = base["loss"] + float(row["delta_loss"])
            tail = base["tail"] + float(row["delta_tail"])
            admit = "---" if policy == "baseline" else f"{float(row['accept_rate']):.2f}"
            epoch = _mean_epoch(root, str(row["suite"]), policy)
            prefix = "    &" if i == 0 else "    &"
            lines.append(
                "{prefix} {selector} & {reported} & {admit} & {epoch:.1f} & {acc:.3f} & {loss:.3f} & {tail:.2f} \\\\".format(
                    prefix=prefix,
                    selector=selector_label[policy],
                    reported=report_policy[policy],
                    admit=admit,
                    epoch=epoch,
                    acc=acc,
                    loss=loss,
                    tail=tail,
                )
            )
        if suite_label == "Camelyon17 ERM":
            lines.append(r"  \midrule")

    lines.extend(
        [
            r"  \midrule",
            r"  \multicolumn{8}{@{}l}{\emph{Cross-modal / regression selector verdicts (deltas from baseline)}} \\",
        ]
    )

    civil = pd.read_csv(root / "artifacts" / "metrics" / "civilcomments_selector_comparison_summary_20260331.csv")
    civil = civil[civil["selection_policy"].isin(["proxy_only", "val_loss_only", "guardrail"])].set_index("selection_policy")
    civil_order = [("proxy_only", "Proxy-only", "suppressive run"), ("val_loss_only", "Val-loss", "suppressive run"), ("guardrail", "1.25x veto", r"\shortstack[l]{baseline\\fallback}")]
    lines.append(r"  \multirow{3}{*}{\shortstack[l]{Frozen\\CivilComments}}")
    for policy, label, reported in civil_order:
        row = civil.loc[policy]
        lines.append(
            "    & {selector} & {reported} & {rate:.2f} & --- & ${dacc:+.3f}$ & ${dloss:+.3f}$ & ${dwg:+.3f}$ \\\\".format(
                selector=label,
                reported=reported,
                rate=float(row["accept_rate"]),
                dacc=float(row["delta_overall_acc"]),
                dloss=float(row["delta_overall_loss"]),
                dwg=float(row["delta_wg_loss"]),
            )
        )
    lines.append(r"  \midrule")

    acs = pd.read_csv(root / "artifacts" / "metrics" / "acs_income" / "phase3_selector_winsorized_p95.csv")
    acs = acs.groupby("selector", as_index=False).agg({"admit": "mean", "delta_test_raw": "mean", "delta_tail": "mean"})
    acs = acs.set_index("selector")
    acs_order = [
        ("Proxy-only", "Proxy-only", "suppressive run"),
        ("Val-loss-only", "Val-loss", "suppressive run"),
        ("1.25x guardrail", "1.25x veto", "suppressive run"),
    ]
    lines.append(r"  \multirow{3}{*}{ACSIncome}")
    for policy, label, reported in acs_order:
        row = acs.loc[policy]
        lines.append(
            "    & {selector} & {reported} & {rate:.2f} & --- & --- & ${dloss:+.3f}$ & ${dtail:+.3f}$ \\\\".format(
                selector=label,
                reported=reported,
                rate=float(row["admit"]),
                dloss=float(row["delta_test_raw"]),
                dtail=float(row["delta_tail"]),
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/metrics")
    ap.add_argument(
        "--out_tex",
        default="paper/neurips2026_selection_risk/tables/table_selector_comparison.tex",
    )
    ap.add_argument("--stamp", default="20260329")
    args = ap.parse_args()

    root = _repo_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    precomputed = []
    for spec in _specs(root):
        precomputed_csv = out_dir / f"{spec.suite}_selector_summary_trueval_{args.stamp}.csv"
        if not precomputed_csv.exists():
            precomputed = []
            break
        suite_summary = pd.read_csv(precomputed_csv)
        suite_summary["suite_label"] = spec.suite_label
        precomputed.append(suite_summary)
    if precomputed:
        combined = pd.concat(precomputed, ignore_index=True)
        suite_order = {"Camelyon17 ERM": 0, "Camelyon17 Finetune": 1}
        policy_order = {"Baseline": 0, "Proxy-only": 1, "Val-loss-only": 2, "1.25x guardrail": 3}
        combined["suite_order"] = combined["suite_label"].map(suite_order)
        combined["policy_order"] = combined["selector_label"].map(policy_order)
        combined = combined.sort_values(["suite_order", "policy_order"]).drop(columns=["suite_order", "policy_order"]).reset_index(drop=True)
        out_csv = out_dir / f"selector_comparison_trueval_{args.stamp}.csv"
        combined.to_csv(out_csv, index=False)
        _write_tex(combined, root / args.out_tex, root)
        print(f"[ok] wrote {out_csv} from precomputed suite summaries")
        print(f"[ok] wrote {root / args.out_tex}")
        return

    suite_summaries = []
    for spec in _specs(root):
        selection_rows = _build_selection_rows(spec)
        selection_rows_csv = out_dir / f"{spec.suite}_selector_rows_trueval_{args.stamp}.csv"
        heldout_rows_csv = out_dir / f"{spec.suite}_selector_eval_rows_trueval_{args.stamp}.csv"
        heldout_summary_csv = out_dir / f"{spec.suite}_selector_eval_summary_trueval_{args.stamp}.csv"
        selection_rows.to_csv(selection_rows_csv, index=False)

        existing_eval = pd.read_csv(spec.existing_eval_rows_csv) if spec.existing_eval_rows_csv.exists() else pd.DataFrame()
        existing_eval = _attach_missing_tags(existing_eval, selection_rows)
        missing = _missing_eval_rows(selection_rows, existing_eval)
        if missing.empty:
            combined_eval = existing_eval.drop_duplicates(subset=["regime", "seed", "epoch", "tag"]).copy()
        else:
            missing_csv = out_dir / f"{spec.suite}_selector_missing_eval_trueval_{args.stamp}.csv"
            missing_out_csv = out_dir / f"{spec.suite}_selector_missing_eval_rows_trueval_{args.stamp}.csv"
            missing_out_summary = out_dir / f"{spec.suite}_selector_missing_eval_summary_trueval_{args.stamp}.csv"
            missing.to_csv(missing_csv, index=False)
            _run_domain_eval(root, spec, missing_csv, missing_out_csv, missing_out_summary)
            new_eval = pd.read_csv(missing_out_csv)
            combined_eval = pd.concat([existing_eval, new_eval], ignore_index=True)
            combined_eval = combined_eval.drop_duplicates(subset=["regime", "seed", "epoch", "tag"]).reset_index(drop=True)

        combined_eval = _attach_missing_tags(combined_eval, selection_rows)
        combined_eval.to_csv(heldout_rows_csv, index=False)
        suite_summary = _summarize_suite(spec, selection_rows, heldout_rows_csv)
        suite_summary.to_csv(out_dir / f"{spec.suite}_selector_summary_trueval_{args.stamp}.csv", index=False)
        suite_summaries.append(suite_summary)

    combined = pd.concat(suite_summaries, ignore_index=True)
    suite_order = {"Camelyon17 ERM": 0, "Camelyon17 Finetune": 1}
    policy_order = {"Baseline": 0, "Proxy-only": 1, "Val-loss-only": 2, "1.25x guardrail": 3}
    combined["suite_order"] = combined["suite_label"].map(suite_order)
    combined["policy_order"] = combined["selector_label"].map(policy_order)
    combined = combined.sort_values(["suite_order", "policy_order"]).drop(columns=["suite_order", "policy_order"]).reset_index(drop=True)

    out_csv = out_dir / f"selector_comparison_trueval_{args.stamp}.csv"
    combined.to_csv(out_csv, index=False)
    _write_tex(combined, root / args.out_tex, root)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {root / args.out_tex}")


if __name__ == "__main__":
    main()
