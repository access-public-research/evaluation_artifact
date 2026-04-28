import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


def _policy_frame(path: Path, suite: str, policy: str) -> pd.DataFrame:
    df = _dedup(pd.read_csv(path))
    return df[(df["suite"] == suite) & (df["selection_policy"] == policy)].copy()


def _mean(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").dropna().mean())


def _suite_rows(spec: SweepSpec) -> list[dict]:
    rows = []
    proxy_row = None
    for budget, path in spec.paths.items():
        base = _policy_frame(path, spec.suite, "baseline")
        proxy = _policy_frame(path, spec.suite, "proxy_only")
        guard = _policy_frame(path, spec.suite, "guardrail")

        common_proxy = sorted(set(base["seed"]) & set(proxy["seed"]))
        common_guard = sorted(set(base["seed"]) & set(guard["seed"]))
        base_proxy = base[base["seed"].isin(common_proxy)].drop_duplicates(subset=["seed"]).set_index("seed")
        proxy = proxy[proxy["seed"].isin(common_proxy)].drop_duplicates(subset=["seed"]).set_index("seed")
        base_guard = base[base["seed"].isin(common_guard)].drop_duplicates(subset=["seed"]).set_index("seed")
        guard = guard[guard["seed"].isin(common_guard)].drop_duplicates(subset=["seed"]).set_index("seed")

        current_proxy = {
            "suite_label": spec.label,
            "rule_label": "Proxy-only",
            "budget_ratio": budget,
            "accept_rate": 1.0,
            "delta_loss": _mean(proxy[spec.heldout_loss_col] - base_proxy[spec.heldout_loss_col]),
            "delta_acc": _mean(proxy[spec.heldout_acc_col] - base_proxy[spec.heldout_acc_col]),
            "delta_tail": _mean(proxy["selected_tail_worst_cvar"] - base_proxy["selected_tail_worst_cvar"]),
        }
        if proxy_row is None:
            proxy_row = current_proxy

        rows.append(
            {
                "suite_label": spec.label,
                "rule_label": f"{budget:.2f}x budget",
                "budget_ratio": budget,
                "accept_rate": 1.0 - _mean(guard["fallback_to_baseline"].astype(float)),
                "delta_loss": _mean(guard[spec.heldout_loss_col] - base_guard[spec.heldout_loss_col]),
                "delta_acc": _mean(guard[spec.heldout_acc_col] - base_guard[spec.heldout_acc_col]),
                "delta_tail": _mean(guard["selected_tail_worst_cvar"] - base_guard["selected_tail_worst_cvar"]),
            }
        )
    if proxy_row is not None:
        rows.insert(0, proxy_row)
    return rows


def _write_tex(path: Path, df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"  \toprule",
        r"  Suite & Rule & Accept rate & $\Delta$Held-out loss$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $\Delta$Tail CVaR$\downarrow$ \\",
        r"  \midrule",
    ]
    last_suite = None
    for _, row in df.iterrows():
        suite = row["suite_label"]
        suite_cell = suite if suite != last_suite else ""
        last_suite = suite
        lines.append(
            "  {suite} & {rule} & {acc_rate:.2f} & {dloss:+.3f} & {dacc:+.3f} & {dtail:+.3f} \\\\".format(
                suite=suite_cell,
                rule=row["rule_label"],
                acc_rate=row["accept_rate"],
                dloss=row["delta_loss"],
                dacc=row["delta_acc"],
                dtail=row["delta_tail"],
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out_path: Path, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    colors = {"Camelyon17 ERM": "#2563eb", "Camelyon17 Finetune": "#dc2626"}
    for ax, metric, ylabel in [
        (axes[0], "delta_loss", "Held-out loss delta"),
        (axes[1], "delta_tail", "Tail CVaR delta"),
    ]:
        for suite, sub in df[df["rule_label"] != "Proxy-only"].groupby("suite_label"):
            sub = sub.sort_values("budget_ratio")
            ax.plot(sub["budget_ratio"], sub[metric], marker="o", linewidth=2.0, color=colors[suite], label=suite)
            proxy_ref = float(df[(df["suite_label"] == suite) & (df["rule_label"] == "Proxy-only")][metric].iloc[0])
            ax.axhline(proxy_ref, color=colors[suite], linestyle="--", linewidth=1.2, alpha=0.5)
        ax.set_xlabel("Validation-loss budget ratio")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", frameon=False)
    fig.suptitle("Budgeted guardrails trade safety against retained gains", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if out_path.suffix.lower() == ".png":
        save_kwargs["dpi"] = 220
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    rows = []
    for spec in _specs(_repo_root()):
        rows.extend(_suite_rows(spec))
    out = pd.DataFrame(rows)
    suite_order = {"Camelyon17 ERM": 0, "Camelyon17 Finetune": 1}
    rule_order = {"Proxy-only": 0, "1.10x budget": 1, "1.25x budget": 2, "1.50x budget": 3}
    out["suite_order"] = out["suite_label"].map(suite_order)
    out["rule_order"] = out["rule_label"].map(rule_order)
    out = out.sort_values(["suite_order", "rule_order"]).drop(columns=["suite_order", "rule_order"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_tex(out_tex, out)
    _plot(out_pdf, out)
    _plot(out_png, out)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")
    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
