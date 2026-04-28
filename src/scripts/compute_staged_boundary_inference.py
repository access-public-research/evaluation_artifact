import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REGIME_ORDER = ["P95", "P97", "P99"]
REGIME_INDEX = {name: i + 1 for i, name in enumerate(REGIME_ORDER)}


def _bootstrap_mean_ci(values: List[float], n_boot: int, rng_seed: int) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    ser = pd.Series(values, dtype="float64")
    mean = float(ser.mean())
    if len(values) == 1:
        return mean, mean, mean

    rng = pd.Series(range(n_boot)).sample(n=n_boot, replace=False, random_state=rng_seed).index
    # deterministic bootstrap via seeded random states per draw
    boots = []
    for i in rng:
        sample = ser.sample(n=len(values), replace=True, random_state=int(i) + rng_seed)
        boots.append(float(sample.mean()))
    boots = pd.Series(boots, dtype="float64").sort_values(ignore_index=True)
    lo = float(boots.quantile(0.025))
    hi = float(boots.quantile(0.975))
    return mean, lo, hi


def _binom_one_sided_p_ge_half(k_success: int, n_total: int) -> float:
    if n_total <= 0:
        return float("nan")
    denom = 2**n_total
    num = 0
    for i in range(k_success, n_total + 1):
        num += math.comb(n_total, i)
    return float(num / denom)


def _load_selected_rows(path: Path, regime_map: Dict[str, str], perf_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["selection_mode"] == "selected_best_proxy"].copy()
    df = df[df["regime"].isin(regime_map)].copy()
    df["regime_label"] = df["regime"].map(regime_map)
    keep = ["seed", "regime_label", "tail_worst_cvar", "frac_clipped_val", perf_col]
    df = df[keep].rename(columns={perf_col: "perf"})
    return df


def _merge_perf_override(df: pd.DataFrame, path: Path, perf_col: str, regime_map: Dict[str, str]) -> pd.DataFrame:
    perf = pd.read_csv(path)
    perf = perf[perf["regime"].isin(regime_map)].copy()
    perf["regime_label"] = perf["regime"].map(regime_map)
    perf = perf[["seed", "regime_label", perf_col]].rename(columns={perf_col: "perf"})
    base = df.drop(columns=["perf"], errors="ignore")
    merged = base.merge(perf, on=["seed", "regime_label"], how="inner", validate="one_to_one")
    return merged


def _load_camelyon_selected_rows(
    selected_rows_csv: Path,
    domain_acc_csv: Path,
    regime_map: Dict[str, str],
) -> pd.DataFrame:
    selected = pd.read_csv(selected_rows_csv)
    selected = selected[selected["selection_mode"] == "selected_best_proxy"].copy()
    selected = selected[selected["regime"].isin(regime_map)].copy()
    selected["regime_label"] = selected["regime"].map(regime_map)
    selected = selected[["seed", "regime_label", "tail_worst_cvar", "frac_clipped_val"]].copy()

    dom = pd.read_csv(domain_acc_csv)
    dom = dom[dom["regime"].isin(regime_map)].copy()
    dom["regime_label"] = dom["regime"].map(regime_map)
    dom = dom[["seed", "regime_label", "test_hosp_2_acc"]].rename(columns={"test_hosp_2_acc": "perf"})

    merged = selected.merge(dom, on=["seed", "regime_label"], how="inner", validate="one_to_one")
    return merged


def _seed_stage_rows(
    df: pd.DataFrame,
    eps_perf: float,
    eps_tail: float,
    dataset: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for seed, sdf in df.groupby("seed"):
        rec = {r["regime_label"]: r for _, r in sdf.iterrows()}
        needed = {"rcgdro", "P95", "P97", "P99"}
        if set(rec.keys()) != needed:
            continue
        base = rec["rcgdro"]
        perf_thr = float(base["perf"]) - eps_perf
        tail_thr = float(base["tail_worst_cvar"]) + eps_tail

        perf_rec = None
        perf_u = float("nan")
        tail_rec = None
        tail_u = float("nan")
        for regime in REGIME_ORDER:
            rr = rec[regime]
            if perf_rec is None and float(rr["perf"]) >= perf_thr:
                perf_rec = regime
                perf_u = float(rr["frac_clipped_val"])
            if tail_rec is None and float(rr["tail_worst_cvar"]) <= tail_thr:
                tail_rec = regime
                tail_u = float(rr["frac_clipped_val"])

        status = "none_reached"
        delta_idx = float("nan")
        if perf_rec is not None and tail_rec is not None:
            delta_idx = float(REGIME_INDEX[perf_rec] - REGIME_INDEX[tail_rec])
            if delta_idx < 0:
                status = "staged_explicit"
            elif delta_idx == 0:
                status = "same_point"
            else:
                status = "tail_before_perf"
        elif perf_rec is not None and tail_rec is None:
            status = "staged_directional_tail_unreached"
        elif perf_rec is None and tail_rec is not None:
            status = "tail_before_perf"

        out.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "eps_perf": eps_perf,
                "eps_tail": eps_tail,
                "u_perf_rec_regime": perf_rec if perf_rec is not None else "not_reached",
                "u_tail_rec_regime": tail_rec if tail_rec is not None else "not_reached",
                "u_perf_rec_fracclip": perf_u,
                "u_tail_rec_fracclip": tail_u,
                "delta_u_index": delta_idx,
                "status": status,
            }
        )
    return out


def _write_tex_table(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "  \\toprule",
        "  Dataset & $n$ & $n_{\\Delta u}$ & Mean $\\Delta u$ (95\\% boot CI) & $\\Pr(\\Delta u<0)$ & Sign-test $p$ & Directional-only \\\\",
        "  \\midrule",
    ]
    for r in rows:
        if math.isnan(float(r["mean_delta"])):
            delta_txt = "N/A"
            pneg_txt = "N/A"
            pval_txt = "N/A"
        else:
            delta_txt = f"{r['mean_delta']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
            pneg_txt = f"{r['p_neg']:.2f}"
            pval_txt = f"{r['p_sign']:.3f}"
        lines.append(
            f"  {r['dataset']} & {int(r['n_total'])} & {int(r['n_delta'])} & {delta_txt} & {pneg_txt} & {pval_txt} & {int(r['n_directional'])} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics")
    ap.add_argument("--tables_dir", default="paper/neurips2026_selection_risk/tables")
    ap.add_argument("--eps_perf", type=float, default=0.02)
    ap.add_argument("--eps_tail", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--rng_seed", type=int, default=20260302)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    celeba = _load_selected_rows(
        metrics_dir / "celeba_selected_vs_epoch30_rows_v7confclip_p60_p95_p97_p99_10s.csv",
        regime_map={
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10": "P95",
            "rcgdro_softclip_p97_a10": "P97",
            "rcgdro_softclip_p99_a10": "P99",
        },
        perf_col="oracle_wg_acc",
    )
    celeba = _merge_perf_override(
        celeba,
        metrics_dir / "celeba_test_wg_selected_v7confclip_p60_p95_p97_p99_10s_20260308.csv",
        perf_col="test_oracle_wg_acc",
        regime_map={
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10": "P95",
            "rcgdro_softclip_p97_a10": "P97",
            "rcgdro_softclip_p99_a10": "P99",
        },
    )
    camelyon = _load_camelyon_selected_rows(
        metrics_dir / "camelyon17_selected_vs_epoch30_rows_cam_softclip_a10_p99_20260207.csv",
        metrics_dir / "camelyon17_resnet50_domain_acc_cam_softclip_a10_p99_20260207.csv",
        regime_map={
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10_cam": "P95",
            "rcgdro_softclip_p97_a10_cam": "P97",
            "rcgdro_softclip_p99_a10_cam": "P99",
        },
    )

    seed_rows: List[Dict[str, object]] = []
    seed_rows.extend(_seed_stage_rows(celeba, args.eps_perf, args.eps_tail, "CelebA"))
    seed_rows.extend(_seed_stage_rows(camelyon, args.eps_perf, args.eps_tail, "Camelyon17"))
    seed_df = pd.DataFrame(seed_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)

    summary_rows: List[Dict[str, object]] = []
    for dataset, sdf in seed_df.groupby("dataset"):
        comp = sdf[sdf["delta_u_index"].notna()].copy()
        directional = int((sdf["status"] == "staged_directional_tail_unreached").sum())
        n_total = int(sdf.shape[0])
        n_delta = int(comp.shape[0])
        if n_delta > 0:
            vals = comp["delta_u_index"].astype(float).tolist()
            mean_delta, ci_lo, ci_hi = _bootstrap_mean_ci(vals, args.n_boot, args.rng_seed)
            k_neg = int((comp["delta_u_index"] < 0).sum())
            p_neg = float(k_neg / n_delta)
            p_sign = _binom_one_sided_p_ge_half(k_neg, n_delta)
        else:
            mean_delta = float("nan")
            ci_lo = float("nan")
            ci_hi = float("nan")
            p_neg = float("nan")
            p_sign = float("nan")
        summary_rows.append(
            {
                "dataset": dataset,
                "n_total": n_total,
                "n_delta": n_delta,
                "n_directional": directional,
                "mean_delta": mean_delta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_neg": p_neg,
                "p_sign": p_sign,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    order = {"CelebA": 0, "Camelyon17": 1}
    summary_df["__ord"] = summary_df["dataset"].map(order).fillna(999)
    summary_df = summary_df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    suffix = f"epsperf{str(args.eps_perf).replace('.', 'p')}_epstail{str(args.eps_tail).replace('.', 'p')}_20260302"
    seed_csv = metrics_dir / f"staged_boundary_inference_seed_rows_head_{suffix}.csv"
    summary_csv = metrics_dir / f"staged_boundary_inference_summary_head_{suffix}.csv"
    seed_df.to_csv(seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    _write_tex_table(
        tables_dir / "table_staged_boundary_inference.tex",
        summary_df.to_dict(orient="records"),
    )
    print(f"[staged-inference] wrote {seed_csv}")
    print(f"[staged-inference] wrote {summary_csv}")
    print(f"[staged-inference] wrote {tables_dir / 'table_staged_boundary_inference.tex'}")


if __name__ == "__main__":
    main()
