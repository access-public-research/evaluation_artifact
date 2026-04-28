import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..utils.stats import ci95_mean


REG_LABEL = {
    "rcgdro": "rcgdro",
    "rcgdro_softclip_p95_a10": "P95",
    "rcgdro_softclip_p97_a10": "P97",
    "rcgdro_softclip_p99_a10": "P99",
    "rcgdro_softclip_p95_a10_wb": "P95",
    "rcgdro_softclip_p97_a10_wb": "P97",
    "rcgdro_softclip_p99_a10_wb": "P99",
    "rcgdro_softclip_p95_a10_wb_h256cal": "P95",
    "rcgdro_softclip_p97_a10_wb_h256cal": "P97",
    "rcgdro_softclip_p99_a10_wb_h256cal": "P99",
    "rcgdro_softclip_p95_a10_cam": "P95",
    "rcgdro_softclip_p97_a10_cam": "P97",
    "rcgdro_softclip_p99_a10_cam": "P99",
}


def _fmt_pm(mean: float, ci: float, nd=3) -> str:
    return f"{mean:.{nd}f} $\\pm$ {ci:.{nd}f}"


def _fmt_delta(v: float, nd=3) -> str:
    return f"({v:+.{nd}f})"


def _write(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _agg_domain(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "test_hosp_2_acc_mean" in d.columns:
        cols = ["regime", "test_hosp_2_acc_mean", "test_hosp_2_acc_ci"]
        if "test_hosp_2_loss_mean" in d.columns and "test_hosp_2_loss_ci" in d.columns:
            cols.extend(["test_hosp_2_loss_mean", "test_hosp_2_loss_ci"])
        return d[cols].copy()
    rows = []
    for reg, g in d.groupby("regime"):
        x = pd.to_numeric(g["test_hosp_2_acc"], errors="coerce")
        n = int(x.notna().sum())
        ci = ci95_mean(x.to_numpy(dtype=np.float64)) if n > 1 else 0.0
        rec = {
            "regime": reg,
            "test_hosp_2_acc_mean": float(x.mean()),
            "test_hosp_2_acc_ci": ci,
        }
        if "test_hosp_2_loss" in g.columns:
            xl = pd.to_numeric(g["test_hosp_2_loss"], errors="coerce")
            nl = int(xl.notna().sum())
            lci = ci95_mean(xl.to_numpy(dtype=np.float64)) if nl > 1 else 0.0
            rec["test_hosp_2_loss_mean"] = float(xl.mean())
            rec["test_hosp_2_loss_ci"] = lci
        rows.append(rec)
    return pd.DataFrame(rows)


def _load_perf_summary(path: Path, mean_col: str, ci_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["regime", mean_col, ci_col]].copy()


def _merge_perf(df: pd.DataFrame, perf_df: pd.DataFrame, mean_col: str, ci_col: str) -> pd.DataFrame:
    out = df.merge(perf_df, on="regime", how="left", validate="one_to_one")
    if out[mean_col].isna().any():
        missing = out.loc[out[mean_col].isna(), "regime"].tolist()
        raise ValueError(f"Missing performance rows for regimes: {missing}")
    return out


def _pick_proxy(row: pd.Series) -> Tuple[float, float]:
    if str(row["regime"]) == "rcgdro":
        return float(row["proxy_worst_loss_mean"]), float(row["proxy_worst_loss_ci"])
    return float(row["proxy_worst_loss_clip_mean"]), float(row["proxy_worst_loss_clip_ci"])


def build_core_table(args):
    c = pd.read_csv(args.celeba_effect)
    c_perf = pd.read_csv(Path(args.celeba_test_selected))
    needed = [
        "regime",
        "test_oracle_wg_acc_mean",
        "test_oracle_wg_acc_ci",
        "test_oracle_wg_loss_mean",
        "test_oracle_wg_loss_ci",
    ]
    missing = [col for col in needed if col not in c_perf.columns]
    if missing:
        raise ValueError(f"CelebA selected summary missing required columns: {missing}")
    c = c.merge(c_perf[needed], on="regime", how="left", validate="one_to_one")
    m = pd.read_csv(args.camelyon_effect)
    md = _agg_domain(Path(args.camelyon_domain))

    datasets = [
        ("CelebA", c, "test_oracle_wg_acc", "test_oracle_wg_loss"),
        ("Camelyon17", m.merge(md, on="regime", how="left"), "test_hosp_2_acc", "test_hosp_2_loss"),
    ]

    lines = [
        "\\begin{tabular}{llccccc}",
        "  \\toprule",
        "  Dataset & Regime & FracClip & Proxy$\\downarrow$ & Tail CVaR$\\downarrow$ ($\\Delta$) & Held-out loss$\\downarrow$ ($\\Delta$) & Held-out acc$\\uparrow$ ($\\Delta$) \\\\",
        "  \\midrule",
    ]
    for di, (dname, df, perf_key, loss_key) in enumerate(datasets):
        base = df[df["regime"] == "rcgdro"].iloc[0]
        base_tail = float(base["tail_worst_cvar_mean"])
        if perf_key == "test_oracle_wg_acc":
            base_perf = float(base["test_oracle_wg_acc_mean"])
        else:
            base_perf = float(base["test_hosp_2_acc_mean"])
        base_loss = float(base[f"{loss_key}_mean"])
        for r in ["rcgdro", *[k for k in REG_LABEL if k.startswith("rcgdro_softclip") and (("_wb" in k) == (dname == "Waterbirds")) and (("_cam" in k) == (dname == "Camelyon17")) and ("_wb" not in k or dname == "Waterbirds") and ("_cam" not in k or dname == "Camelyon17")]]:
            if r not in set(df["regime"]):
                continue
            row = df[df["regime"] == r].iloc[0]
            reg_lbl = REG_LABEL.get(r, r)
            if r == "rcgdro":
                frac_txt = "0.000"
            else:
                frac_txt = _fmt_pm(float(row["frac_clipped_val_mean"]), float(row["frac_clipped_val_ci"]), nd=3)
            pmean, pci = _pick_proxy(row)
            proxy_txt = _fmt_pm(pmean, pci, nd=3)
            tail = float(row["tail_worst_cvar_mean"])
            tail_ci = float(row["tail_worst_cvar_ci"])
            tail_txt = f"{_fmt_pm(tail, tail_ci, nd=2)} {_fmt_delta(tail - base_tail, nd=2)}"
            loss = float(row[f"{loss_key}_mean"])
            loss_ci = float(row[f"{loss_key}_ci"])
            loss_txt = f"{_fmt_pm(loss, loss_ci, nd=3)} {_fmt_delta(loss - base_loss, nd=3)}"
            if perf_key == "test_oracle_wg_acc":
                perf = float(row["test_oracle_wg_acc_mean"])
                perf_ci = float(row["test_oracle_wg_acc_ci"])
            else:
                perf = float(row["test_hosp_2_acc_mean"])
                perf_ci = float(row["test_hosp_2_acc_ci"])
            perf_txt = f"{_fmt_pm(perf, perf_ci, nd=3)} {_fmt_delta(perf - base_perf, nd=3)}"
            lines.append(f"  {dname} & {reg_lbl} & {frac_txt} & {proxy_txt} & {tail_txt} & {loss_txt} & {perf_txt} \\\\")
        if di < len(datasets) - 1:
            lines.append("  \\midrule")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    _write(Path(args.out_core_tex), lines)


def build_groupdro_true_table(args):
    c = pd.read_csv(args.celeba_groupdro_true)
    w = pd.read_csv(args.waterbirds_groupdro_true)
    c_perf = _load_perf_summary(
        Path(args.celeba_groupdro_true_test_selected),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    )
    w_perf = _load_perf_summary(
        Path(args.waterbirds_groupdro_true_test_selected),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    )
    c = _merge_perf(c, c_perf, "test_oracle_wg_acc_mean", "test_oracle_wg_acc_ci")
    w = _merge_perf(w, w_perf, "test_oracle_wg_acc_mean", "test_oracle_wg_acc_ci")
    all_rows = []
    for dname, df in [("CelebA", c), ("Waterbirds", w)]:
        base = df[df["regime"] == "rcgdro"].iloc[0]
        bproxy, _ = _pick_proxy(base)
        btail = float(base["tail_worst_cvar_mean"])
        bperf = float(base["test_oracle_wg_acc_mean"])
        regs = ["rcgdro"] + [
            k
            for k in REG_LABEL
            if k != "rcgdro" and ("_wb" in k) == (dname == "Waterbirds") and ("_cam" not in k)
        ]
        for r in regs:
            if r not in set(df["regime"]):
                continue
            row = df[df["regime"] == r].iloc[0]
            lbl = REG_LABEL.get(r, r)
            frac_txt = "0.000 $\\pm$ 0.000" if r == "rcgdro" else _fmt_pm(float(row["frac_clipped_val_mean"]), float(row["frac_clipped_val_ci"]), nd=3)
            pmean, pci = _pick_proxy(row)
            proxy_txt = f"{_fmt_pm(pmean, pci, nd=3)} {_fmt_delta(pmean - bproxy, nd=3)}"
            tmean = float(row["tail_worst_cvar_mean"])
            tci = float(row["tail_worst_cvar_ci"])
            tail_txt = f"{_fmt_pm(tmean, tci, nd=2)} {_fmt_delta(tmean - btail, nd=2)}"
            vmean = float(row["test_oracle_wg_acc_mean"])
            vci = float(row["test_oracle_wg_acc_ci"])
            perf_txt = f"{_fmt_pm(vmean, vci, nd=3)} {_fmt_delta(vmean - bperf, nd=3)}"
            all_rows.append((dname, lbl, frac_txt, proxy_txt, tail_txt, perf_txt))

    lines = [
        "\\begin{tabular}{llcccc}",
        "  \\toprule",
        "  Dataset & Regime & FracClip & Proxy$\\downarrow$ ($\\Delta$) & Tail CVaR$\\downarrow$ ($\\Delta$) & Test WG$\\uparrow$ ($\\Delta$) \\\\",
        "  \\midrule",
    ]
    last = None
    for d, lbl, frac, proxy, tail, perf in all_rows:
        if last is not None and d != last:
            lines.append("  \\midrule")
        dcell = d if d != last else " "
        lines.append(f"  {dcell} & {lbl} & {frac} & {proxy} & {tail} & {perf} \\\\")
        last = d
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    _write(Path(args.out_groupdro_true_tex), lines)

    # Also write combined CSV used in snapshots.
    c2 = c.copy()
    c2.insert(0, "dataset", "celeba")
    w2 = w.copy()
    w2.insert(0, "dataset", "waterbirds")
    pd.concat([c2, w2], ignore_index=True).to_csv(args.out_groupdro_true_csv, index=False)


def build_groupdro_cam_table(args):
    eff = pd.read_csv(args.cam_groupdro_effect)
    dom = _agg_domain(Path(args.cam_groupdro_domain))
    df = eff.merge(dom, on="regime", how="left")
    base = df[df["regime"] == "rcgdro"].iloc[0]
    bproxy, _ = _pick_proxy(base)
    btail = float(base["tail_worst_cvar_mean"])
    btest = float(base["test_hosp_2_acc_mean"])
    rows = []
    for r in ["rcgdro", "rcgdro_softclip_p95_a10_cam", "rcgdro_softclip_p97_a10_cam", "rcgdro_softclip_p99_a10_cam"]:
        row = df[df["regime"] == r].iloc[0]
        lbl = REG_LABEL.get(r, r)
        frac = "0.000 $\\pm$ 0.000" if r == "rcgdro" else _fmt_pm(float(row["frac_clipped_val_mean"]), float(row["frac_clipped_val_ci"]), nd=3)
        pmean, pci = _pick_proxy(row)
        proxy = f"{_fmt_pm(pmean, pci, nd=3)} {_fmt_delta(pmean - bproxy, nd=3)}"
        tmean = float(row["tail_worst_cvar_mean"])
        tci = float(row["tail_worst_cvar_ci"])
        tail = f"{_fmt_pm(tmean, tci, nd=2)} {_fmt_delta(tmean - btail, nd=2)}"
        dmean = float(row["test_hosp_2_acc_mean"])
        dci = float(row["test_hosp_2_acc_ci"])
        test = f"{_fmt_pm(dmean, dci, nd=3)} {_fmt_delta(dmean - btest, nd=3)}"
        rows.append((lbl, frac, proxy, tail, test))
    lines = [
        "\\begin{tabular}{lcccc}",
        "  \\toprule",
        "  Regime & FracClip & Proxy$\\downarrow$ ($\\Delta$) & Tail CVaR$\\downarrow$ ($\\Delta$) & Test-H2$\\uparrow$ ($\\Delta$) \\\\",
        "  \\midrule",
    ]
    for lbl, frac, proxy, tail, test in rows:
        lines.append(f"  {lbl} & {frac} & {proxy} & {tail} & {test} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    _write(Path(args.out_groupdro_cam_tex), lines)


def build_objective_generality_table(args):
    rows = []

    # Adaptive pseudo-group main sweeps
    celeba_ad = pd.read_csv(args.celeba_effect)
    celeba_ad_perf = pd.read_csv(Path(args.celeba_test_selected))
    need_cols = [
        "regime",
        "test_oracle_wg_acc_mean",
        "test_oracle_wg_loss_mean",
    ]
    celeba_ad = celeba_ad.merge(celeba_ad_perf[need_cols], on="regime", how="left", validate="one_to_one")
    base = celeba_ad[celeba_ad["regime"] == "rcgdro"].iloc[0]
    row = celeba_ad[celeba_ad["regime"] == "rcgdro_softclip_p95_a10"].iloc[0]
    pmean, _ = _pick_proxy(row)
    bproxy, _ = _pick_proxy(base)
    rows.append(
        (
            "Adaptive PseudoGroupDRO",
            "CelebA",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_oracle_wg_loss_mean"] - base["test_oracle_wg_loss_mean"]),
            float(row["test_oracle_wg_acc_mean"] - base["test_oracle_wg_acc_mean"]),
        )
    )

    cam_ad = pd.read_csv(args.camelyon_effect).merge(_agg_domain(Path(args.camelyon_domain)), on="regime", how="left")
    base = cam_ad[cam_ad["regime"] == "rcgdro"].iloc[0]
    row = cam_ad[cam_ad["regime"] == "rcgdro_softclip_p95_a10_cam"].iloc[0]
    pmean, _ = _pick_proxy(row)
    bproxy, _ = _pick_proxy(base)
    rows.append(
        (
            "Adaptive PseudoGroupDRO",
            "Camelyon17",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_hosp_2_loss_mean"] - base["test_hosp_2_loss_mean"]),
            float(row["test_hosp_2_acc_mean"] - base["test_hosp_2_acc_mean"]),
        )
    )

    # Fixed pseudo-group controls
    celeba_fx = pd.read_csv(args.celeba_fixed_effect)
    celeba_fx_perf = pd.read_csv(Path(args.celeba_fixed_test_selected))
    celeba_fx = celeba_fx.merge(celeba_fx_perf[need_cols], on="regime", how="left", validate="one_to_one")
    base = celeba_fx[celeba_fx["regime"] == "rcgdro"].iloc[0]
    row = celeba_fx[celeba_fx["regime"] == "rcgdro_softclip_p95_a10"].iloc[0]
    pmean, _ = _pick_proxy(row)
    bproxy, _ = _pick_proxy(base)
    rows.append(
        (
            "Fixed PseudoGroupDRO",
            "CelebA",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_oracle_wg_loss_mean"] - base["test_oracle_wg_loss_mean"]),
            float(row["test_oracle_wg_acc_mean"] - base["test_oracle_wg_acc_mean"]),
        )
    )

    cam_fx = pd.read_csv(args.cam_fixed_effect).merge(_agg_domain(Path(args.cam_fixed_domain)), on="regime", how="left")
    base = cam_fx[cam_fx["regime"] == "rcgdro"].iloc[0]
    row = cam_fx[cam_fx["regime"] == "rcgdro_softclip_p95_a10_cam"].iloc[0]
    pmean, _ = _pick_proxy(row)
    bproxy, _ = _pick_proxy(base)
    rows.append(
        (
            "Fixed PseudoGroupDRO",
            "Camelyon17",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_hosp_2_loss_mean"] - base["test_hosp_2_loss_mean"]),
            float(row["test_hosp_2_acc_mean"] - base["test_hosp_2_acc_mean"]),
        )
    )

    # Published / standard baselines
    celeba_true = pd.read_csv(args.celeba_groupdro_true)
    celeba_true_perf = pd.read_csv(Path(args.celeba_groupdro_true_test_selected))
    celeba_true = celeba_true.merge(celeba_true_perf[need_cols], on="regime", how="left", validate="one_to_one")
    base = celeba_true[celeba_true["regime"] == "rcgdro"].iloc[0]
    row = celeba_true[celeba_true["regime"] == "rcgdro_softclip_p95_a10"].iloc[0]
    pmean, _ = _pick_proxy(row)
    bproxy, _ = _pick_proxy(base)
    rows.append(
        (
            "True-group GroupDRO",
            "CelebA",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_oracle_wg_loss_mean"] - base["test_oracle_wg_loss_mean"]),
            float(row["test_oracle_wg_acc_mean"] - base["test_oracle_wg_acc_mean"]),
        )
    )

    cam_dom = pd.read_csv(args.cam_groupdro_effect).merge(_agg_domain(Path(args.cam_groupdro_domain)), on="regime", how="left")
    base = cam_dom[cam_dom["regime"] == "rcgdro"].iloc[0]
    row = cam_dom[cam_dom["regime"] == "rcgdro_softclip_p95_a10_cam"].iloc[0]
    pmean, _ = _pick_proxy(row)
    bproxy, _ = _pick_proxy(base)
    rows.append(
        (
            "Domain-group GroupDRO",
            "Camelyon17",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_hosp_2_loss_mean"] - base["test_hosp_2_loss_mean"]),
            float(row["test_hosp_2_acc_mean"] - base["test_hosp_2_acc_mean"]),
        )
    )

    erm = pd.read_csv(args.cam_erm_effect).merge(_agg_domain(Path(args.cam_erm_domain)), on="regime", how="left")
    base = erm[erm["regime"] == "erm"].iloc[0]
    row = erm[erm["regime"] == "erm_softclip_p95_a10_cam"].iloc[0]
    pmean = float(row["proxy_worst_loss_clip_mean"])
    bproxy = float(base["proxy_worst_loss_mean"])
    rows.append(
        (
            "ERM",
            "Camelyon17",
            pmean - bproxy,
            float(row["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"]),
            float(row["test_hosp_2_loss_mean"] - base["test_hosp_2_loss_mean"]),
            float(row["test_hosp_2_acc_mean"] - base["test_hosp_2_acc_mean"]),
        )
    )

    lines = [
        "\\begin{tabular}{llcccc}",
        "  \\toprule",
        "  Training family & Dataset & $\\Delta$Proxy$\\downarrow$ & $\\Delta$Tail$\\downarrow$ & $\\Delta$Held-out loss$\\downarrow$ & $\\Delta$Perf$\\uparrow$ \\\\",
        "  \\midrule",
    ]
    for fam, ds, dproxy, dtail, dloss, dperf in rows:
        lines.append(f"  {fam} & {ds} & {dproxy:+.3f} & {dtail:+.2f} & {dloss:+.3f} & {dperf:+.3f} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    _write(Path(args.out_objective_generality_tex), lines)


def build_selected_fixed_table(args):
    c = pd.read_csv(args.celeba_sel_fixed)
    w = pd.read_csv(args.waterbirds_sel_fixed)
    m = pd.read_csv(args.camelyon_sel_fixed)
    c_sel_perf = _load_perf_summary(
        Path(args.celeba_test_selected),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    ).rename(
        columns={
            "test_oracle_wg_acc_mean": "perf_selected_mean",
            "test_oracle_wg_acc_ci": "perf_selected_ci",
        }
    )
    c_fix_perf = _load_perf_summary(
        Path(args.celeba_test_fixed),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    ).rename(
        columns={
            "test_oracle_wg_acc_mean": "perf_fixed_mean",
            "test_oracle_wg_acc_ci": "perf_fixed_ci",
        }
    )
    w_sel_perf = _load_perf_summary(
        Path(args.waterbirds_test_selected),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    ).rename(
        columns={
            "test_oracle_wg_acc_mean": "perf_selected_mean",
            "test_oracle_wg_acc_ci": "perf_selected_ci",
        }
    )
    w_fix_perf = _load_perf_summary(
        Path(args.waterbirds_test_fixed),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    ).rename(
        columns={
            "test_oracle_wg_acc_mean": "perf_fixed_mean",
            "test_oracle_wg_acc_ci": "perf_fixed_ci",
        }
    )
    md_sel = pd.read_csv(args.camelyon_domain_selected)
    md_fix = pd.read_csv(args.camelyon_domain_fixed)
    # The Camelyon domain-accuracy CSVs already correspond to selected-best and
    # fixed-epoch-30 checkpoints, so we aggregate directly by regime here.
    perf_map = {
        "selected_best_proxy": md_sel.groupby("regime")["test_hosp_2_acc"].mean().to_dict(),
        "fixed_epoch_30": md_fix.groupby("regime")["test_hosp_2_acc"].mean().to_dict(),
    }

    def _fetch(df, mode, regime, perf_col):
        row = df[(df["selection_mode"] == mode) & (df["regime"] == regime)].iloc[0]
        return float(row["tail_worst_cvar_mean"]), float(row[perf_col])

    lines = [
        "\\begin{tabular}{llcccc}",
        "  \\toprule",
        "  Dataset & Regime & Tail (sel.)$\\downarrow$ & Tail (ep30)$\\downarrow$ & Perf (sel.)$\\uparrow$ & Perf (ep30)$\\uparrow$ \\\\",
        "  \\midrule",
    ]
    specs = [
        ("CelebA", c, c_sel_perf, c_fix_perf, "rcgdro", "rcgdro_softclip_p95_a10", "rcgdro_softclip_p99_a10"),
        ("Waterbirds", w, w_sel_perf, w_fix_perf, "rcgdro", "rcgdro_softclip_p95_a10_wb_h256cal", "rcgdro_softclip_p99_a10_wb_h256cal"),
    ]
    for di, (dname, df, perf_sel, perf_fix, r0, r95, r99) in enumerate(specs):
        for j, r in enumerate([r0, r95, r99]):
            lbl = "rcgdro" if r == r0 else REG_LABEL[r]
            ts, _ = _fetch(df, "selected_best_proxy", r, "oracle_wg_acc_mean")
            tf, _ = _fetch(df, "fixed_epoch_30", r, "oracle_wg_acc_mean")
            ps = float(perf_sel[perf_sel["regime"] == r]["perf_selected_mean"].iloc[0])
            pf = float(perf_fix[perf_fix["regime"] == r]["perf_fixed_mean"].iloc[0])
            dcell = dname if j == 0 else " "
            lines.append(f"  {dcell} & {lbl} & {ts:.2f} & {tf:.2f} & {ps:.3f} & {pf:.3f} \\\\")
        lines.append("  \\midrule")

    for j, r in enumerate(["rcgdro", "rcgdro_softclip_p95_a10_cam", "rcgdro_softclip_p99_a10_cam"]):
        row_sel = m[(m["selection_mode"] == "selected_best_proxy") & (m["regime"] == r)].iloc[0]
        row_fix = m[(m["selection_mode"] == "fixed_epoch_30") & (m["regime"] == r)].iloc[0]
        ts = float(row_sel["tail_worst_cvar_mean"])
        tf = float(row_fix["tail_worst_cvar_mean"])
        ps = float(perf_map["selected_best_proxy"][r])
        pf = float(perf_map["fixed_epoch_30"][r])
        lbl = "rcgdro" if r == "rcgdro" else REG_LABEL[r]
        dcell = "Camelyon17" if j == 0 else " "
        lines.append(f"  {dcell} & {lbl} & {ts:.2f} & {tf:.2f} & {ps:.3f} & {pf:.3f} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    _write(Path(args.out_sel_fixed_tex), lines)


def build_waterbirds_supportive_table(args):
    w = pd.read_csv(args.waterbirds_effect)
    w_perf = _load_perf_summary(
        Path(args.waterbirds_test_selected),
        mean_col="test_oracle_wg_acc_mean",
        ci_col="test_oracle_wg_acc_ci",
    )
    w_acc = _load_perf_summary(
        Path(args.waterbirds_test_selected),
        mean_col="test_overall_acc_mean",
        ci_col="test_overall_acc_ci",
    )
    w = _merge_perf(w, w_perf, "test_oracle_wg_acc_mean", "test_oracle_wg_acc_ci")
    w = _merge_perf(w, w_acc, "test_overall_acc_mean", "test_overall_acc_ci")
    base = w[w["regime"] == "rcgdro"].iloc[0]
    regs = [
        "rcgdro",
        "rcgdro_softclip_p95_a10_wb_h256cal",
        "rcgdro_softclip_p97_a10_wb_h256cal",
        "rcgdro_softclip_p99_a10_wb_h256cal",
    ]
    lines = [
        "\\begin{tabular}{lccccc}",
        "  \\toprule",
        "  Regime & FracClip & Proxy$\\downarrow$ & Tail CVaR$\\downarrow$ ($\\Delta$) & Oracle WG$\\uparrow$ ($\\Delta$) & Val Acc$\\uparrow$ ($\\Delta$) \\\\",
        "  \\midrule",
    ]
    btail = float(base["tail_worst_cvar_mean"])
    borg = float(base["test_oracle_wg_acc_mean"])
    bval = float(base["test_overall_acc_mean"])
    for r in regs:
        row = w[w["regime"] == r].iloc[0]
        lbl = "rcgdro" if r == "rcgdro" else REG_LABEL[r]
        frac = "0.000" if r == "rcgdro" else _fmt_pm(float(row["frac_clipped_val_mean"]), float(row["frac_clipped_val_ci"]), nd=3)
        pmean, pci = _pick_proxy(row)
        proxy = _fmt_pm(pmean, pci, nd=3)
        tail = f"{_fmt_pm(float(row['tail_worst_cvar_mean']), float(row['tail_worst_cvar_ci']), nd=2)} {_fmt_delta(float(row['tail_worst_cvar_mean']) - btail, nd=2)}"
        org = f"{_fmt_pm(float(row['test_oracle_wg_acc_mean']), float(row['test_oracle_wg_acc_ci']), nd=3)} {_fmt_delta(float(row['test_oracle_wg_acc_mean']) - borg, nd=3)}"
        val = f"{_fmt_pm(float(row['test_overall_acc_mean']), float(row['test_overall_acc_ci']), nd=3)} {_fmt_delta(float(row['test_overall_acc_mean']) - bval, nd=3)}"
        lines.append(f"  {lbl} & {frac} & {proxy} & {tail} & {org} & {val} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    _write(Path(args.out_wb_support_tex), lines)


def build_anchor_tables(args):
    def _one(in_csv, out_tex, mapping):
        df = pd.read_csv(in_csv)
        fam_map = {
            "teacher_difficulty": "TeacherDiff Tail$\\downarrow$",
            "decoupled_proj": "DecProj Tail$\\downarrow$",
            "global_hash": "GlobalHash Tail$\\downarrow$",
        }
        regs = [k for k in mapping if k in set(df["regime"])]
        base_reg = regs[0]
        base_vals: Dict[str, float] = {}
        for fam in fam_map:
            v = float(df[(df["regime"] == base_reg) & (df["family"] == fam)]["tail_metric_mean"].iloc[0])
            base_vals[fam] = v
        lines = [
            "\\begin{tabular}{lccc}",
            "  \\toprule",
            f"  Regime & {fam_map['teacher_difficulty']} & {fam_map['decoupled_proj']} & {fam_map['global_hash']} \\\\",
            "  \\midrule",
        ]
        for r in regs:
            lbl = mapping[r]
            cells = []
            for fam in ["teacher_difficulty", "decoupled_proj", "global_hash"]:
                row = df[(df["regime"] == r) & (df["family"] == fam)].iloc[0]
                mean = float(row["tail_metric_mean"])
                ci = float(row["tail_metric_ci"])
                d = mean - base_vals[fam]
                cells.append(f"{_fmt_pm(mean, ci, nd=2)} {_fmt_delta(d, nd=2)}")
            lines.append(f"  {lbl} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
        lines.extend(["  \\bottomrule", "\\end{tabular}"])
        _write(Path(out_tex), lines)

    _one(
        args.celeba_anchor_csv,
        args.out_anchor_tex,
        {
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10": "P95",
            "rcgdro_softclip_p97_a10": "P97",
            "rcgdro_softclip_p99_a10": "P99",
        },
    )
    _one(
        args.cam_anchor_csv,
        args.out_anchor_cam_tex,
        {
            "rcgdro": "rcgdro",
            "rcgdro_softclip_p95_a10_cam": "P95",
            "rcgdro_softclip_p97_a10_cam": "P97",
            "rcgdro_softclip_p99_a10_cam": "P99",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba_effect", required=True)
    ap.add_argument("--celeba_test_selected", required=True)
    ap.add_argument("--celeba_test_fixed", required=True)
    ap.add_argument("--waterbirds_effect", required=True)
    ap.add_argument("--waterbirds_test_selected", required=True)
    ap.add_argument("--waterbirds_test_fixed", required=True)
    ap.add_argument("--camelyon_effect", required=True)
    ap.add_argument("--camelyon_domain", required=True)
    ap.add_argument("--out_core_tex", required=True)

    ap.add_argument("--celeba_groupdro_true", required=True)
    ap.add_argument("--celeba_groupdro_true_test_selected", required=True)
    ap.add_argument("--waterbirds_groupdro_true", required=True)
    ap.add_argument("--waterbirds_groupdro_true_test_selected", required=True)
    ap.add_argument("--out_groupdro_true_tex", required=True)
    ap.add_argument("--out_groupdro_true_csv", required=True)

    ap.add_argument("--cam_groupdro_effect", required=True)
    ap.add_argument("--cam_groupdro_domain", required=True)
    ap.add_argument("--out_groupdro_cam_tex", required=True)

    ap.add_argument("--celeba_sel_fixed", required=True)
    ap.add_argument("--waterbirds_sel_fixed", required=True)
    ap.add_argument("--camelyon_sel_fixed", required=True)
    ap.add_argument("--camelyon_rows", required=True)
    ap.add_argument("--camelyon_domain_selected", required=True)
    ap.add_argument("--camelyon_domain_fixed", required=True)
    ap.add_argument("--out_sel_fixed_tex", required=True)
    ap.add_argument("--out_wb_support_tex", required=True)

    ap.add_argument("--celeba_anchor_csv", required=True)
    ap.add_argument("--cam_anchor_csv", required=True)
    ap.add_argument("--out_anchor_tex", required=True)
    ap.add_argument("--out_anchor_cam_tex", required=True)

    ap.add_argument("--celeba_fixed_effect", required=True)
    ap.add_argument("--celeba_fixed_test_selected", required=True)
    ap.add_argument("--cam_fixed_effect", required=True)
    ap.add_argument("--cam_fixed_domain", required=True)
    ap.add_argument("--cam_erm_effect", required=True)
    ap.add_argument("--cam_erm_domain", required=True)
    ap.add_argument("--out_objective_generality_tex", required=True)
    args = ap.parse_args()

    build_core_table(args)
    build_objective_generality_table(args)
    build_groupdro_true_table(args)
    build_groupdro_cam_table(args)
    build_selected_fixed_table(args)
    build_waterbirds_supportive_table(args)
    build_anchor_tables(args)
    print("[rebuild-paper-tables] done")


if __name__ == "__main__":
    main()
