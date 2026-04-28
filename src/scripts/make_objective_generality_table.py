import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _proxy(row: pd.Series) -> float:
    v = row.get("proxy_worst_loss_clip_mean")
    if pd.notna(v):
        return float(v)
    return float(row["proxy_worst_loss_mean"])


def _delta(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}"


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cam_domain_mean(dom_csv: Path, regime: str) -> float:
    dom = pd.read_csv(dom_csv)
    x = pd.to_numeric(dom[dom["regime"] == regime]["test_hosp_2_acc"], errors="coerce")
    return float(x.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core_head_csv", required=True)
    ap.add_argument("--core_head_domain_csv", required=True)
    ap.add_argument("--groupdro_true_celeba_csv", required=True)
    ap.add_argument("--groupdro_domain_cam_csv", required=True)
    ap.add_argument("--groupdro_domain_cam_domain_csv", required=True)
    ap.add_argument("--erm_cam_csv", required=True)
    ap.add_argument("--erm_cam_domain_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    core = pd.read_csv(args.core_head_csv)
    gtrue_c = pd.read_csv(args.groupdro_true_celeba_csv)
    gcam = pd.read_csv(args.groupdro_domain_cam_csv)
    erm = pd.read_csv(args.erm_cam_csv)

    # 1) PseudoGroupDRO (Camelyon head-only)
    b = core[core["regime"] == "rcgdro"].iloc[0]
    r = core[core["regime"] == "rcgdro_softclip_p95_a10_cam"].iloc[0]
    row_pg = (
        "PseudoGroupDRO",
        "Camelyon17",
        "P95",
        _delta(_proxy(r) - _proxy(b), 3),
        _delta(float(r["tail_worst_cvar_mean"]) - float(b["tail_worst_cvar_mean"]), 2),
        _delta(_cam_domain_mean(Path(args.core_head_domain_csv), "rcgdro_softclip_p95_a10_cam")
               - _cam_domain_mean(Path(args.core_head_domain_csv), "rcgdro"), 3),
        "test-hosp2",
    )

    # 2) True-group GroupDRO (CelebA)
    b = gtrue_c[gtrue_c["regime"] == "rcgdro"].iloc[0]
    r = gtrue_c[gtrue_c["regime"] == "rcgdro_softclip_p95_a10"].iloc[0]
    row_tg = (
        "True-group GroupDRO",
        "CelebA",
        "P95",
        _delta(_proxy(r) - _proxy(b), 3),
        _delta(float(r["tail_worst_cvar_mean"]) - float(b["tail_worst_cvar_mean"]), 2),
        _delta(float(r["val_overall_acc_mean"]) - float(b["val_overall_acc_mean"]), 3),
        "val acc",
    )

    # 3) Domain-group GroupDRO (Camelyon)
    b = gcam[gcam["regime"] == "rcgdro"].iloc[0]
    r = gcam[gcam["regime"] == "rcgdro_softclip_p95_a10_cam"].iloc[0]
    row_dg = (
        "Domain-group GroupDRO",
        "Camelyon17",
        "P95",
        _delta(_proxy(r) - _proxy(b), 3),
        _delta(float(r["tail_worst_cvar_mean"]) - float(b["tail_worst_cvar_mean"]), 2),
        _delta(_cam_domain_mean(Path(args.groupdro_domain_cam_domain_csv), "rcgdro_softclip_p95_a10_cam")
               - _cam_domain_mean(Path(args.groupdro_domain_cam_domain_csv), "rcgdro"), 3),
        "test-hosp2",
    )

    # 4) ERM (Camelyon, corrected selected)
    b = erm[erm["regime"] == "erm"].iloc[0]
    r = erm[erm["regime"] == "erm_softclip_p95_a10_cam"].iloc[0]
    row_erm = (
        "ERM",
        "Camelyon17",
        "P95",
        _delta(_proxy(r) - _proxy(b), 3),
        _delta(float(r["tail_worst_cvar_mean"]) - float(b["tail_worst_cvar_mean"]), 2),
        _delta(_cam_domain_mean(Path(args.erm_cam_domain_csv), "erm_softclip_p95_a10_cam")
               - _cam_domain_mean(Path(args.erm_cam_domain_csv), "erm"), 3),
        "test-hosp2",
    )

    rows = [row_pg, row_tg, row_dg, row_erm]

    lines = [
        r"\begin{tabular}{llccccc}",
        r"  \toprule",
        r"  Training family & Dataset & Regime & $\Delta$Proxy$\downarrow$ & $\Delta$Tail$\downarrow$ & $\Delta$Perf$\uparrow$ & Perf metric \\",
        r"  \midrule",
    ]
    for rf in rows:
        lines.append(
            f"  {rf[0]} & {rf[1]} & {rf[2]} & {rf[3]} & {rf[4]} & {rf[5]} & {rf[6]} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    _write(Path(args.out_tex), lines)
    print("[objective-generality-table] wrote", args.out_tex)


if __name__ == "__main__":
    main()
