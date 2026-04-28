import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


REGIME_ORDER = ["P95", "P97", "P99"]
IDX = {r: i for i, r in enumerate(REGIME_ORDER)}


def _first_perf_recovery(perf_map: Dict[str, float], base: float, eps_perf: float) -> str:
    thr = base - eps_perf
    for r in REGIME_ORDER:
        if perf_map[r] >= thr:
            return r
    return "not_reached"


def _first_tail_recovery(tail_map: Dict[str, float], base: float, eps_tail: float) -> str:
    thr = base + eps_tail
    for r in REGIME_ORDER:
        if tail_map[r] <= thr:
            return r
    return "not_reached"


def _status(u_perf: str, u_tail: str) -> str:
    if u_perf == "not_reached" and u_tail == "not_reached":
        return "none_reached"
    if u_perf != "not_reached" and u_tail == "not_reached":
        return "directional_staged"
    if u_perf == "not_reached" and u_tail != "not_reached":
        return "tail_before_perf"
    if IDX[u_perf] < IDX[u_tail]:
        return "explicit_staged"
    if IDX[u_perf] == IDX[u_tail]:
        return "same_point"
    return "tail_before_perf"


def _format_u(u: str) -> str:
    return "$>{}$P99" if u == "not_reached" else u


def _write_tex(path: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        "\\begin{tabular}{llccc}",
        "  \\toprule",
        "  Dataset & Anchor Family & $u_{\\text{perf-rec}}$ & $u_{\\text{tail-rec}}$ & Status \\\\",
        "  \\midrule",
    ]
    for r in rows:
        family_tex = str(r["family"]).replace("_", "\\_")
        lines.append(
            f"  {r['dataset']} & {family_tex} & {_format_u(r['u_perf'])} & {_format_u(r['u_tail'])} & {r['status_tex']} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics")
    ap.add_argument("--tables_dir", default="paper/neurips2026_selection_risk/tables")
    ap.add_argument("--eps_perf", type=float, default=0.02)
    ap.add_argument("--eps_tail", type=float, default=1.0)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    # CelebA head-only
    celeba_perf = pd.read_csv(metrics_dir / "celeba_effect_size_v7confclip_p60_p95_p97_p99_10s.csv")
    perf_map_celeba = {
        "P95": float(celeba_perf.loc[celeba_perf["regime"] == "rcgdro_softclip_p95_a10", "oracle_wg_acc_mean"].iloc[0]),
        "P97": float(celeba_perf.loc[celeba_perf["regime"] == "rcgdro_softclip_p97_a10", "oracle_wg_acc_mean"].iloc[0]),
        "P99": float(celeba_perf.loc[celeba_perf["regime"] == "rcgdro_softclip_p99_a10", "oracle_wg_acc_mean"].iloc[0]),
    }
    base_perf_celeba = float(celeba_perf.loc[celeba_perf["regime"] == "rcgdro", "oracle_wg_acc_mean"].iloc[0])
    u_perf_celeba = _first_perf_recovery(perf_map_celeba, base_perf_celeba, args.eps_perf)

    celeba_anchor = pd.read_csv(metrics_dir / "celeba_anchor_sensitivity_v9overhaul_20260221.csv")
    for family, fam_df in celeba_anchor.groupby("family"):
        tmap = {
            "P95": float(fam_df.loc[fam_df["regime"] == "rcgdro_softclip_p95_a10", "tail_metric_mean"].iloc[0]),
            "P97": float(fam_df.loc[fam_df["regime"] == "rcgdro_softclip_p97_a10", "tail_metric_mean"].iloc[0]),
            "P99": float(fam_df.loc[fam_df["regime"] == "rcgdro_softclip_p99_a10", "tail_metric_mean"].iloc[0]),
        }
        tbase = float(fam_df.loc[fam_df["regime"] == "rcgdro", "tail_metric_mean"].iloc[0])
        u_tail = _first_tail_recovery(tmap, tbase, args.eps_tail)
        st = _status(u_perf_celeba, u_tail)
        rows.append(
            {
                "dataset": "CelebA",
                "family": str(family),
                "u_perf": u_perf_celeba,
                "u_tail": u_tail,
                "status": st,
                "status_tex": st.replace("_", "\\_"),
            }
        )

    # Camelyon domain-group extension (where anchor sensitivity is available)
    cam_dom = pd.read_csv(metrics_dir / "camelyon17_resnet50_domain_acc_v8groupdrocamdom_10s_20260224.csv")
    perf_map_cam = {
        "P95": float(cam_dom.loc[cam_dom["regime"] == "rcgdro_softclip_p95_a10_cam", "test_hosp_2_acc"].mean()),
        "P97": float(cam_dom.loc[cam_dom["regime"] == "rcgdro_softclip_p97_a10_cam", "test_hosp_2_acc"].mean()),
        "P99": float(cam_dom.loc[cam_dom["regime"] == "rcgdro_softclip_p99_a10_cam", "test_hosp_2_acc"].mean()),
    }
    base_perf_cam = float(cam_dom.loc[cam_dom["regime"] == "rcgdro", "test_hosp_2_acc"].mean())
    u_perf_cam = _first_perf_recovery(perf_map_cam, base_perf_cam, args.eps_perf)

    cam_anchor = pd.read_csv(metrics_dir / "camelyon17_anchor_sensitivity_v8groupdrocamdom_10s_20260226.csv")
    for family, fam_df in cam_anchor.groupby("family"):
        tmap = {
            "P95": float(fam_df.loc[fam_df["regime"] == "rcgdro_softclip_p95_a10_cam", "tail_metric_mean"].iloc[0]),
            "P97": float(fam_df.loc[fam_df["regime"] == "rcgdro_softclip_p97_a10_cam", "tail_metric_mean"].iloc[0]),
            "P99": float(fam_df.loc[fam_df["regime"] == "rcgdro_softclip_p99_a10_cam", "tail_metric_mean"].iloc[0]),
        }
        tbase = float(fam_df.loc[fam_df["regime"] == "rcgdro", "tail_metric_mean"].iloc[0])
        u_tail = _first_tail_recovery(tmap, tbase, args.eps_tail)
        st = _status(u_perf_cam, u_tail)
        rows.append(
            {
                "dataset": "Camelyon17 (domain-group)",
                "family": str(family),
                "u_perf": u_perf_cam,
                "u_tail": u_tail,
                "status": st,
                "status_tex": st.replace("_", "\\_"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = metrics_dir / "anchor_staging_sensitivity_epsperf0p02_epstail1p0_20260302.csv"
    out_df.to_csv(out_csv, index=False)
    _write_tex(tables_dir / "table_anchor_staging_sensitivity.tex", out_df.to_dict(orient="records"))
    print(f"[anchor-staging] wrote {out_csv}")
    print(f"[anchor-staging] wrote {tables_dir / 'table_anchor_staging_sensitivity.tex'}")


if __name__ == "__main__":
    main()
