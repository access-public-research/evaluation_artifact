import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loo_agg_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_aggregate_sign_camloo_foldcal_a10_10s_20260304.csv",
    )
    ap.add_argument(
        "--alignment_controls_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_alignment_geometry_20260305_controls.csv",
    )
    ap.add_argument(
        "--density_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_density_topology_20260305_summary.csv",
    )
    ap.add_argument(
        "--activation_pooled_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_activation_inflection_20260305_pooled_summary.csv",
    )
    ap.add_argument(
        "--weighting_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_objective_weighting_signflip_20260305_summary.csv",
    )
    ap.add_argument(
        "--grad_summary_csv",
        default="replication_rcg/artifacts/metrics/camelyon17_tail_core_grad_ratio_20260305_summary.csv",
    )
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/staged_geometry_claim_matrix_20260305")
    return ap.parse_args()


def _status(flag_support: bool, flag_conditional: bool) -> str:
    if flag_support:
        return "supported"
    if flag_conditional:
        return "conditional"
    return "not_supported"


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    loo = pd.read_csv(args.loo_agg_csv)
    ali = pd.read_csv(args.alignment_controls_csv)
    den = pd.read_csv(args.density_summary_csv)
    act = pd.read_csv(args.activation_pooled_csv)
    wgt = pd.read_csv(args.weighting_summary_csv)
    grd = pd.read_csv(args.grad_summary_csv)

    claims: List[dict] = []

    # Claim 1: LOO staged hardness direction.
    r = loo[loo["metric"] == "top10_recover_late_gt_early"].iloc[0]
    p = float(r["p_one_sided"])
    succ = int(r["successes"])
    trials = int(r["trials"])
    rate = float(r["success_rate"])
    claims.append(
        {
            "claim_id": "C1_loo_staging_direction",
            "metric": "success_rate_late_gt_early",
            "value": f"{succ}/{trials} ({rate:.3f}), p_one_sided={p:.4f}",
            "status": _status(p < 0.05, rate >= 0.55),
        }
    )

    # Claim 2: Implicit weighting sign flip.
    sw = wgt[wgt["regime"] == "rcgdro_softclip_p95_a10_cam"]["tail_over_core_ratio"].iloc[0]
    fw = wgt[wgt["regime"] == "erm_focal_g2_cam"]["tail_over_core_ratio"].iloc[0]
    ew = wgt[wgt["regime"] == "erm"]["tail_over_core_ratio"].iloc[0] if (wgt["regime"] == "erm").any() else np.nan
    claims.append(
        {
            "claim_id": "C2_weighting_sign_flip",
            "metric": "tail_over_core_ratio_softclip_vs_focal",
            "value": f"softclip_p95={sw:.3f}, focal_g2={fw:.3f}, erm={ew:.3f}",
            "status": _status((sw < 1.0) and (fw > 1.0), (sw < 1.0) or (fw > 1.0)),
        }
    )

    # Claim 3: Parameter-gradient redistribution direction.
    sg = grd[grd["regime"] == "rcgdro_softclip_p95_a10_cam"]["tail_over_core_grad_ratio"].iloc[0]
    bg = grd[grd["regime"] == "rcgdro"]["tail_over_core_grad_ratio"].iloc[0]
    fg = grd[grd["regime"] == "erm_focal_g2_cam"]["tail_over_core_grad_ratio"].iloc[0]
    eg = grd[grd["regime"] == "erm"]["tail_over_core_grad_ratio"].iloc[0]
    claims.append(
        {
            "claim_id": "C3_grad_ratio_shift",
            "metric": "tail_over_core_grad_ratio",
            "value": f"rcgdro={bg:.3f}, softclip_p95={sg:.3f}, erm={eg:.3f}, focal_g2={fg:.3f}",
            "status": _status((sg < bg) and (fg >= eg), (sg < bg) or (fg >= eg)),
        }
    )

    # Claim 4: Activation inflection by transition state.
    def get_act(state: str, reg: str) -> float:
        s = act[(act["state"] == state) & (act["regime"] == reg)]
        return float(s["active_frac_mean"].iloc[0]) if not s.empty else np.nan

    rec97_p95 = get_act("rec_p97", "p95")
    rec97_p97 = get_act("rec_p97", "p97")
    rec99_p97 = get_act("rec_p99", "p97")
    rec99_p99 = get_act("rec_p99", "p99")
    cond = (rec97_p97 > rec97_p95) and (rec99_p99 > rec99_p97)
    weak = (rec97_p97 > rec97_p95) or (rec99_p99 > rec99_p97)
    claims.append(
        {
            "claim_id": "C4_activation_inflection",
            "metric": "active_frac_transition_order",
            "value": f"rec_p97: p95={rec97_p95:.3f}->p97={rec97_p97:.3f}; rec_p99: p97={rec99_p97:.3f}->p99={rec99_p99:.3f}",
            "status": _status(cond, weak),
        }
    )

    # Claim 5: Density difference rec_p99 vs rec_p95 (k=50 local_hard_density).
    d = den[(den["state"].isin(["rec_p95", "rec_p99"])) & (den["k"] == 50) & (den["metric"] == "local_hard_density")]
    pivot = d.pivot_table(index="fold", columns="state", values="mean").reset_index()
    if {"rec_p95", "rec_p99"}.issubset(set(pivot.columns)):
        diff = (pivot["rec_p99"] - pivot["rec_p95"]).to_numpy(dtype=np.float64)
        pos_rate = float(np.mean(diff > 0))
        claims.append(
            {
                "claim_id": "C5_density_gap",
                "metric": "rec_p99_minus_rec_p95_local_hard_density",
                "value": f"mean_diff={float(np.mean(diff)):.6f}, fold_pos_rate={pos_rate:.3f}",
                "status": _status(pos_rate >= 0.8, pos_rate >= 0.6),
            }
        )
    else:
        claims.append(
            {
                "claim_id": "C5_density_gap",
                "metric": "rec_p99_minus_rec_p95_local_hard_density",
                "value": "insufficient_data",
                "status": "not_supported",
            }
        )

    # Claim 6: Alignment contrast (rec_p99 - rec_p95).
    a = ali[ali["contrast"] == "rec_p99_minus_rec_p95"]
    if not a.empty:
        diff = a["mean_diff"].to_numpy(dtype=np.float64)
        pvals = a["p_perm_two_sided"].to_numpy(dtype=np.float64)
        sig_rate = float(np.mean(pvals < 0.05))
        abs_rate = float(np.mean(np.abs(diff) > 0.02))
        claims.append(
            {
                "claim_id": "C6_alignment_contrast",
                "metric": "rec_p99_minus_rec_p95_cos_easy_core",
                "value": f"mean_diff={float(np.mean(diff)):.6f}, sig_fold_rate={sig_rate:.3f}, |diff|>0.02_rate={abs_rate:.3f}",
                "status": _status((sig_rate >= 0.6) and (abs_rate >= 0.6), (sig_rate >= 0.6) or (abs_rate >= 0.6)),
            }
        )
    else:
        claims.append(
            {
                "claim_id": "C6_alignment_contrast",
                "metric": "rec_p99_minus_rec_p95_cos_easy_core",
                "value": "insufficient_data",
                "status": "not_supported",
            }
        )

    claims_df = pd.DataFrame(claims)
    out_csv = out_prefix.with_suffix(".csv")
    claims_df.to_csv(out_csv, index=False)

    md_lines = [
        "# Staged Geometry Claim Matrix",
        "",
        "| Claim | Metric | Value | Status |",
        "|---|---|---|---|",
    ]
    for _, r in claims_df.iterrows():
        md_lines.append(f"| {r['claim_id']} | {r['metric']} | {r['value']} | {r['status']} |")
    out_md = out_prefix.with_suffix(".md")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("[claim-matrix] wrote:")
    print(f" - {out_csv}")
    print(f" - {out_md}")


if __name__ == "__main__":
    main()
