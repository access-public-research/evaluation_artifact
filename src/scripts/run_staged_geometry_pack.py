import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260305_geompack")
    ap.add_argument("--source_suffix", default="camloo_foldcal_a10_10s_20260304")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", type=int, default=1)
    return ap.parse_args()


def _run(cmd: List[str], cwd: Path, logf) -> None:
    logf.write("\n$ " + " ".join(cmd) + "\n")
    logf.flush()
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    logf.write(p.stdout + "\n")
    logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    rcg_root = script_path.parents[2]
    workspace_root = rcg_root.parent
    py = sys.executable

    metrics_dir = workspace_root / "replication_rcg" / "artifacts" / "metrics"
    logs_dir = metrics_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"geompack_{args.tag}.log"
    manifest_path = logs_dir / f"geompack_{args.tag}_manifest.json"

    # Output prefixes.
    preflight_prefix = f"replication_rcg/artifacts/metrics/geompack_preflight_{args.tag}"
    canonical_prefix = f"replication_rcg/artifacts/metrics/camelyon17_transition_canonical_{args.tag}"
    alignment_prefix = f"replication_rcg/artifacts/metrics/camelyon17_alignment_geometry_{args.tag}"
    density_prefix = f"replication_rcg/artifacts/metrics/camelyon17_density_topology_{args.tag}"
    activation_prefix = f"replication_rcg/artifacts/metrics/camelyon17_activation_inflection_{args.tag}"
    signflip_prefix = f"replication_rcg/artifacts/metrics/camelyon17_objective_weighting_signflip_{args.tag}"
    gradratio_prefix = f"replication_rcg/artifacts/metrics/camelyon17_tail_core_grad_ratio_{args.tag}"
    loo_cond_prefix = f"replication_rcg/artifacts/metrics/camelyon17_loo_conditionality_{args.tag}"
    loo_geom_prefix = f"replication_rcg/artifacts/metrics/camelyon17_loo_transition_geometry_{args.tag}"
    dashboard_prefix = f"replication_rcg/artifacts/metrics/posthoc_dashboard_{args.tag}"
    claim_prefix = f"replication_rcg/artifacts/metrics/staged_geometry_claim_matrix_{args.tag}"

    commands = [
        [
            py,
            "replication_rcg/src/scripts/preflight_staged_geometry.py",
            "--suffix",
            str(args.source_suffix),
            "--out_prefix",
            preflight_prefix,
            "--strict",
            "1",
        ],
        [
            py,
            "replication_rcg/src/scripts/build_camelyon_transition_canonical.py",
            "--suffix",
            str(args.source_suffix),
            "--split",
            "test",
            "--device",
            str(args.device),
            "--out_prefix",
            canonical_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/analyze_alignment_geometry.py",
            "--majority_rows_csv",
            canonical_prefix + "_majority_rows.csv",
            "--out_prefix",
            alignment_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/analyze_density_topology.py",
            "--majority_rows_csv",
            canonical_prefix + "_majority_rows.csv",
            "--device",
            str(args.device),
            "--out_prefix",
            density_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/analyze_activation_inflection.py",
            "--cache_glob",
            canonical_prefix + "_cache_camelyon17_loo_h*.npz",
            "--out_prefix",
            activation_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/analyze_objective_weighting_signflip.py",
            "--out_prefix",
            signflip_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/analyze_tail_core_gradient_ratio.py",
            "--device",
            str(args.device),
            "--out_prefix",
            gradratio_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/analyze_camelyon_loo_conditionality.py",
            "--fold_summary_csv",
            f"replication_rcg/artifacts/metrics/camelyon17_loo_pathway1_fold_summary_{args.source_suffix}.csv",
            "--selected_rows_pattern",
            f"replication_rcg/artifacts/metrics/camelyon17_loo_h{{h}}_pathway1_selected_rows_{args.source_suffix}.csv",
            "--device",
            str(args.device),
            "--out_prefix",
            loo_cond_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/plot_camelyon_loo_transition_geometry.py",
            "--suffix",
            str(args.source_suffix),
            "--split",
            "test",
            "--device",
            str(args.device),
            "--out_prefix",
            loo_geom_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/build_posthoc_analysis_dashboard.py",
            "--loo_conditionality_csv",
            loo_cond_prefix + "_fold_features.csv",
            "--weighting_summary_csv",
            signflip_prefix + "_summary.csv",
            "--grad_summary_csv",
            gradratio_prefix + "_summary.csv",
            "--out_prefix",
            dashboard_prefix,
        ],
        [
            py,
            "replication_rcg/src/scripts/synthesize_staged_geometry_claims.py",
            "--alignment_controls_csv",
            alignment_prefix + "_controls.csv",
            "--density_summary_csv",
            density_prefix + "_summary.csv",
            "--activation_pooled_csv",
            activation_prefix + "_pooled_summary.csv",
            "--weighting_summary_csv",
            signflip_prefix + "_summary.csv",
            "--grad_summary_csv",
            gradratio_prefix + "_summary.csv",
            "--out_prefix",
            claim_prefix,
        ],
    ]

    manifest = {
        "tag": args.tag,
        "source_suffix": args.source_suffix,
        "device": args.device,
        "python": py,
        "workspace_root": str(workspace_root),
        "started_unix": time.time(),
        "commands": [],
        "log_path": str(log_path),
    }

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"[geompack] tag={args.tag}\n")
        logf.write(f"[geompack] python={py}\n")
        logf.write(f"[geompack] workspace={workspace_root}\n")
        for cmd in commands:
            t0 = time.time()
            _run(cmd, cwd=workspace_root, logf=logf)
            manifest["commands"].append({"cmd": cmd, "seconds": time.time() - t0, "status": "ok"})
        manifest["status"] = "ok"
        manifest["finished_unix"] = time.time()

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[geompack] complete")
    print(f" - log: {log_path}")
    print(f" - manifest: {manifest_path}")
    print(f" - claim matrix: {claim_prefix}.csv")


if __name__ == "__main__":
    main()
