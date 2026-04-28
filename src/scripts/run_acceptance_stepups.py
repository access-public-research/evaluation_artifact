import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path) -> dict:
    t0 = time.time()
    print(f"[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": round(time.time() - t0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date_tag", default="20260328")
    args = ap.parse_args()

    root = _repo_root()
    py = sys.executable
    art = root / "artifacts" / "metrics"
    tables = root / "paper" / "neurips2026_selection_risk" / "tables"
    art.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    steps = [
        {
            "name": "camelyon_finetune_core_calibration",
            "cmd": [
                py,
                "-m",
                "src.scripts.evaluate_selected_camelyon_calibration",
                "--config",
                "configs/camelyon_base_scivalid_10seeds.yaml",
                "--dataset",
                "camelyon17",
                "--regimes",
                "rcgdro_finetune,rcgdro_softclip_p95_a10_cam_finetune",
                "--metrics_suffix",
                "finetune_cam_scivalid10s_20260326",
                "--proxy_family",
                "conf_teacher_wpl",
                "--selection_metric_mode",
                "auto",
                "--tag_filter",
                "scivalid10s",
                "--eval_split",
                "val_skew",
                "--out_rows",
                str(art / f"camelyon_finetune_selected_calibration_p95_{args.date_tag}_rows.csv"),
                "--out_summary",
                str(art / f"camelyon_finetune_selected_calibration_p95_{args.date_tag}_summary.csv"),
            ],
        },
        {
            "name": "camelyon_finetune_objfam_calibration",
            "cmd": [
                py,
                "-m",
                "src.scripts.evaluate_selected_camelyon_calibration",
                "--config",
                "configs/camelyon_base_scivalid_objfam_10seeds.yaml",
                "--dataset",
                "camelyon17",
                "--regimes",
                "erm_finetune,erm_softclip_p95_a10_cam_finetune,erm_labelsmooth_e10_cam_finetune,erm_focal_g2_cam_finetune",
                "--metrics_suffix",
                "finetune_cam_objfam_scivalid10s_20260327d",
                "--proxy_family",
                "conf_teacher_wpl",
                "--selection_metric_mode",
                "auto",
                "--tag_filter",
                "scivalid_objfam10s",
                "--eval_split",
                "val_skew",
                "--out_rows",
                str(art / f"camelyon_finetune_objfam_selected_calibration_{args.date_tag}_rows.csv"),
                "--out_summary",
                str(art / f"camelyon_finetune_objfam_selected_calibration_{args.date_tag}_summary.csv"),
            ],
        },
        {
            "name": "camelyon_erm_seedmatched_lure",
            "cmd": [
                py,
                "-m",
                "src.scripts.analyze_seedmatched_proxy_lure",
                "--config",
                "configs/base_v11_erm_softclip_camelyon_10seeds_nw0_fix.yaml",
                "--dataset",
                "camelyon17",
                "--baseline_regime",
                "erm",
                "--softclip_regime",
                "erm_softclip_p95_a10_cam",
                "--metrics_suffix",
                "v11erm_softclip_cam_10s_fix_20260228",
                "--proxy_family",
                "conf_teacher_wpl",
                "--selection_metric_mode",
                "auto",
                "--tag_filter",
                "v11ermsoftclipfix_cam_10s",
                "--out_rows",
                str(art / f"camelyon_erm_seedmatched_proxy_check_{args.date_tag}.csv"),
                "--out_summary",
                str(art / f"camelyon_erm_seedmatched_proxy_check_{args.date_tag}_summary.csv"),
            ],
        },
        {
            "name": "camelyon_finetune_seedmatched_lure",
            "cmd": [
                py,
                "-m",
                "src.scripts.analyze_seedmatched_proxy_lure",
                "--config",
                "configs/camelyon_base_scivalid_10seeds.yaml",
                "--dataset",
                "camelyon17",
                "--baseline_regime",
                "rcgdro_finetune",
                "--softclip_regime",
                "rcgdro_softclip_p95_a10_cam_finetune",
                "--metrics_suffix",
                "finetune_cam_scivalid10s_20260326",
                "--proxy_family",
                "conf_teacher_wpl",
                "--selection_metric_mode",
                "auto",
                "--tag_filter",
                "scivalid10s",
                "--out_rows",
                str(art / f"camelyon_finetune_seedmatched_proxy_check_{args.date_tag}.csv"),
                "--out_summary",
                str(art / f"camelyon_finetune_seedmatched_proxy_check_{args.date_tag}_summary.csv"),
            ],
        },
        {
            "name": "acceptance_stepup_summaries",
            "cmd": [
                py,
                "-m",
                "src.scripts.make_acceptance_stepup_tables",
                "--date_tag",
                str(args.date_tag),
                "--out_dir",
                str(art),
                "--paper_tables_dir",
                str(tables),
            ],
        },
    ]

    results = []
    status = "completed"
    step = None
    try:
        for step in steps:
            print(f"[step] {step['name']}", flush=True)
            rec = _run(step["cmd"], cwd=root)
            rec["name"] = step["name"]
            results.append(rec)
    except subprocess.CalledProcessError as ex:
        status = "failed"
        results.append(
            {
                "name": step["name"] if step else "unknown",
                "cmd": step["cmd"] if step else [],
                "returncode": int(ex.returncode),
                "elapsed_sec": None,
            }
        )
        raise
    finally:
        manifest = {
            "status": status,
            "date_tag": str(args.date_tag),
            "steps": results,
            "outputs": {
                "core_calibration_summary": str(art / f"camelyon_finetune_selected_calibration_p95_{args.date_tag}_summary.csv"),
                "objfam_calibration_summary": str(art / f"camelyon_finetune_objfam_selected_calibration_{args.date_tag}_summary.csv"),
                "camelyon_erm_lure_summary": str(art / f"camelyon_erm_seedmatched_proxy_check_{args.date_tag}_summary.csv"),
                "camelyon_finetune_lure_summary": str(art / f"camelyon_finetune_seedmatched_proxy_check_{args.date_tag}_summary.csv"),
                "paired_effects": str(art / f"acceptance_stepup_paired_effects_{args.date_tag}.csv"),
                "dominance": str(art / f"acceptance_stepup_dominance_{args.date_tag}.csv"),
            },
        }
        manifest_path = art / f"acceptance_stepups_manifest_{args.date_tag}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[ok] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
