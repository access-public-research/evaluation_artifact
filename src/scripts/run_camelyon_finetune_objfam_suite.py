import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/camelyon_base_scivalid_objfam_10seeds.yaml")
    ap.add_argument("--dataset", default="camelyon17")
    ap.add_argument("--suite_suffix", default="finetune_cam_objfam_scivalid10s_20260327")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    py = args.python
    config = args.config
    dataset = args.dataset
    suite_suffix = args.suite_suffix
    tag_filter = "scivalid_objfam10s"
    regimes = ["erm", "erm_softclip_p95_a10_cam", "erm_labelsmooth_e10_cam", "erm_focal_g2_cam"]
    ft_regimes = [f"{r}_finetune" for r in regimes]
    regimes_csv = ",".join(ft_regimes)

    if not args.skip_train:
        for regime in regimes:
            _run([py, "-m", "src.scripts.finetune", "--config", config, "--dataset", dataset, "--regime", regime], cwd=repo_root)

    _run(
        [
            py,
            "-m",
            "src.scripts.phase0_eval",
            "--config",
            config,
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--tag_filter",
            tag_filter,
            "--out_suffix",
            suite_suffix,
            "--overwrite",
            "1",
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            "-m",
            "src.scripts.phase1_eval",
            "--config",
            config,
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--tag_filter",
            tag_filter,
            "--out_suffix",
            suite_suffix,
            "--overwrite",
            "1",
        ],
        cwd=repo_root,
    )
    for fixed_epoch, suffix in [(None, suite_suffix), (10, f"{suite_suffix}_fixed10")]:
        cmd = [
            py,
            "-m",
            "src.scripts.make_properness_plots",
            "--config",
            config,
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--tag_filter",
            tag_filter,
            "--proxy_family",
            "conf_teacher_wpl",
            "--tail_family",
            "teacher_difficulty",
            "--out_suffix",
            suffix,
        ]
        if fixed_epoch is not None:
            cmd.extend(["--fixed_epoch", str(fixed_epoch)])
        _run(cmd, cwd=repo_root)
    for suffix in [suite_suffix, f"{suite_suffix}_fixed10"]:
        _run(
            [
                py,
                "-m",
                "src.scripts.make_effect_size_table",
                "--summary_csv",
                str(repo_root / "figures" / f"camelyon17_properness_summary_{suffix}.csv"),
                "--out_csv",
                str(repo_root / "artifacts" / "metrics" / f"camelyon17_effect_size_{suffix}.csv"),
            ],
            cwd=repo_root,
        )
        summary_kind = "selected" if suffix == suite_suffix else "fixed10"
        _run(
            [
                py,
                "-m",
                "src.scripts.camelyon_domain_eval",
                "--config",
                config,
                "--dataset",
                dataset,
                "--summary_csv",
                str(repo_root / "figures" / f"camelyon17_properness_summary_{suffix}.csv"),
                "--tag_filter",
                tag_filter,
                "--out_csv",
                str(repo_root / "artifacts" / "metrics" / f"camelyon17_domain_acc_with_loss_{suite_suffix}_{summary_kind}.csv"),
                "--out_summary",
                str(repo_root / "artifacts" / "metrics" / f"camelyon17_domain_acc_with_loss_{suite_suffix}_{summary_kind}_summary.csv"),
            ],
            cwd=repo_root,
        )

    manifest = {
        "config": config,
        "dataset": dataset,
        "suite_suffix": suite_suffix,
        "tag_filter": tag_filter,
        "regimes": regimes,
        "finetune_regimes": ft_regimes,
    }
    manifest_path = repo_root / "artifacts" / "metrics" / f"camelyon_finetune_objfam_manifest_{suite_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
