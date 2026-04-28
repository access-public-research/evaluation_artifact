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
    ap.add_argument("--config", default="configs/camelyon_base_scivalid_10seeds.yaml")
    ap.add_argument("--dataset", default="camelyon17")
    ap.add_argument("--suite_suffix", default="finetune_cam_scivalid10s_20260326")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--regimes",
        default="rcgdro,rcgdro_softclip_p95_a10_cam,rcgdro_softclip_p99_a10_cam",
        help="Comma-separated base regimes; finetune suffix is added automatically for evaluation.",
    )
    ap.add_argument("--tag_filter", default="scivalid10s")
    ap.add_argument("--fixed_epoch", type=int, default=10)
    ap.add_argument("--skip_train", type=int, default=0, help="Set to 1 to reuse existing finetune runs.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    py = args.python
    config = args.config
    dataset = args.dataset
    suite_suffix = args.suite_suffix
    tag_filter = args.tag_filter
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    ft_regimes = [f"{r}_finetune" for r in regimes]
    regimes_csv = ",".join(ft_regimes)

    if not args.skip_train:
        for regime in regimes:
            _run(
                [py, "-m", "src.scripts.finetune", "--config", config, "--dataset", dataset, "--regime", regime],
                cwd=repo_root,
            )

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
    _run(
        [
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
            suite_suffix,
        ],
        cwd=repo_root,
    )
    _run(
        [
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
            "--fixed_epoch",
            str(args.fixed_epoch),
            "--out_suffix",
            f"{suite_suffix}_fixed{args.fixed_epoch}",
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.make_effect_size_table",
            "--summary_csv",
            str(repo_root / "figures" / f"camelyon17_properness_summary_{suite_suffix}.csv"),
            "--out_csv",
            str(repo_root / "artifacts" / "metrics" / f"camelyon17_effect_size_{suite_suffix}_selected.csv"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.make_effect_size_table",
            "--summary_csv",
            str(repo_root / "figures" / f"camelyon17_properness_summary_{suite_suffix}_fixed10.csv"),
            "--out_csv",
            str(repo_root / "artifacts" / "metrics" / f"camelyon17_effect_size_{suite_suffix}_fixed{args.fixed_epoch}.csv"),
        ],
        cwd=repo_root,
    )
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
            str(repo_root / "figures" / f"camelyon17_properness_summary_{suite_suffix}.csv"),
            "--tag_filter",
            tag_filter,
            "--out_csv",
            str(repo_root / "artifacts" / "metrics" / f"camelyon17_domain_acc_with_loss_{suite_suffix}_selected.csv"),
            "--out_summary",
            str(repo_root / "artifacts" / "metrics" / f"camelyon17_domain_acc_with_loss_{suite_suffix}_selected_summary.csv"),
        ],
        cwd=repo_root,
    )
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
            str(repo_root / "figures" / f"camelyon17_properness_summary_{suite_suffix}_fixed{args.fixed_epoch}.csv"),
            "--tag_filter",
            tag_filter,
            "--out_csv",
            str(repo_root / "artifacts" / "metrics" / f"camelyon17_domain_acc_with_loss_{suite_suffix}_fixed{args.fixed_epoch}.csv"),
            "--out_summary",
            str(repo_root / "artifacts" / "metrics" / f"camelyon17_domain_acc_with_loss_{suite_suffix}_fixed{args.fixed_epoch}_summary.csv"),
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
    manifest_path = repo_root / "artifacts" / "metrics" / f"camelyon_finetune_suite_manifest_{suite_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
