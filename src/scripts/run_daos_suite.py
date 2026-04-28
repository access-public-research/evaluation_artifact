import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass
class DatasetSuite:
    dataset: str
    config: str
    regimes: List[str]
    proxy_family: str
    tail_family: str
    run_domain_eval: bool


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_tag_suffix(cfg_path: Path) -> str:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    suffix = str(cfg.get("training", {}).get("tag_suffix", "")).strip()
    if not suffix:
        raise ValueError(f"training.tag_suffix must be set in {cfg_path}")
    return suffix


def _load_backbone(cfg_path: Path) -> str:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return str(cfg.get("embeddings", {}).get("backbone", "resnet50"))


def _out_paths(repo_root: Path, dataset: str, backbone: str, suite_suffix: str):
    metrics_dir = repo_root / "artifacts" / "metrics"
    figures_dir = repo_root / "figures"
    return {
        "phase0": metrics_dir / f"{dataset}_{backbone}_phase0_val_metrics_{suite_suffix}.csv",
        "phase1": metrics_dir / f"{dataset}_{backbone}_phase1_pockets_{suite_suffix}.csv",
        "selected_summary": figures_dir / f"{dataset}_properness_summary_{suite_suffix}.csv",
        "fixed_summary": figures_dir / f"{dataset}_properness_summary_{suite_suffix}_fixed30.csv",
        "rows": metrics_dir / f"{dataset}_{backbone}_selected_vs_fixed_rows_{suite_suffix}.csv",
        "summary": metrics_dir / f"{dataset}_{backbone}_selected_vs_fixed_summary_{suite_suffix}.csv",
        "dist_rows": metrics_dir / f"{dataset}_tail_distortion_rows_{suite_suffix}.csv",
        "dist_summary": metrics_dir / f"{dataset}_tail_distortion_summary_{suite_suffix}.csv",
        "effect_size": metrics_dir / f"{dataset}_effect_size_{suite_suffix}.csv",
        "domain_selected": metrics_dir / f"{dataset}_{backbone}_domain_acc_{suite_suffix}.csv",
        "domain_fixed": metrics_dir / f"{dataset}_{backbone}_domain_acc_{suite_suffix}_fixed30.csv",
        "daos_summary_csv": metrics_dir / f"{dataset}_daos_summary_{suite_suffix}.csv",
        "daos_summary_md": metrics_dir / f"{dataset}_daos_summary_{suite_suffix}.md",
    }


def _train_regimes(repo_root: Path, py: str, suite: DatasetSuite) -> None:
    for regime in suite.regimes:
        _run(
            [
                py,
                "-m",
                "src.scripts.train",
                "--config",
                suite.config,
                "--dataset",
                suite.dataset,
                "--regime",
                regime,
            ],
            cwd=repo_root,
        )


def _eval_and_summarize(
    repo_root: Path,
    py: str,
    suite: DatasetSuite,
    suite_suffix: str,
) -> None:
    cfg_path = repo_root / suite.config
    tag_suffix = _load_tag_suffix(cfg_path)
    backbone = _load_backbone(cfg_path)
    out = _out_paths(repo_root, suite.dataset, backbone, suite_suffix)
    regimes_csv = ",".join(suite.regimes)

    # Phase 0 / 1 evaluation.
    _run(
        [
            py,
            "-m",
            "src.scripts.phase0_eval",
            "--config",
            suite.config,
            "--dataset",
            suite.dataset,
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--tag_filter",
            tag_suffix,
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
            suite.config,
            "--dataset",
            suite.dataset,
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--tag_filter",
            tag_suffix,
            "--out_suffix",
            suite_suffix,
            "--overwrite",
            "1",
        ],
        cwd=repo_root,
    )

    # Selected-by-proxy summary.
    _run(
        [
            py,
            "-m",
            "src.scripts.make_properness_plots",
            "--config",
            suite.config,
            "--dataset",
            suite.dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--tag_filter",
            tag_suffix,
            "--proxy_family",
            suite.proxy_family,
            "--tail_family",
            suite.tail_family,
            "--out_suffix",
            suite_suffix,
        ],
        cwd=repo_root,
    )

    # Fixed epoch 30 summary.
    _run(
        [
            py,
            "-m",
            "src.scripts.make_properness_plots",
            "--config",
            suite.config,
            "--dataset",
            suite.dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--tag_filter",
            tag_suffix,
            "--proxy_family",
            suite.proxy_family,
            "--tail_family",
            suite.tail_family,
            "--fixed_epoch",
            "30",
            "--out_suffix",
            f"{suite_suffix}_fixed30",
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            "-m",
            "src.scripts.build_selected_vs_fixed_tables",
            "--selected_summary_csv",
            str(out["selected_summary"]),
            "--fixed_summary_csv",
            str(out["fixed_summary"]),
            "--phase0_csv",
            str(out["phase0"]),
            "--out_rows_csv",
            str(out["rows"]),
            "--out_summary_csv",
            str(out["summary"]),
            "--fixed_label",
            "fixed_epoch_30",
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            "-m",
            "src.scripts.compute_tail_distortion_diagnostics",
            "--config",
            suite.config,
            "--dataset",
            suite.dataset,
            "--selected_rows_csv",
            str(out["rows"]),
            "--pockets_csv",
            str(out["phase1"]),
            "--families",
            "teacher_difficulty,decoupled_proj,global_hash",
            "--banks",
            "A,B",
            "--regimes",
            regimes_csv,
            "--min_cell",
            "20",
            "--cvar_q",
            "0.1",
            "--split",
            "val_skew",
            "--out_suffix",
            suite_suffix,
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            "-m",
            "src.scripts.make_effect_size_table",
            "--summary_csv",
            str(out["selected_summary"]),
            "--out_csv",
            str(out["effect_size"]),
        ],
        cwd=repo_root,
    )

    if suite.run_domain_eval:
        _run(
            [
                py,
                "-m",
                "src.scripts.camelyon_domain_eval",
                "--config",
                suite.config,
                "--dataset",
                suite.dataset,
                "--summary_csv",
                str(out["selected_summary"]),
                "--tag_filter",
                tag_suffix,
                "--out_csv",
                str(out["domain_selected"]),
            ],
            cwd=repo_root,
        )
        _run(
            [
                py,
                "-m",
                "src.scripts.camelyon_domain_eval",
                "--config",
                suite.config,
                "--dataset",
                suite.dataset,
                "--summary_csv",
                str(out["fixed_summary"]),
                "--tag_filter",
                tag_suffix,
                "--out_csv",
                str(out["domain_fixed"]),
            ],
            cwd=repo_root,
        )

    summarize_cmd = [
        py,
        "-m",
        "src.scripts.summarize_daos_suite",
        "--dataset",
        suite.dataset,
        "--rows_csv",
        str(out["rows"]),
        "--summary_csv",
        str(out["summary"]),
        "--distortion_rows_csv",
        str(out["dist_rows"]),
        "--selection_mode",
        "selected_best_proxy",
        "--baseline_regime",
        "rcgdro",
        "--out_csv",
        str(out["daos_summary_csv"]),
        "--out_md",
        str(out["daos_summary_md"]),
    ]
    if suite.run_domain_eval:
        summarize_cmd.extend(["--domain_csv", str(out["domain_selected"]), "--domain_metric", "test_hosp_2_acc"])
    _run(summarize_cmd, cwd=repo_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--skip_train", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    py = str(args.python_exe)
    if not Path(py).exists():
        raise FileNotFoundError(f"python_exe not found: {py}")

    run_stamp = args.suffix.strip() or datetime.now().strftime("daos_%Y%m%d")

    suites = [
        DatasetSuite(
            dataset="camelyon17",
            config="configs/base_v13_daos_camelyon_10seeds.yaml",
            regimes=[
                "rcgdro",
                "rcgdro_softclip_p95_a10_cam",
                "rcgdro_softclip_p99_a10_cam",
                "rcgdro_softclip_daos_p95_a10_cam",
                "rcgdro_softclip_daos2_p95_a10_cam",
            ],
            proxy_family="conf_teacher_wpl",
            tail_family="teacher_difficulty",
            run_domain_eval=True,
        ),
        DatasetSuite(
            dataset="celeba",
            config="configs/base_v13_daos_celeba_5seeds.yaml",
            regimes=[
                "rcgdro",
                "rcgdro_softclip_p95_a10",
                "rcgdro_softclip_p99_a10",
                "rcgdro_softclip_daos_p95_a10",
                "rcgdro_softclip_daos2_p95_a10",
            ],
            proxy_family="conf_teacher_wpl",
            tail_family="teacher_difficulty",
            run_domain_eval=False,
        ),
    ]

    manifest = {
        "suite_suffix": run_stamp,
        "python_exe": py,
        "repo_root": str(repo_root),
        "datasets": [],
    }

    for suite in suites:
        ds_suffix = f"{suite.dataset}_{run_stamp}"
        print(f"\n=== DAOS suite: {suite.dataset} ({ds_suffix}) ===")
        if not int(args.skip_train):
            _train_regimes(repo_root=repo_root, py=py, suite=suite)
        _eval_and_summarize(repo_root=repo_root, py=py, suite=suite, suite_suffix=ds_suffix)

        cfg_path = repo_root / suite.config
        backbone = _load_backbone(cfg_path)
        out = _out_paths(repo_root, suite.dataset, backbone, ds_suffix)
        manifest["datasets"].append(
            {
                "dataset": suite.dataset,
                "config": suite.config,
                "regimes": suite.regimes,
                "suffix": ds_suffix,
                "outputs": {k: str(v) for k, v in out.items()},
            }
        )

    manifest_path = repo_root / "artifacts" / "metrics" / f"daos_suite_manifest_{run_stamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[daos-suite] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
