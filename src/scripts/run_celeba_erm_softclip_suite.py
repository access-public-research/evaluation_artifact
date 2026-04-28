import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Suite:
    dataset: str
    config: str
    regimes: List[str]
    proxy_family: str
    tail_family: str


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
        "selected_effect": metrics_dir / f"{dataset}_effect_size_{suite_suffix}.csv",
        "fixed_effect": metrics_dir / f"{dataset}_effect_size_{suite_suffix}_fixed30.csv",
        "test_rows": metrics_dir / f"{dataset}_test_wg_selected_with_loss_rows_{suite_suffix}.csv",
        "test_summary": metrics_dir / f"{dataset}_test_wg_selected_with_loss_summary_{suite_suffix}.csv",
        "test_rows_fixed": metrics_dir / f"{dataset}_test_wg_fixed30_with_loss_rows_{suite_suffix}.csv",
        "test_summary_fixed": metrics_dir / f"{dataset}_test_wg_fixed30_with_loss_summary_{suite_suffix}.csv",
        "manifest": metrics_dir / f"{dataset}_erm_softclip_manifest_{suite_suffix}.json",
    }


def _train_regimes(repo_root: Path, py: str, suite: Suite) -> None:
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


def _eval_and_summarize(repo_root: Path, py: str, suite: Suite, suite_suffix: str) -> None:
    cfg_path = repo_root / suite.config
    tag_suffix = _load_tag_suffix(cfg_path)
    backbone = _load_backbone(cfg_path)
    out = _out_paths(repo_root, suite.dataset, backbone, suite_suffix)
    regimes_csv = ",".join(suite.regimes)

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
            "src.scripts.make_effect_size_table",
            "--summary_csv",
            str(out["selected_summary"]),
            "--out_csv",
            str(out["selected_effect"]),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.make_effect_size_table",
            "--summary_csv",
            str(out["fixed_summary"]),
            "--out_csv",
            str(out["fixed_effect"]),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.evaluate_selected_group_test",
            "--config",
            suite.config,
            "--dataset",
            suite.dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--proxy_family",
            suite.proxy_family,
            "--selection_metric_mode",
            "auto",
            "--tag_filter",
            tag_suffix,
            "--out_rows",
            str(out["test_rows"]),
            "--out_summary",
            str(out["test_summary"]),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.evaluate_selected_group_test",
            "--config",
            suite.config,
            "--dataset",
            suite.dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--proxy_family",
            suite.proxy_family,
            "--selection_metric_mode",
            "auto",
            "--tag_filter",
            tag_suffix,
            "--fixed_epoch",
            "30",
            "--out_rows",
            str(out["test_rows_fixed"]),
            "--out_summary",
            str(out["test_summary_fixed"]),
        ],
        cwd=repo_root,
    )

    manifest = {
        "dataset": suite.dataset,
        "config": suite.config,
        "tag_suffix": tag_suffix,
        "regimes": suite.regimes,
        "outputs": {k: str(v) for k, v in out.items()},
    }
    out["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out['manifest']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--suffix", default="erm_softclip_celeba_10s_20260325")
    ap.add_argument("--skip_train", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    py = str(args.python_exe)
    if not Path(py).exists():
        raise FileNotFoundError(f"python_exe not found: {py}")

    suite = Suite(
        dataset="celeba",
        config="configs/base_v25_erm_softclip_celeba_10seeds.yaml",
        regimes=[
            "erm",
            "erm_softclip_p95_a10",
            "erm_softclip_p97_a10",
            "erm_softclip_p99_a10",
        ],
        proxy_family="conf_teacher_wpl",
        tail_family="teacher_difficulty",
    )

    if not int(args.skip_train):
        _train_regimes(repo_root=repo_root, py=py, suite=suite)
    _eval_and_summarize(repo_root=repo_root, py=py, suite=suite, suite_suffix=str(args.suffix))


if __name__ == "__main__":
    main()
