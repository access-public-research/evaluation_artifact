import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_cfg(cfg_path: Path) -> dict:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_v27_erm_softclip_civilcomments_10seeds.yaml")
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--suite_suffix", default="civilcomments_erm_softclip_10s_20260327")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_embed", type=int, default=0)
    ap.add_argument("--skip_train", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    py = str(args.python)
    config = str(args.config)
    dataset = str(args.dataset)
    suite_suffix = str(args.suite_suffix)

    cfg = _load_cfg(repo_root / config)
    tag_filter = str(cfg.get("training", {}).get("tag_suffix", "")).strip()
    if not tag_filter:
        raise ValueError("training.tag_suffix must be set in config.")
    backbone = str(cfg.get("embeddings", {}).get("backbone", "distilbert-base-uncased"))

    regimes = ["erm", "erm_softclip_p95_a10"]
    regimes_csv = ",".join(regimes)

    if not int(args.skip_embed):
        _run([py, "-m", "src.scripts.embed_cache", "--config", config, "--dataset", dataset, "--overwrite", "1"], cwd=repo_root)
    _run([py, "-m", "src.scripts.build_partitions", "--config", config, "--dataset", dataset, "--overwrite", "1"], cwd=repo_root)
    _run([py, "-m", "src.scripts.build_eval_banks", "--config", config, "--dataset", dataset, "--overwrite", "1"], cwd=repo_root)

    if not int(args.skip_train):
        for regime in regimes:
            _run([py, "-m", "src.scripts.train", "--config", config, "--dataset", dataset, "--regime", regime], cwd=repo_root)

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
            "global_hash",
            "--tail_family",
            "decoupled_proj",
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
            "global_hash",
            "--tail_family",
            "decoupled_proj",
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
            str(repo_root / "figures" / f"civilcomments_properness_summary_{suite_suffix}.csv"),
            "--out_csv",
            str(repo_root / "artifacts" / "metrics" / f"civilcomments_effect_size_{suite_suffix}.csv"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.make_effect_size_table",
            "--summary_csv",
            str(repo_root / "figures" / f"civilcomments_properness_summary_{suite_suffix}_fixed30.csv"),
            "--out_csv",
            str(repo_root / "artifacts" / "metrics" / f"civilcomments_effect_size_{suite_suffix}_fixed30.csv"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.evaluate_selected_civilcomments_test",
            "--config",
            config,
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--proxy_family",
            "global_hash",
            "--selection_metric_mode",
            "auto",
            "--tag_filter",
            tag_filter,
            "--out_rows",
            str(repo_root / "artifacts" / "metrics" / f"civilcomments_test_wilds_selected_rows_{suite_suffix}.csv"),
            "--out_summary",
            str(repo_root / "artifacts" / "metrics" / f"civilcomments_test_wilds_selected_summary_{suite_suffix}.csv"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "-m",
            "src.scripts.evaluate_selected_civilcomments_test",
            "--config",
            config,
            "--dataset",
            dataset,
            "--regimes",
            regimes_csv,
            "--metrics_suffix",
            suite_suffix,
            "--proxy_family",
            "global_hash",
            "--selection_metric_mode",
            "auto",
            "--tag_filter",
            tag_filter,
            "--fixed_epoch",
            "30",
            "--out_rows",
            str(repo_root / "artifacts" / "metrics" / f"civilcomments_test_wilds_fixed30_rows_{suite_suffix}.csv"),
            "--out_summary",
            str(repo_root / "artifacts" / "metrics" / f"civilcomments_test_wilds_fixed30_summary_{suite_suffix}.csv"),
        ],
        cwd=repo_root,
    )

    manifest = {
        "config": config,
        "dataset": dataset,
        "suite_suffix": suite_suffix,
        "tag_filter": tag_filter,
        "backbone": backbone,
        "regimes": regimes,
    }
    manifest_path = repo_root / "artifacts" / "metrics" / f"civilcomments_erm_softclip_manifest_{suite_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
