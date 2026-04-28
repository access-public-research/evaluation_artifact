import argparse
import subprocess
import sys
from pathlib import Path


SUITES = {
    "cam_dense_adaptive": {
        "config": "configs/camelyon_base_v20_10seeds.yaml",
        "dataset": "camelyon17",
        "regimes": ["rcgdro_softclip_p96_a10_cam", "rcgdro_softclip_p98_a10_cam"],
    },
    "cam_fixed_control": {
        "config": "configs/camelyon_base_v20_fixedconf_10seeds.yaml",
        "dataset": "camelyon17",
        "regimes": ["rcgdro", "rcgdro_softclip_p95_a10_cam", "rcgdro_softclip_p97_a10_cam", "rcgdro_softclip_p99_a10_cam"],
    },
    "celeba_fixed_control": {
        "config": "configs/base_v20_fixedconf_celeba_10seeds.yaml",
        "dataset": "celeba",
        "regimes": ["rcgdro", "rcgdro_softclip_p95_a10"],
    },
}


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(SUITES) + ["all"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    py = sys.executable
    suite_names = list(SUITES) if str(args.suite) == "all" else [str(args.suite)]
    for suite_name in suite_names:
        spec = SUITES[suite_name]
        print(f"[suite] {suite_name}", flush=True)
        for regime in spec["regimes"]:
            _run(
                [
                    py,
                    "-m",
                    "src.scripts.train",
                    "--config",
                    spec["config"],
                    "--dataset",
                    spec["dataset"],
                    "--regime",
                    regime,
                ],
                cwd=root,
            )


if __name__ == "__main__":
    main()
