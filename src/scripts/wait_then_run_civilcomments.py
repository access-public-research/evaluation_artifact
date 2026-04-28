import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wait_manifest",
        default="artifacts/metrics/camelyon_finetune_objfam_manifest_finetune_cam_objfam_scivalid10s_20260327d.json",
    )
    ap.add_argument(
        "--civilcomments_config",
        default="configs/base_v27_erm_softclip_civilcomments_10seeds.yaml",
    )
    ap.add_argument("--civilcomments_dataset", default="civilcomments")
    ap.add_argument("--civilcomments_suite_suffix", default="civilcomments_erm_softclip_10s_20260327")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--poll_seconds", type=int, default=300)
    ap.add_argument("--timeout_hours", type=float, default=24.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / args.wait_manifest
    py = str(args.python)

    print(f"[wait] watching for {manifest_path}", flush=True)
    deadline = time.time() + float(args.timeout_hours) * 3600.0
    while not manifest_path.exists():
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for manifest: {manifest_path}")
        time.sleep(int(args.poll_seconds))

    print(f"[wait] detected manifest {manifest_path}", flush=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"[wait] camelyon suite complete: {manifest.get('suite_suffix', 'unknown')}", flush=True)
    except Exception:
        print("[wait] manifest present but could not be parsed; continuing", flush=True)

    _run(
        [
            py,
            "-m",
            "src.scripts.run_civilcomments_erm_softclip_suite",
            "--config",
            args.civilcomments_config,
            "--dataset",
            args.civilcomments_dataset,
            "--suite_suffix",
            args.civilcomments_suite_suffix,
        ],
        cwd=repo_root,
    )


if __name__ == "__main__":
    main()
