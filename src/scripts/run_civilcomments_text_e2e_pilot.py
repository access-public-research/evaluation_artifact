import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..config import load_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_dir(cfg: dict, dataset_name: str, regime_name: str, seed: int) -> Path:
    runs_root = Path(cfg["project"]["runs_dir"])
    tag = str(cfg["training"]["tag_suffix"])
    return runs_root / f"{dataset_name}_e2e" / regime_name / f"seed{int(seed)}" / tag


def _heartbeat_ok(heartbeat_path: Path, now: float, stale_seconds: int) -> tuple[bool, dict]:
    if not heartbeat_path.exists():
        return False, {}
    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except Exception:
        return False, {}
    wall = float(payload.get("wall_time", 0.0) or 0.0)
    if wall <= 0.0:
        return False, payload
    return (now - wall) <= float(stale_seconds), payload


def _launch_training(
    repo_root: Path,
    py: str,
    config: str,
    dataset: str,
    regime: str,
    seed: int,
    *,
    safe_mode: bool,
    overwrite: bool,
    log_path: Path,
    epochs_override: int,
) -> subprocess.Popen:
    cmd = [
        py,
        "-m",
        "src.scripts.finetune_civilcomments_text",
        "--config",
        config,
        "--dataset",
        dataset,
        "--regime",
        regime,
        "--seed",
        str(int(seed)),
        "--overwrite",
        "1" if overwrite else "0",
    ]
    if int(epochs_override) > 0:
        cmd.extend(["--epochs_override", str(int(epochs_override))])
    if safe_mode:
        cmd.extend(
            [
                "--batch_size_override",
                "8",
                "--eval_batch_size_override",
                "16",
                "--amp_override",
                "0",
            ]
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write(f"\n[launch] {time.strftime('%Y-%m-%d %H:%M:%S')} safe_mode={int(safe_mode)} overwrite={int(overwrite)}\n")
    log_f.flush()
    return subprocess.Popen(cmd, cwd=str(repo_root), stdout=log_f, stderr=subprocess.STDOUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_v29_civilcomments_e2e_3seeds.yaml")
    ap.add_argument("--dataset", default="civilcomments")
    ap.add_argument("--baseline_regime", default="erm")
    ap.add_argument("--distorted_regime", default="erm_softclip_p95_a10_cc")
    ap.add_argument("--guardrail_rho", type=float, default=1.25)
    ap.add_argument("--fixed_epoch", type=int, default=10)
    ap.add_argument("--health_check_minutes", type=int, default=30)
    ap.add_argument("--poll_seconds", type=int, default=60)
    ap.add_argument("--stale_minutes", type=int, default=15)
    ap.add_argument("--epochs_override", type=int, default=-1)
    ap.add_argument("--seeds_csv", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--suite_suffix", default="civilcomments_e2e_softclip_3s_20260329")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    py = str(args.python)
    cfg = load_config(args.config, dataset_path=f"configs/datasets/{args.dataset}.yaml")
    dataset_name = cfg["dataset"]["name"]
    if str(args.seeds_csv).strip():
        seeds = [int(s.strip()) for s in str(args.seeds_csv).split(",") if s.strip()]
    else:
        seeds = [int(s) for s in cfg["training"]["seeds"]]
    regimes = [str(args.baseline_regime), str(args.distorted_regime)]

    logs_dir = repo_root / "artifacts" / "logs" / "civilcomments_e2e"
    metrics_dir = repo_root / "artifacts" / "metrics"
    manifest_path = metrics_dir / f"civilcomments_e2e_manifest_{args.suite_suffix}.json"
    rows_csv = metrics_dir / f"civilcomments_e2e_selector_rows_{args.suite_suffix}.csv"
    summary_csv = metrics_dir / f"civilcomments_e2e_selector_summary_{args.suite_suffix}.csv"

    manifest = {
        "config": args.config,
        "dataset": args.dataset,
        "suite_suffix": args.suite_suffix,
        "started_at": time.time(),
        "seed_order": seeds,
        "regimes": regimes,
        "runs": [],
        "evaluation": {},
    }
    _write_json(manifest_path, manifest)

    for seed in seeds:
        for regime in regimes:
            run_dir = _run_dir(cfg, dataset_name, regime, int(seed))
            done_path = run_dir / "done.json"
            heartbeat_path = run_dir / "heartbeat.json"
            if done_path.exists():
                manifest["runs"].append(
                    {
                        "seed": int(seed),
                        "regime": regime,
                        "status": "skipped_existing",
                        "run_dir": str(run_dir),
                        "completed_at": time.time(),
                    }
                )
                _write_json(manifest_path, manifest)
                continue

            success = False
            for attempt_idx, safe_mode in enumerate([False, True], start=1):
                launch_time = time.time()
                log_path = logs_dir / f"{args.suite_suffix}_seed{int(seed)}_{regime}_attempt{attempt_idx}.log"
                record = {
                    "seed": int(seed),
                    "regime": regime,
                    "attempt": int(attempt_idx),
                    "safe_mode": int(safe_mode),
                    "status": "launching",
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "launched_at": float(launch_time),
                }
                manifest["runs"].append(record)
                record_idx = len(manifest["runs"]) - 1
                _write_json(manifest_path, manifest)
                try:
                    proc = _launch_training(
                        repo_root,
                        py,
                        args.config,
                        args.dataset,
                        regime,
                        int(seed),
                        safe_mode=safe_mode,
                        overwrite=(attempt_idx > 1),
                        log_path=log_path,
                        epochs_override=int(args.epochs_override),
                    )
                    manifest["runs"][record_idx]["status"] = "running"
                    manifest["runs"][record_idx]["pid"] = int(proc.pid)
                    _write_json(manifest_path, manifest)
                    checked_30m = False
                    timed_out_health = False
                    while proc.poll() is None:
                        time.sleep(int(args.poll_seconds))
                        now = time.time()
                        hb_ok, hb_payload = _heartbeat_ok(heartbeat_path, now, stale_seconds=int(args.stale_minutes) * 60)
                        if not checked_30m and (now - launch_time) >= int(args.health_check_minutes) * 60:
                            checked_30m = True
                            if not hb_ok:
                                timed_out_health = True
                                proc.terminate()
                                break
                        if checked_30m and heartbeat_path.exists() and not hb_ok:
                            timed_out_health = True
                            proc.terminate()
                            break

                    ret = proc.wait()
                    _hb_ok, hb_payload = _heartbeat_ok(heartbeat_path, time.time(), stale_seconds=int(args.stale_minutes) * 60)
                    manifest["runs"][record_idx].update(
                        {
                            "status": "completed" if (ret == 0 and done_path.exists()) else "failed",
                            "returncode": int(ret),
                            "health_timeout": int(timed_out_health),
                            "elapsed_sec": float(time.time() - launch_time),
                            "heartbeat": hb_payload,
                            "completed_at": float(time.time()),
                        }
                    )
                    _write_json(manifest_path, manifest)

                    if ret == 0 and done_path.exists():
                        success = True
                        break
                except Exception:
                    manifest["runs"][record_idx].update(
                        {
                            "status": "runner_exception",
                            "elapsed_sec": float(time.time() - launch_time),
                            "traceback": traceback.format_exc()[-8000:],
                            "completed_at": float(time.time()),
                        }
                    )
                    _write_json(manifest_path, manifest)
                    break

            if not success:
                manifest["runs"].append(
                    {
                        "seed": int(seed),
                        "regime": regime,
                        "status": "failed_after_retries",
                        "run_dir": str(run_dir),
                        "completed_at": time.time(),
                    }
                )
                _write_json(manifest_path, manifest)

    eval_cmd = [
        py,
        "-m",
        "src.scripts.evaluate_civilcomments_text_e2e",
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--baseline_regime",
        args.baseline_regime,
        "--distorted_regime",
        args.distorted_regime,
        "--guardrail_rho",
        str(float(args.guardrail_rho)),
        "--fixed_epoch",
        str(int(args.fixed_epoch)),
        "--out_rows",
        str(rows_csv),
        "--out_summary",
        str(summary_csv),
    ]
    eval_t0 = time.time()
    eval_proc = subprocess.run(eval_cmd, cwd=str(repo_root), capture_output=True, text=True)
    manifest["evaluation"] = {
        "returncode": int(eval_proc.returncode),
        "elapsed_sec": float(time.time() - eval_t0),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "stdout_tail": eval_proc.stdout[-4000:],
        "stderr_tail": eval_proc.stderr[-4000:],
    }
    _write_json(manifest_path, manifest)
    if eval_proc.returncode != 0:
        raise SystemExit(eval_proc.returncode)


if __name__ == "__main__":
    main()
