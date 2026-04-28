import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--suffix", default="camloo_foldcal_a10_10s_20260304")
    ap.add_argument("--metrics_root", default="replication_rcg/artifacts/metrics")
    ap.add_argument("--embeds_root", default="replication_rcg/artifacts/embeds")
    ap.add_argument("--partitions_eval_root", default="replication_rcg/artifacts/partitions_eval")
    ap.add_argument("--out_prefix", default="replication_rcg/artifacts/metrics/geompack_preflight_20260305")
    ap.add_argument("--strict", type=int, default=1)
    return ap.parse_args()


def _exists(p: Path) -> bool:
    return p.exists()


def main() -> None:
    args = parse_args()
    folds = [int(x) for x in str(args.folds).split(",") if x.strip()]
    metrics_root = Path(args.metrics_root)
    embeds_root = Path(args.embeds_root)
    partitions_root = Path(args.partitions_eval_root)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    checks: List[dict] = []
    missing: List[dict] = []

    def check(path: Path, kind: str, fold: str, detail: str) -> None:
        ok = _exists(path)
        row = {"fold": fold, "kind": kind, "detail": detail, "path": str(path), "exists": int(ok)}
        checks.append(row)
        if not ok:
            missing.append(row)

    for h in folds:
        fold = f"camelyon17_loo_h{h}"
        fold_res = f"{fold}_resnet50"
        selected_csv = metrics_root / f"{fold}_pathway1_selected_rows_{args.suffix}.csv"
        check(selected_csv, "selected_rows", fold, "selected rows csv")

        embed_dir = embeds_root / fold_res
        for fname in ["X_test.npy", "y_test.npy", "meta_test.npy", "X_validation.npy", "y_validation.npy", "meta_validation.npy"]:
            check(embed_dir / fname, "embed", fold, fname)

        teacher_meta = partitions_root / fold_res / "teacher_difficulty" / "meta.json"
        check(teacher_meta, "partition", fold, "teacher difficulty meta")

        if selected_csv.exists():
            df = pd.read_csv(selected_csv)
            expected_regimes = {
                "rcgdro",
                f"rcgdro_softclip_p95_a10_cam_loo_h{h}cal",
                f"rcgdro_softclip_p97_a10_cam_loo_h{h}cal",
                f"rcgdro_softclip_p99_a10_cam_loo_h{h}cal",
            }
            seeds = sorted(df["seed"].unique().tolist())
            for regime in sorted(expected_regimes):
                sub = df[df["regime"] == regime]
                for s in seeds:
                    ss = sub[sub["seed"] == s]
                    detail = f"{regime} seed{s}"
                    if ss.empty:
                        checks.append({"fold": fold, "kind": "run", "detail": detail, "path": "", "exists": 0})
                        missing.append({"fold": fold, "kind": "run", "detail": detail, "path": "", "exists": 0})
                        continue
                    run_dir = Path(ss.iloc[0]["run_dir"])
                    ep = int(ss.iloc[0]["epoch_selected"])
                    check(run_dir, "run_dir", fold, detail)
                    check(run_dir / f"ckpt_epoch{ep:03d}.pt", "checkpoint", fold, detail)
                    check(run_dir / "config.json", "config", fold, detail)
                    check(run_dir / "metrics.jsonl", "metrics_jsonl", fold, detail)

    checks_df = pd.DataFrame(checks)
    missing_df = pd.DataFrame(missing)
    out_checks = out_prefix.with_suffix(".csv")
    out_missing = out_prefix.with_name(out_prefix.name + "_missing.csv")
    checks_df.to_csv(out_checks, index=False)
    missing_df.to_csv(out_missing, index=False)

    print("[geom-preflight] wrote:")
    print(f" - {out_checks}")
    print(f" - {out_missing}")
    print(f"[geom-preflight] missing count = {len(missing_df)}")
    if int(args.strict) and len(missing_df) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
