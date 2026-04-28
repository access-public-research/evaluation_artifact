import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0_csv", required=True)
    ap.add_argument("--regimes", required=True, help="Comma-separated regimes to include.")
    ap.add_argument("--fixed_epoch", type=int, required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    phase0 = pd.read_csv(args.phase0_csv)
    regimes = {r.strip() for r in args.regimes.split(",") if r.strip()}

    keep = (
        phase0["regime"].isin(regimes)
        & (pd.to_numeric(phase0["epoch"], errors="coerce") == int(args.fixed_epoch))
    )
    rows = (
        phase0.loc[keep, ["regime", "seed", "tag", "epoch"]]
        .drop_duplicates()
        .sort_values(["regime", "seed", "tag"])
        .reset_index(drop=True)
    )
    if rows.empty:
        raise FileNotFoundError("No matching fixed-epoch rows found.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
