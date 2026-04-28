import argparse
from pathlib import Path

import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_effect_csv", required=True)
    ap.add_argument("--test_wg_summary_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    eff = _read(args.selected_effect_csv).copy()
    test = _read(args.test_wg_summary_csv).copy()

    eff = eff.rename(
        columns={
            "proxy_worst_loss_mean": "proxy_worst_loss",
            "tail_worst_cvar_mean": "tail_worst_cvar",
            "oracle_wg_acc_mean": "val_oracle_wg_acc",
            "val_overall_acc_mean": "val_overall_acc",
        }
    )
    keep_eff = [
        c
        for c in [
            "regime",
            "n",
            "proxy_worst_loss",
            "tail_worst_cvar",
            "val_oracle_wg_acc",
            "val_overall_acc",
            "frac_clipped_val_mean",
            "label_smoothing",
            "focal_gamma",
        ]
        if c in eff.columns
    ]
    eff = eff[keep_eff]

    test = test.rename(
        columns={
            "test_oracle_wg_acc_mean": "test_oracle_wg_acc",
            "test_overall_acc_mean": "test_overall_acc",
        }
    )
    test = test[[c for c in ["regime", "n", "test_oracle_wg_acc", "test_overall_acc"] if c in test.columns]]

    merged = eff.merge(test, on="regime", how="left", suffixes=("", "_test"))
    if "erm" not in set(merged["regime"]):
        raise ValueError("ERM baseline row missing from objective-family summary.")
    base = merged[merged["regime"] == "erm"].iloc[0]

    merged["delta_proxy_vs_erm"] = merged["proxy_worst_loss"] - float(base["proxy_worst_loss"])
    merged["delta_tail_vs_erm"] = merged["tail_worst_cvar"] - float(base["tail_worst_cvar"])
    if "test_oracle_wg_acc" in merged.columns:
        merged["delta_test_wg_vs_erm"] = merged["test_oracle_wg_acc"] - float(base["test_oracle_wg_acc"])
    if "test_overall_acc" in merged.columns:
        merged["delta_test_overall_vs_erm"] = merged["test_overall_acc"] - float(base["test_overall_acc"])
    if "val_oracle_wg_acc" in merged.columns:
        merged["delta_val_wg_vs_erm"] = merged["val_oracle_wg_acc"] - float(base["val_oracle_wg_acc"])
    if "val_overall_acc" in merged.columns:
        merged["delta_val_overall_vs_erm"] = merged["val_overall_acc"] - float(base["val_overall_acc"])

    merged = merged.sort_values("regime").reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
