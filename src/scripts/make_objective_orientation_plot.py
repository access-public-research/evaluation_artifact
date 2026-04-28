import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weighting_summary",
        default="replication_rcg/artifacts/metrics/camelyon17_objective_weighting_signflip_20260305_geompack_summary.csv",
    )
    ap.add_argument(
        "--softclip_effect",
        default="replication_rcg/artifacts/metrics/camelyon17_effect_size_cam_softclip_a10_p99_20260207.csv",
    )
    ap.add_argument(
        "--labelsmooth_effect",
        default="replication_rcg/artifacts/metrics/camelyon17_labelsmooth_effect_size_n10_20260304.csv",
    )
    ap.add_argument(
        "--focal_effect",
        default="replication_rcg/artifacts/metrics/camelyon17_focal_effect_size_n10_20260304.csv",
    )
    ap.add_argument(
        "--out_png",
        default="replication_rcg/artifacts/metrics/objective_orientation_tail_sign_20260305.png",
    )
    ap.add_argument(
        "--out_csv",
        default="replication_rcg/artifacts/metrics/objective_orientation_tail_sign_20260305.csv",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    w = _read(args.weighting_summary)[["regime", "tail_over_core_ratio"]].copy()
    w = w.rename(columns={"tail_over_core_ratio": "R_w"})

    sc = _read(args.softclip_effect)[["regime", "tail_worst_cvar_mean"]].copy()
    if "rcgdro" in set(sc["regime"]):
        sc_base_regime = "rcgdro"
        sc_regimes = ["rcgdro_softclip_p95_a10_cam", "rcgdro_softclip_p97_a10_cam", "rcgdro_softclip_p99_a10_cam"]
    elif "erm" in set(sc["regime"]):
        sc_base_regime = "erm"
        sc_regimes = ["erm_softclip_p95_a10_cam", "erm_softclip_p97_a10_cam", "erm_softclip_p99_a10_cam"]
    else:
        raise ValueError("softclip_effect must include either rcgdro or erm baseline")
    sc_base = float(sc.loc[sc["regime"] == sc_base_regime, "tail_worst_cvar_mean"].iloc[0])
    sc = sc[sc["regime"].isin(sc_regimes)].copy()
    sc["delta_tail"] = sc["tail_worst_cvar_mean"] - sc_base
    sc["family"] = "SoftClip"

    ls = _read(args.labelsmooth_effect)[["regime", "delta_tail_vs_erm"]].copy()
    ls = ls[ls["regime"] != "erm"].rename(columns={"delta_tail_vs_erm": "delta_tail"})
    ls["family"] = "LabelSmooth"

    fc = _read(args.focal_effect)[["regime", "delta_tail_vs_erm"]].copy()
    fc = fc[fc["regime"] != "erm"].rename(columns={"delta_tail_vs_erm": "delta_tail"})
    fc["family"] = "Focal"

    pts = pd.concat([sc[["regime", "delta_tail", "family"]], ls[["regime", "delta_tail", "family"]], fc[["regime", "delta_tail", "family"]]], ignore_index=True)
    pts = pts.merge(w, on="regime", how="left")
    pts = pts.dropna(subset=["R_w", "delta_tail"]).copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pts.to_csv(out_csv, index=False)

    colors: Dict[str, str] = {"SoftClip": "tab:red", "LabelSmooth": "tab:green", "Focal": "tab:blue"}
    markers: Dict[str, str] = {"SoftClip": "o", "LabelSmooth": "s", "Focal": "^"}

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    for fam in ["SoftClip", "LabelSmooth", "Focal"]:
        sub = pts[pts["family"] == fam]
        ax.scatter(
            sub["R_w"],
            sub["delta_tail"],
            label=fam,
            s=58,
            alpha=0.9,
            c=colors[fam],
            marker=markers[fam],
            edgecolor="black",
            linewidth=0.3,
        )
    ax.axvline(1.0, color="gray", ls="--", lw=1.0)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_xlabel(r"$R_w$ (stabilized tail/core amplification ratio)")
    ax.set_ylabel(r"$\Delta$Tail CVaR")
    ax.set_title("Orientation Predicts Tail Direction")
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print("[objective-orientation] wrote:")
    print(f" - {out_png}")
    print(f" - {out_csv}")


if __name__ == "__main__":
    main()
