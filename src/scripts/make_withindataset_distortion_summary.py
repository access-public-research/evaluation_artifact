import argparse
from pathlib import Path

import pandas as pd


PRETTY_SCOPE = {
    "celeba": "CelebA",
    "waterbirds": "Waterbirds",
    "camelyon17": "Camelyon17",
    "all": "Pooled",
}


def _format_num(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}"


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[df["predictor"].isin(["distortion_mass_selected", "frac_clipped_selected"])].copy()
    keep = keep[keep["scope"].isin(["per_dataset", "pooled"])].copy()

    wide = (
        keep.pivot_table(
            index=["scope", "dataset", "n"],
            columns="predictor",
            values="pearson",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    wide["dataset_pretty"] = wide["dataset"].map(PRETTY_SCOPE).fillna(wide["dataset"])
    wide["r_distmass"] = pd.to_numeric(wide["distortion_mass_selected"], errors="coerce")
    wide["r_fracclip"] = pd.to_numeric(wide["frac_clipped_selected"], errors="coerce")
    wide["delta_r"] = wide["r_distmass"] - wide["r_fracclip"]

    sort_key = {"CelebA": 0, "Waterbirds": 1, "Camelyon17": 2, "Pooled": 3}
    wide["sort_key"] = wide["dataset_pretty"].map(sort_key).fillna(99)
    wide = wide.sort_values(["sort_key", "dataset_pretty"]).reset_index(drop=True)

    return wide[
        [
            "dataset_pretty",
            "n",
            "r_distmass",
            "r_fracclip",
            "delta_r",
        ]
    ].copy()


def _write_latex(summary: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "  \\toprule",
        "  Scope & $r$(DistMass, Tail$\\Delta$) & $r$(FracClip, Tail$\\Delta$) & $\\Delta r$ & $n$ \\\\",
        "  \\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "  {scope} & {rd} & {rf} & {dr} & {n} \\\\".format(
                scope=row["dataset_pretty"],
                rd=_format_num(float(row["r_distmass"])),
                rf=_format_num(float(row["r_fracclip"])),
                dr=_format_num(float(row["delta_r"])),
                n=int(row["n"]),
            )
        )
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corr_csv",
        default="artifacts/metrics/tail_predictor_correlations_head_20260227.csv",
    )
    ap.add_argument(
        "--out_csv",
        default="artifacts/metrics/distortion_withindataset_summary_20260305.csv",
    )
    ap.add_argument(
        "--out_tex",
        default="paper/neurips2026_selection_risk/tables/table_distortion_withindataset_mini.tex",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.corr_csv)
    summary = _build_summary(df)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    _write_latex(summary, Path(args.out_tex))

    print(f"[ok] wrote {args.out_csv}")
    print(f"[ok] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
