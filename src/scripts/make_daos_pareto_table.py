import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _pick_perf_col(df: pd.DataFrame) -> str:
    for cand in ["test_hosp_2_acc_mean", "oracle_wg_acc_mean", "val_overall_acc_mean"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not determine performance column.")


def _fmt(v: float, digits: int = 3) -> str:
    return f"{float(v):.{digits}f}"


def _fmt_signed(v: float, digits: int = 2) -> str:
    return f"{float(v):+.{digits}f}"


def _write_table(rows: List[Dict[str, str]], out_tex: Path) -> None:
    lines = [
        "\\begin{tabular}{llcccccc}",
        "  \\toprule",
        "  Dataset & Regime & Proxy$\\downarrow$ & Tail$\\downarrow$ & Perf$\\uparrow$ & $\\Delta$Tail vs base & $\\Delta$Perf vs base & passes\\_all \\\\",
        "  \\midrule",
    ]
    last_dataset = None
    for row in rows:
        if last_dataset is not None and row["dataset"] != last_dataset:
            lines.append("  \\midrule")
        lines.append(
            "  {dataset} & {regime} & {proxy} & {tail} & {perf} & {dtail} & {dperf} & {passes} \\\\".format(**row)
        )
        last_dataset = row["dataset"]
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _regime_label(regime: str) -> str:
    mapping = {
        "rcgdro": "Baseline",
        "rcgdro_softclip_p95_a10_cam": "Static P95",
        "rcgdro_softclip_p99_a10_cam": "Static P99",
        "rcgdro_softclip_daos_p95_a10_cam": "DAOS-v1",
        "rcgdro_softclip_daos2_p95_a10_cam": "DAOS-v2",
        "rcgdro_softclip_p95_a10": "Static P95",
        "rcgdro_softclip_p99_a10": "Static P99",
        "rcgdro_softclip_daos_p95_a10": "DAOS-v1",
        "rcgdro_softclip_daos2_p95_a10": "DAOS-v2",
    }
    return mapping.get(str(regime), str(regime))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_csv", default="replication_rcg/artifacts/metrics/camelyon17_daos_proxy_tol5_summary_daos2_20260305.csv")
    ap.add_argument("--celeba_csv", default="replication_rcg/artifacts/metrics/celeba_daos_proxy_tol5_summary_daos2_20260305.csv")
    ap.add_argument("--out_tex", default="paper/neurips2026_selection_risk/tables/table_daos_pareto_proxy_tol5.tex")
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    for dataset_name, csv_path, dataset_label in [
        ("camelyon17", args.cam_csv, "Camelyon"),
        ("celeba", args.celeba_csv, "CelebA"),
    ]:
        df = _load(csv_path)
        perf_col = _pick_perf_col(df)
        wanted = [
            "rcgdro",
            "rcgdro_softclip_p95_a10_cam" if dataset_name == "camelyon17" else "rcgdro_softclip_p95_a10",
            "rcgdro_softclip_daos_p95_a10_cam" if dataset_name == "camelyon17" else "rcgdro_softclip_daos_p95_a10",
            "rcgdro_softclip_daos2_p95_a10_cam" if dataset_name == "camelyon17" else "rcgdro_softclip_daos2_p95_a10",
            "rcgdro_softclip_p99_a10_cam" if dataset_name == "camelyon17" else "rcgdro_softclip_p99_a10",
        ]
        df = df[df["regime"].isin(wanted)].copy()
        order = {reg: i for i, reg in enumerate(wanted)}
        df["order"] = df["regime"].map(order)
        df = df.sort_values("order")
        base = df[df["regime"] == "rcgdro"].iloc[0]
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": dataset_label,
                    "regime": _regime_label(str(r["regime"])),
                    "proxy": _fmt(r["proxy_selected_mean"], 3),
                    "tail": _fmt(r["tail_worst_cvar_mean"], 2),
                    "perf": _fmt(r[perf_col], 3),
                    "dtail": _fmt_signed(r["tail_worst_cvar_mean"] - base["tail_worst_cvar_mean"], 2),
                    "dperf": _fmt_signed(r[perf_col] - base[perf_col], 3),
                    "passes": "n/a" if str(r["regime"]) == "rcgdro" else ("yes" if bool(r.get("passes_all", False)) else "no"),
                }
            )

    _write_table(rows, Path(args.out_tex))
    print(f"[ok] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
