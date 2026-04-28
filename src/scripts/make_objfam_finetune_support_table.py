from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "paper" / "neurips2026_selection_risk" / "tables"

DOMAIN_SUMMARY = ART / "camelyon17_domain_acc_with_loss_finetune_cam_objfam_scivalid10s_20260327d_selected_summary.csv"
TAIL_SUMMARY = ART / "camelyon17_effect_size_finetune_cam_objfam_scivalid10s_20260327d.csv"
CAL_SUMMARY = ART / "camelyon_finetune_objfam_selected_calibration_20260328_summary.csv"
OUT_CSV = ART / "camelyon_finetune_objfam_table24_support_20260427.csv"
OUT_TEX = TABLES / "table_objfam_finetune_support.tex"

BASELINE = "erm_finetune"
ROWS = [
    ("SoftClip P95", "erm_softclip_p95_a10_cam_finetune"),
    ("Label smoothing 0.10", "erm_labelsmooth_e10_cam_finetune"),
    ("Focal 2", "erm_focal_g2_cam_finetune"),
]


def _read_by_regime(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["regime"]: row for row in csv.DictReader(f)}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def build_rows() -> list[dict[str, object]]:
    domain = _read_by_regime(DOMAIN_SUMMARY)
    tail = _read_by_regime(TAIL_SUMMARY)
    cal = _read_by_regime(CAL_SUMMARY)

    base_domain = domain[BASELINE]
    base_tail = tail[BASELINE]
    base_cal = cal[BASELINE]

    out: list[dict[str, object]] = []
    for label, regime in ROWS:
        row = {
            "family": label,
            "regime": regime,
            "baseline_regime": BASELINE,
            "n": int(float(domain[regime]["n"])),
            "delta_heldout_loss": _float(domain[regime], "test_loss_mean") - _float(base_domain, "test_loss_mean"),
            "delta_heldout_acc": _float(domain[regime], "test_acc_mean") - _float(base_domain, "test_acc_mean"),
            "delta_tail_cvar": _float(tail[regime], "tail_worst_cvar_mean") - _float(base_tail, "tail_worst_cvar_mean"),
            "delta_ece": _float(cal[regime], "val_ece_mean") - _float(base_cal, "val_ece_mean"),
        }
        out.append(row)
    return out


def write_csv(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "regime",
        "baseline_regime",
        "n",
        "delta_heldout_loss",
        "delta_heldout_acc",
        "delta_tail_cvar",
        "delta_ece",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, object]]) -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Family & $\Delta$Held-out loss$\downarrow$ & $\Delta$Held-out acc$\uparrow$ & $\Delta$Tail CVaR$\downarrow$ & $\Delta$ECE$\downarrow$ \\",
        r"  \midrule",
    ]
    for row in rows:
        lines.append(
            "  {family} & {loss} & {acc} & {tail} & {ece} \\\\".format(
                family=row["family"],
                loss=_fmt(float(row["delta_heldout_loss"])),
                acc=_fmt(float(row["delta_heldout_acc"])),
                tail=_fmt(float(row["delta_tail_cvar"])),
                ece=_fmt(float(row["delta_ece"])),
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_tex(rows)
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
