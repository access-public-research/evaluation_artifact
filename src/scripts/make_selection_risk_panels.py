import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


@dataclass(frozen=True)
class PanelSpec:
    title: str
    subtitle: str
    regime: str
    selected_effect_csv: Path
    fixed_effect_csv: Path
    selected_loss_csv: Path
    fixed_loss_csv: Path
    loss_col: str


P95_COLOR = "#c43c35"
HAZARD_FILL = "#fde7e6"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_proxy_tail(effect_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(effect_csv)
    if "proxy_selected_mean" in df.columns and "tail_worst_cvar_mean" in df.columns:
        return df[["regime", "proxy_selected_mean", "tail_worst_cvar_mean"]].rename(
            columns={"proxy_selected_mean": "proxy_mean", "tail_worst_cvar_mean": "tail_mean"}
        )
    if "proxy_worst_loss_clip_mean" in df.columns or "proxy_worst_loss_mean" in df.columns:
        proxy_clip = pd.to_numeric(df.get("proxy_worst_loss_clip_mean"), errors="coerce")
        proxy_raw = pd.to_numeric(df.get("proxy_worst_loss_mean"), errors="coerce")
        out = df[["regime"]].copy()
        out["proxy_mean"] = proxy_clip.fillna(proxy_raw)
        out["tail_mean"] = pd.to_numeric(df["tail_worst_cvar_mean"], errors="coerce")
        return out
    if "proxy_worst_loss" in df.columns or "proxy_worst_loss_clip" in df.columns:
        proxy_clip = pd.to_numeric(df.get("proxy_worst_loss_clip"), errors="coerce")
        proxy_raw = pd.to_numeric(df.get("proxy_worst_loss"), errors="coerce")
        work = df[["regime"]].copy()
        work["proxy_mean"] = proxy_clip.fillna(proxy_raw)
        work["tail_mean"] = pd.to_numeric(df["tail_worst_cvar"], errors="coerce")
        return work.groupby("regime", as_index=False)[["proxy_mean", "tail_mean"]].mean(numeric_only=True)
    raise ValueError(f"Unsupported proxy/tail schema in {effect_csv}")


def _load_loss(loss_csv: Path, loss_col: str) -> pd.DataFrame:
    df = pd.read_csv(loss_csv)
    return df[["regime", loss_col]].rename(columns={loss_col: "loss_mean"})


def _panel_point(spec: PanelSpec) -> dict[str, float | str]:
    sel_pt = _load_proxy_tail(spec.selected_effect_csv).set_index("regime")
    fix_pt = _load_proxy_tail(spec.fixed_effect_csv).set_index("regime")
    sel_loss = _load_loss(spec.selected_loss_csv, spec.loss_col).set_index("regime")
    fix_loss = _load_loss(spec.fixed_loss_csv, spec.loss_col).set_index("regime")

    baseline_regime = sel_pt.index[0]
    if baseline_regime not in fix_pt.index:
        baseline_regime = fix_pt.index[0]

    base_proxy_sel = float(sel_pt.loc[baseline_regime, "proxy_mean"])
    base_loss_sel = float(sel_loss.loc[baseline_regime, "loss_mean"])
    base_proxy_fix = float(fix_pt.loc[baseline_regime, "proxy_mean"])
    base_loss_fix = float(fix_loss.loc[baseline_regime, "loss_mean"])

    return {
        "title": spec.title,
        "subtitle": spec.subtitle,
        "sel_x": float(sel_pt.loc[spec.regime, "proxy_mean"] - base_proxy_sel),
        "sel_y": float(sel_loss.loc[spec.regime, "loss_mean"] - base_loss_sel),
        "fix_x": float(fix_pt.loc[spec.regime, "proxy_mean"] - base_proxy_fix),
        "fix_y": float(fix_loss.loc[spec.regime, "loss_mean"] - base_loss_fix),
    }


def _default_panels(root: Path) -> list[PanelSpec]:
    art = root / "artifacts" / "metrics"
    figs = root / "figures"
    return [
        PanelSpec(
            title="CelebA ERM",
            subtitle="Transient hazard",
            regime="erm_softclip_p95_a10",
            selected_effect_csv=art / "celeba_effect_size_erm_softclip_celeba_10s_20260325.csv",
            fixed_effect_csv=art / "celeba_effect_size_erm_softclip_celeba_10s_20260325_fixed30.csv",
            selected_loss_csv=art / "celeba_test_wg_selected_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
            fixed_loss_csv=art / "celeba_test_wg_fixed30_with_loss_summary_erm_softclip_celeba_10s_20260325.csv",
            loss_col="test_oracle_wg_loss_mean",
        ),
        PanelSpec(
            title="Camelyon17 ERM",
            subtitle="Persistent hazard",
            regime="erm_softclip_p95_a10_cam",
            selected_effect_csv=figs / "camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_20260228.csv",
            fixed_effect_csv=figs / "camelyon17_properness_summary_v11erm_softclip_cam_10s_fix_fixed30_20260228.csv",
            selected_loss_csv=art / "camelyon17_resnet50_domain_acc_with_loss_selected_summary_v11erm_softclip_cam_10s_fix_20260325.csv",
            fixed_loss_csv=art / "camelyon17_resnet50_domain_acc_with_loss_fixed30_summary_v11erm_softclip_cam_10s_fix_20260326.csv",
            loss_col="test_hosp_2_loss_mean",
        ),
        PanelSpec(
            title="Camelyon17 Finetune",
            subtitle="Persistent hazard, end-to-end",
            regime="rcgdro_softclip_p95_a10_cam_finetune",
            selected_effect_csv=art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_selected.csv",
            fixed_effect_csv=art / "camelyon17_effect_size_finetune_cam_scivalid10s_20260326_fixed10.csv",
            selected_loss_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_selected_summary.csv",
            fixed_loss_csv=art / "camelyon17_domain_acc_with_loss_finetune_cam_scivalid10s_20260326_fixed10_summary.csv",
            loss_col="test_hosp_2_loss_mean",
        ),
    ]


def _draw_panel(ax, point: dict[str, float | str], xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.75)
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.75)
    ax.fill_between([xlim[0], 0.0], [0.0, 0.0], [ylim[1], ylim[1]], color=HAZARD_FILL, alpha=0.75, zorder=0)
    ax.annotate(
        "",
        xy=(point["fix_x"], point["fix_y"]),
        xytext=(point["sel_x"], point["sel_y"]),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=P95_COLOR),
    )
    ax.scatter(point["sel_x"], point["sel_y"], s=120, color=P95_COLOR, zorder=3)
    ax.scatter(point["fix_x"], point["fix_y"], s=130, facecolors="white", edgecolors=P95_COLOR, linewidths=2.2, zorder=4)
    ax.text(point["sel_x"], point["sel_y"] + 0.04 * (ylim[1] - ylim[0]), "P95 sel.", color=P95_COLOR, fontsize=10, ha="center")
    ax.text(point["fix_x"], point["fix_y"] - 0.05 * (ylim[1] - ylim[0]), "P95 fix.", color=P95_COLOR, fontsize=10, ha="center")
    ax.text(
        xlim[0] + 0.03 * (xlim[1] - xlim[0]),
        ylim[1] - 0.08 * (ylim[1] - ylim[0]),
        "Hazard region",
        fontsize=10,
        color="#8f1f1f",
        ha="left",
        va="top",
    )
    ax.set_title(f"{point['title']}\n{point['subtitle']}", fontsize=12, pad=10)
    ax.set_xlabel(r"$\Delta$ Proxy (left is better)")
    ax.grid(alpha=0.22, lw=0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    points = [_panel_point(spec) for spec in _default_panels(_repo_root())]
    xs = [float(p["sel_x"]) for p in points] + [float(p["fix_x"]) for p in points] + [0.0]
    ys = [float(p["sel_y"]) for p in points] + [float(p["fix_y"]) for p in points] + [0.0]
    x_pad = max(0.05, 0.1 * (max(xs) - min(xs) or 1.0))
    y_pad = max(0.08, 0.12 * (max(ys) - min(ys) or 1.0))
    xlim = (min(xs) - x_pad, max(xs) + x_pad)
    ylim = (min(ys) - y_pad, max(ys) + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True, sharex=True, sharey=True)
    for ax, point in zip(axes, points):
        _draw_panel(ax, point, xlim=xlim, ylim=ylim)
    axes[0].set_ylabel(r"$\Delta$ Held-out loss (up is worse)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=P95_COLOR, markeredgecolor=P95_COLOR, markersize=9, label="Proxy-selected checkpoint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=P95_COLOR, markeredgewidth=2.0, markersize=9, label="Fixed-horizon checkpoint"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))

    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[selection-risk] wrote {out_pdf}")
    print(f"[selection-risk] wrote {out_png}")


if __name__ == "__main__":
    main()
