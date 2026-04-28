from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "neurips2026_selection_risk"


def _make_lure_region(ax: plt.Axes) -> None:
    alpha = np.linspace(0.02, 0.98, 400)
    ac = np.linspace(0.0, 1.2, 400)
    aa, cc = np.meshgrid(alpha, ac)
    at = 1.0

    proxy_improves = cc > aa * at
    risk_worsens = cc < at
    lure = proxy_improves & risk_worsens

    img = np.zeros((*lure.shape, 4), dtype=float)
    img[..., 0] = 0.85
    img[..., 1] = 0.2
    img[..., 2] = 0.2
    img[..., 3] = lure.astype(float) * 0.32
    ax.imshow(
        img,
        origin="lower",
        extent=[alpha.min(), alpha.max(), ac.min(), ac.max()],
        aspect="auto",
    )

    ax.plot(alpha, alpha * at, color="#1f4e79", lw=2.0)
    ax.axhline(at, color="#444444", lw=1.8, ls="--")
    ax.fill_between(alpha, alpha * at, at, color="#dc2626", alpha=0.12)

    ax.set_xlabel(r"Suppression slope $\alpha$")
    ax.set_ylabel(r"Core improvement mass $A_c$")
    ax.set_title("Local lure region", fontsize=11)
    ax.text(
        0.07,
        0.84,
        "Proxy improves;\nstandard risk worsens",
        transform=ax.transAxes,
        fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9),
    )
    ax.text(
        0.58,
        0.18,
        "Stronger suppression\nwidens the region",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
    )
    ax.text(0.16, 0.08, r"$A_c=\alpha A_t$", color="#1f4e79", transform=ax.transAxes, fontsize=8.6)
    ax.text(0.05, 0.94, r"$A_c=A_t$", color="#444444", transform=ax.transAxes, fontsize=8.6)
    ax.grid(alpha=0.2)


def _piecewise_paths() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0, 31, dtype=float)
    proxy = np.zeros_like(t)
    risk_transient = np.zeros_like(t)
    risk_persistent = np.zeros_like(t)
    phase_break = 12

    # Phase A: suppressive lure regime.
    proxy[: phase_break + 1] = -0.06 * t[: phase_break + 1]
    risk_transient[: phase_break + 1] = 0.045 * t[: phase_break + 1]
    risk_persistent[: phase_break + 1] = 0.045 * t[: phase_break + 1]

    # Phase B: recovery vs entrenchment.
    post_t = t[phase_break:] - phase_break
    proxy[phase_break:] = proxy[phase_break] - 0.01 * post_t
    risk_transient[phase_break:] = risk_transient[phase_break] - 0.07 * post_t
    risk_persistent[phase_break:] = risk_persistent[phase_break] + 0.01 * post_t
    return t, proxy, risk_transient, risk_persistent, np.full_like(t, phase_break)


def _make_phase_paths(ax: plt.Axes) -> None:
    t, proxy, risk_transient, risk_persistent, phase_break = _piecewise_paths()
    ax.plot(t, proxy, color="#1f4e79", lw=2.2)
    ax.plot(t, risk_transient, color="#c06c2b", lw=2.2)
    ax.plot(t, risk_persistent, color="#b91c1c", lw=2.2)
    ax.axvline(float(phase_break[0]), color="#6b7280", ls="--", lw=1.5)
    ax.text(
        0.05,
        0.10,
        "Phase A: suppressive mismatch",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
    )
    ax.text(
        0.64,
        0.92,
        "Phase B: recovery or entrenchment",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
    )
    ax.text(29.2, float(proxy[-1]) + 0.02, "Proxy", fontsize=8.5, color="#1f4e79", ha="right")
    ax.text(29.2, float(risk_transient[-1]) - 0.02, "Transient risk", fontsize=8.5, color="#c06c2b", ha="right")
    ax.text(29.2, float(risk_persistent[-1]) + 0.02, "Persistent risk", fontsize=8.5, color="#b91c1c", ha="right")
    ax.set_xlabel("Epoch / checkpoint index")
    ax.set_ylabel("Relative metric value")
    ax.set_title("Two-phase toy: transient vs persistent", fontsize=11)
    ax.grid(alpha=0.2)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=180)
    _make_lure_region(axes[0])
    _make_phase_paths(axes[1])
    fig.tight_layout()

    out_pdf = PAPER / "figures" / "fig_theory_bridge.pdf"
    out_png = PAPER / "figures" / "fig_theory_bridge.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
