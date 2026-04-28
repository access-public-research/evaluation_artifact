import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.io import ensure_dir
from ..utils.stats import ci95_mean, cvar_top_fraction


def _make_split(n: int, eps_hard: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    z_hard = (rng.random(n) < eps_hard).astype(np.int64)
    sign = 2.0 * y - 1.0

    noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    x = np.empty((n, 1), dtype=np.float32)

    # Easy subgroup: high margin.
    m_easy = 2.0
    # Hard subgroup: low margin + adverse shift (rare catastrophic pocket).
    m_hard = 0.5
    adverse = np.where(y > 0.5, -1.0, 1.0).astype(np.float32) * 0.7

    x_easy = sign * m_easy + noise
    x_hard = sign * m_hard + adverse + noise
    x[:, 0] = np.where(z_hard == 1, x_hard, x_easy)

    return {"x": x, "y": y, "hard": z_hard}


def _cvar(vals: np.ndarray, q: float) -> float:
    return cvar_top_fraction(vals, q)


def _train_one(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    tau: float | None,
    alpha: float,
    epochs: int,
    lr: float,
    q: float,
    seed: int,
) -> Tuple[pd.DataFrame, int]:
    torch.manual_seed(seed)
    w = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)

    x_tr = torch.from_numpy(train["x"])
    y_tr = torch.from_numpy(train["y"]).view(-1, 1)
    x_val = torch.from_numpy(val["x"])
    y_val = torch.from_numpy(val["y"]).view(-1, 1)
    hard_val = val["hard"]

    opt = torch.optim.SGD([w, b], lr=lr)
    rows: List[Dict] = []
    for e in range(1, epochs + 1):
        logits = x_tr @ w + b
        losses = F.binary_cross_entropy_with_logits(logits, y_tr, reduction="none").view(-1)
        if tau is None or alpha >= 1.0:
            losses_obj = losses
        else:
            losses_obj = torch.where(losses <= tau, losses, tau + alpha * (losses - tau))
        loss = losses_obj.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            lv = x_val @ w + b
            lval = F.binary_cross_entropy_with_logits(lv, y_val, reduction="none").view(-1).cpu().numpy()
            pval = torch.sigmoid(lv).view(-1).cpu().numpy()
            pred = (pval >= 0.5).astype(np.int64)
            yv = val["y"].astype(np.int64)
            acc = float(np.mean(pred == yv))
            hard_acc = float(np.mean(pred[hard_val == 1] == yv[hard_val == 1]))
            frac_clip = 0.0 if tau is None else float(np.mean(lval > float(tau)))
            proxy = float(np.mean(lval)) if tau is None else float(np.mean(np.where(lval <= tau, lval, tau + alpha * (lval - tau))))
            tail_hard = _cvar(lval[hard_val == 1], q=q)
            rows.append(
                {
                    "epoch": e,
                    "proxy_metric": proxy,
                    "tail_hard_cvar": tail_hard,
                    "hard_acc": hard_acc,
                    "overall_acc": acc,
                    "frac_clipped_val": frac_clip,
                    "val_loss_mean": float(np.mean(lval)),
                }
            )

    df = pd.DataFrame(rows)
    best_epoch = int(df.loc[df["proxy_metric"].idxmin(), "epoch"])
    return df, best_epoch


def _mean_ci(v: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    if v.size == 1:
        return m, 0.0
    ci = ci95_mean(v)
    return m, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eps_hard", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--out_suffix", default="toy_softclip_20260227")
    args = ap.parse_args()

    rows: List[Dict] = []
    for seed in range(int(args.seeds)):
        train = _make_split(n=int(args.n_train), eps_hard=float(args.eps_hard), seed=1000 + seed)
        val = _make_split(n=int(args.n_val), eps_hard=float(args.eps_hard), seed=2000 + seed)

        # Baseline run for threshold calibration at epoch-1 train losses.
        torch.manual_seed(seed)
        w0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
        b0 = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
        opt0 = torch.optim.SGD([w0, b0], lr=float(args.lr))
        x_tr = torch.from_numpy(train["x"])
        y_tr = torch.from_numpy(train["y"]).view(-1, 1)
        for _ in range(1):
            l0 = F.binary_cross_entropy_with_logits(x_tr @ w0 + b0, y_tr, reduction="none").view(-1)
            opt0.zero_grad(set_to_none=True)
            l0.mean().backward()
            opt0.step()
        with torch.no_grad():
            l_ep1 = F.binary_cross_entropy_with_logits(x_tr @ w0 + b0, y_tr, reduction="none").view(-1).cpu().numpy()
        tau95 = float(np.quantile(l_ep1, 0.95))
        tau97 = float(np.quantile(l_ep1, 0.97))
        tau99 = float(np.quantile(l_ep1, 0.99))

        regimes = [
            ("rcgdro", None, 1.0),
            ("p95", tau95, float(args.alpha)),
            ("p97", tau97, float(args.alpha)),
            ("p99", tau99, float(args.alpha)),
        ]
        for regime, tau, alpha in regimes:
            df_ep, best_epoch = _train_one(
                train=train,
                val=val,
                tau=tau,
                alpha=alpha,
                epochs=int(args.epochs),
                lr=float(args.lr),
                q=float(args.q),
                seed=seed,
            )
            rbest = df_ep[df_ep["epoch"] == best_epoch].iloc[0]
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "tau": np.nan if tau is None else float(tau),
                    "alpha": float(alpha),
                    "selected_epoch": int(best_epoch),
                    "proxy_metric": float(rbest["proxy_metric"]),
                    "tail_hard_cvar": float(rbest["tail_hard_cvar"]),
                    "hard_acc": float(rbest["hard_acc"]),
                    "overall_acc": float(rbest["overall_acc"]),
                    "frac_clipped_val": float(rbest["frac_clipped_val"]),
                }
            )

    df_rows = pd.DataFrame(rows)
    base_tail = float(df_rows[df_rows["regime"] == "rcgdro"]["tail_hard_cvar"].mean())
    df_rows["tail_delta_vs_baseline"] = df_rows["tail_hard_cvar"] - base_tail

    srows: List[Dict] = []
    for regime, grp in df_rows.groupby("regime"):
        pm, pci = _mean_ci(grp["proxy_metric"].to_numpy())
        tm, tci = _mean_ci(grp["tail_hard_cvar"].to_numpy())
        hm, hci = _mean_ci(grp["hard_acc"].to_numpy())
        fm, fci = _mean_ci(grp["frac_clipped_val"].to_numpy())
        dm, dci = _mean_ci(grp["tail_delta_vs_baseline"].to_numpy())
        srows.append(
            {
                "regime": regime,
                "n": int(grp.shape[0]),
                "proxy_metric_mean": pm,
                "proxy_metric_ci95": pci,
                "tail_hard_cvar_mean": tm,
                "tail_hard_cvar_ci95": tci,
                "hard_acc_mean": hm,
                "hard_acc_ci95": hci,
                "frac_clipped_val_mean": fm,
                "frac_clipped_val_ci95": fci,
                "tail_delta_vs_baseline_mean": dm,
                "tail_delta_vs_baseline_ci95": dci,
            }
        )
    df_summary = pd.DataFrame(srows)

    artifacts_dir = Path("artifacts") / "metrics"
    figures_dir = Path("figures")
    ensure_dir(artifacts_dir)
    ensure_dir(figures_dir)
    suffix = str(args.out_suffix).strip()
    suffix = f"_{suffix}" if suffix else ""
    rows_path = artifacts_dir / f"synthetic_toy_rows{suffix}.csv"
    summary_path = artifacts_dir / f"synthetic_toy_summary{suffix}.csv"
    df_rows.to_csv(rows_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # Simple phase portrait.
    rank = {"rcgdro": 0, "p95": 1, "p97": 2, "p99": 3}
    dfp = df_summary.copy()
    dfp["ord"] = dfp["regime"].map(rank)
    dfp = dfp.sort_values("ord")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(dfp["frac_clipped_val_mean"], dfp["proxy_metric_mean"], marker="o")
    ax[0].set_xlabel("FracClip")
    ax[0].set_ylabel("Proxy (lower better)")
    ax[0].set_title("Proxy vs FracClip")

    ax[1].plot(dfp["frac_clipped_val_mean"], dfp["tail_hard_cvar_mean"], marker="o")
    ax[1].set_xlabel("FracClip")
    ax[1].set_ylabel("Tail CVaR (hard group)")
    ax[1].set_title("Tail vs FracClip")

    ax[2].plot(dfp["frac_clipped_val_mean"], dfp["hard_acc_mean"], marker="o")
    ax[2].set_xlabel("FracClip")
    ax[2].set_ylabel("Hard-group accuracy")
    ax[2].set_title("Perf vs FracClip")
    fig.tight_layout()
    fig_path = figures_dir / f"synthetic_toy_phase_portrait{suffix}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[toy] wrote rows: {rows_path}")
    print(f"[toy] wrote summary: {summary_path}")
    print(f"[toy] wrote figure: {fig_path}")


if __name__ == "__main__":
    main()
