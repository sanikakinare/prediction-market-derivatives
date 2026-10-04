"""Part A (path simulation) and Part B (numerical validation) of the baseline model.

Run from the project root:
    .venv/bin/python experiments/01_paths_and_validation.py

Figures are written to results/; a validation report is printed to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.information_model import brownian_bridge_variance  # noqa: E402
from src.simulation import simulate_market_paths  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---- baseline parameters -----------------------------------------------------
p0 = 0.60
T = 1.0
kappa = 1.5
n_steps = 300
seed = 42
n_plot_paths = 8
n_validation_paths = 20_000  # full paths: 20k x 301 floats ~ 48 MB per array

# ---- plot style --------------------------------------------------------------
COLOR_YES = "#2a78d6"   # Y = 1
COLOR_NO = "#eb6834"    # Y = 0
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
})


def plot_sample_paths(rng: np.random.Generator) -> None:
    """Part A: a handful of paths of xi_t and S_t, coloured by the realised outcome."""
    paths = simulate_market_paths(p0, T, kappa, n_steps, n_plot_paths, rng)
    t = paths.time

    fig, (ax_xi, ax_S) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i in range(n_plot_paths):
        color = COLOR_YES if paths.Y[i] == 1 else COLOR_NO
        ax_xi.plot(t, paths.information_signal[i], color=color, alpha=0.85)
        ax_S.plot(t, paths.prediction_price[i], color=color, alpha=0.85)

    ax_xi.plot(t, kappa * t, color=INK_MUTED, ls="--", lw=1.2, label=r"$\kappa t$ (signal if $Y=1$)")
    ax_xi.axhline(0, color=INK_MUTED, ls=":", lw=1.2, label=r"$0$ (signal if $Y=0$)")
    ax_xi.set(title=r"Information signal $\xi_t = \kappa t Y + \beta_{tT}$", xlabel="t", ylabel=r"$\xi_t$")

    ax_S.axhline(p0, color=INK_MUTED, ls="--", lw=1.2, label=rf"$p_0={p0}$")
    ax_S.set(title=r"Prediction-market price $S_t = Q(Y=1\mid\mathcal{F}_t)$",
             xlabel="t", ylabel=r"$S_t$", ylim=(-0.03, 1.03))

    handles = [plt.Line2D([], [], color=COLOR_YES, label="Y = 1 (resolves YES)"),
               plt.Line2D([], [], color=COLOR_NO, label="Y = 0 (resolves NO)")]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    ax_xi.legend(loc="upper left")
    ax_S.legend(loc="upper left")
    fig.suptitle(rf"Sample paths, $p_0={p0}$, $\kappa={kappa}$, $T={T}$", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RESULTS / "A_sample_paths.png")
    plt.close(fig)
    print(f"Sample paths: Y = {paths.Y.tolist()}, S_T = {paths.prediction_price[:, -1].tolist()}")


def validate(rng: np.random.Generator) -> None:
    """Part B: pathwise checks plus the martingale check E[S_t] = p0."""
    paths = simulate_market_paths(p0, T, kappa, n_steps, n_validation_paths, rng)
    t, S, beta, Y = paths.time, paths.prediction_price, paths.brownian_bridge, paths.Y
    N = n_validation_paths

    print(f"\n=== Pathwise checks ({N:,} paths x {n_steps + 1} times) ===")
    print(f"min S_t = {S.min():.3e},  max S_t = {S.max():.12f}   -> bounds ok: {bool((S >= 0).all() and (S <= 1).all())}")
    print(f"S_0 == p0 on every path:   {bool(np.all(S[:, 0] == p0))}  (max |S_0 - p0| = {np.abs(S[:, 0] - p0).max():.2e})")
    print(f"S_T == Y on every path:    {bool(np.all(S[:, -1] == Y))}")
    print(f"max |beta_0| = {np.abs(beta[:, 0]).max():.2e},  max |beta_T| = {np.abs(beta[:, -1]).max():.2e}")
    print(f"empirical Q(Y=1) = {Y.mean():.4f}  (p0 = {p0}, SE = {np.sqrt(p0 * (1 - p0) / N):.4f})")
    # How close to resolution is the price just before T?
    S_pre = S[:, -2]
    print(f"at t = {t[-2]:.4f}: max |S_t - Y| = {np.abs(S_pre - Y).max():.2e}")

    # Bridge variance check: Var(beta_t) should be t(T - t)/T
    var_emp = beta.var(axis=0, ddof=1)
    var_theory = brownian_bridge_variance(t, T)

    # Martingale check: E[S_t] should equal p0 at every t
    mean_S = S.mean(axis=0)
    se_S = S.std(axis=0, ddof=1) / np.sqrt(N)

    check_times = [0.0, 0.1, 0.25, 0.5, 2 / 3, 0.9, 0.99, 1.0]
    idx = [int(np.argmin(np.abs(t - c))) for c in check_times]
    table = pd.DataFrame({
        "t": t[idx],
        "mean S_t": mean_S[idx],
        "MC s.e.": se_S[idx],
        # At t = 0 and t = T-ish the s.e. can be ~1e-17 (floating-point noise around a
        # constant column), so z is reported only where the s.e. is meaningful.
        "z = (mean - p0)/se": np.where(se_S[idx] > 1e-12, (mean_S[idx] - p0) / np.maximum(se_S[idx], 1e-12), np.nan),
        "Var(beta) emp": var_emp[idx],
        "Var(beta) theory": var_theory[idx],
    })
    print("\n=== Martingale and bridge-variance check ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    z_all = (mean_S[1:] - p0) / se_S[1:]
    print(f"\nacross all {z_all.size} non-trivial grid times: max |z| = {np.abs(z_all).max():.2f}")
    print("note: z-scores at different t are strongly correlated (all paths share the same Y draws),")
    print(f"      so their common sign mostly reflects the realised Y-frequency, z_Y = "
          f"{(Y.mean() - p0) / np.sqrt(p0 * (1 - p0) / N):.2f}.")
    table.to_csv(RESULTS / "B_martingale_check.csv", index=False)

    # --- plots
    fig, (ax_m, ax_v) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_m.fill_between(t, mean_S - 1.96 * se_S, mean_S + 1.96 * se_S, color=COLOR_YES, alpha=0.18, lw=0,
                      label="95% MC band")
    ax_m.plot(t, mean_S, color=COLOR_YES, label=r"mean of $S_t$ across paths")
    ax_m.axhline(p0, color=INK, ls="--", lw=1.2, label=rf"$p_0 = {p0}$")
    ax_m.set(title=rf"Martingale check: $E^Q[S_t]$ ({N:,} paths)", xlabel="t", ylabel=r"$E^Q[S_t]$",
             ylim=(p0 - 0.03, p0 + 0.03))
    ax_m.legend(loc="lower left")

    ax_v.plot(t, var_theory, color=INK, ls="--", lw=1.2, label=r"theory $t(T-t)/T$")
    ax_v.plot(t, var_emp, color=COLOR_NO, label="empirical")
    ax_v.set(title=r"Brownian-bridge variance $\mathrm{Var}(\beta_{tT})$", xlabel="t", ylabel="variance")
    ax_v.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(RESULTS / "B_martingale_and_bridge.png")
    plt.close(fig)


if __name__ == "__main__":
    # Independent, reproducible streams: changing n_plot_paths cannot change the
    # validation draws, and vice versa.
    plot_seed, validation_seed = np.random.SeedSequence(seed).spawn(2)
    plot_sample_paths(np.random.default_rng(plot_seed))
    validate(np.random.default_rng(validation_seed))
    print(f"\nFigures written to {RESULTS.relative_to(ROOT)}/")
