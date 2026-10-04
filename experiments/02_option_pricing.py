"""Part C: Monte Carlo pricing of the European call, validated against benchmarks.

Run from the project root:
    .venv/bin/python experiments/02_option_pricing.py

Outputs (results/):
    C_convergence.csv          estimate, s.e. and 95% CI for each N
    C_replications.csv         repeated-run stability and CI coverage per N
    C_convergence.png          estimate +/- CI vs N, and s.e. vs N on log-log axes
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import call_price_closed_form, call_price_quadrature  # noqa: E402
from src.option_pricing import (  # noqa: E402
    call_payoff,
    mc_convergence_study,
    price_european_call_mc,
    simulate_price_at_option_expiry,
    summarize_mc,
)
from src.simulation import simulate_market_paths  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---- baseline parameters -----------------------------------------------------
p0 = 0.60
T = 1.0
T_option = 2 / 3
K = 0.70
kappa = 1.5
n_simulations = 100_000
n_steps = 300
seed = 42

N_LIST = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
N_REPLICATION_LIST = [1_000, 5_000, 10_000, 50_000, 100_000]
n_replications = 200
n_full_paths = 20_000

# ---- plot style (shared palette with experiment 01) --------------------------
COLOR_MC = "#2a78d6"
COLOR_REP = "#eb6834"
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

fmt = lambda x: f"{x:.6f}"  # noqa: E731


def main() -> None:
    # One SeedSequence, independent child streams for each experiment below.
    ss_baseline, ss_convergence, ss_replications, ss_paths = np.random.SeedSequence(seed).spawn(4)
    convergence_seed = int(ss_convergence.generate_state(1)[0])

    # ---- benchmarks ------------------------------------------------------------
    C_closed = call_price_closed_form(p0, T, T_option, K, kappa)
    C_quad = call_price_quadrature(p0, T, T_option, K, kappa)
    print("=== Benchmarks (no Monte Carlo) ===")
    print(f"closed form (exercise threshold): {C_closed:.10f}")
    print(f"numerical quadrature:             {C_quad:.10f}")
    print(f"difference:                       {C_closed - C_quad:.2e}")

    # ---- baseline MC price -----------------------------------------------------
    res = price_european_call_mc(p0, T, T_option, K, kappa, n_simulations, np.random.default_rng(ss_baseline))
    print(f"\n=== Baseline MC price (p0={p0}, T={T}, T_option={T_option:.4f}, K={K}, kappa={kappa}) ===")
    print(f"N = {res.n_simulations:,}")
    print(f"option price   = {res.option_price:.6f}")
    print(f"standard error = {res.standard_error:.6f}")
    print(f"95% CI         = [{res.ci_lower:.6f}, {res.ci_upper:.6f}]")
    print(f"benchmark inside CI: {res.ci_lower <= C_closed <= res.ci_upper}"
          f"   (z = {(res.option_price - C_closed) / res.standard_error:+.2f})")

    # ---- convergence study -----------------------------------------------------
    conv = mc_convergence_study(p0, T, T_option, K, kappa, N_LIST, convergence_seed)
    conv["error_vs_benchmark"] = conv["estimate"] - C_closed
    conv["z"] = conv["error_vs_benchmark"] / conv["standard_error"]
    conv["se_x_sqrtN"] = conv["standard_error"] * np.sqrt(conv["N"])
    print("\n=== Convergence study (independent stream per N) ===")
    print(conv.to_string(index=False, float_format=fmt))
    slope, intercept = np.polyfit(np.log(conv["N"]), np.log(conv["standard_error"]), 1)
    print(f"\nlog-log fit: standard_error ~ N^{slope:.4f}   (theory: N^-0.5)")
    conv.to_csv(RESULTS / "C_convergence.csv", index=False)

    # ---- repeated runs: is the reported s.e. honest, and does the CI cover? -----
    rep_streams = ss_replications.spawn(len(N_REPLICATION_LIST))
    rows = []
    for n, ss in zip(N_REPLICATION_LIST, rep_streams):
        rng = np.random.default_rng(ss)
        reps = [price_european_call_mc(p0, T, T_option, K, kappa, n, rng) for _ in range(n_replications)]
        est = np.array([r.option_price for r in reps])
        rows.append({
            "N": n,
            "mean_of_estimates": est.mean(),
            "sd_of_estimates": est.std(ddof=1),
            "mean_reported_se": np.mean([r.standard_error for r in reps]),
            "ci_coverage": np.mean([r.ci_lower <= C_closed <= r.ci_upper for r in reps]),
        })
    rep = pd.DataFrame(rows)
    print(f"\n=== {n_replications} independent replications per N ===")
    print(rep.to_string(index=False, float_format=fmt))
    print(f"(coverage s.e. for {n_replications} reps at 95%: "
          f"{np.sqrt(0.95 * 0.05 / n_replications):.3f})")
    rep.to_csv(RESULTS / "C_replications.csv", index=False)

    # ---- shortcut vs full paths: same law for S_{T_option}? ---------------------
    rng_paths = np.random.default_rng(ss_paths)
    paths = simulate_market_paths(p0, T, kappa, n_steps, n_full_paths, rng_paths)
    j = int(np.argmin(np.abs(paths.time - T_option)))
    S_full = paths.prediction_price[:, j]
    S_direct = simulate_price_at_option_expiry(p0, T, paths.time[j], kappa, n_full_paths, rng_paths)
    res_full = summarize_mc(call_payoff(S_full, K))
    res_direct = summarize_mc(call_payoff(S_direct, K))
    ks = ks_2samp(S_full, S_direct)
    print(f"\n=== Direct sampling at T_option vs full paths (N = {n_full_paths:,} each, t = {paths.time[j]:.6f}) ===")
    print(f"full paths:  {res_full.option_price:.6f} +/- {res_full.standard_error:.6f}")
    print(f"direct:      {res_direct.option_price:.6f} +/- {res_direct.standard_error:.6f}")
    print(f"benchmark:   {C_closed:.6f}")
    print(f"two-sample KS test on S_T_option: statistic = {ks.statistic:.4f}, p-value = {ks.pvalue:.3f}")

    plot_convergence(conv, rep, C_closed)
    print(f"\nFigures and tables written to {RESULTS.relative_to(ROOT)}/")


def plot_convergence(conv: pd.DataFrame, rep: pd.DataFrame, C_benchmark: float) -> None:
    fig, (ax_est, ax_se) = plt.subplots(1, 2, figsize=(11, 4.4))

    N = conv["N"].to_numpy()
    ax_est.axhline(C_benchmark, color=INK, ls="--", lw=1.2, label=f"benchmark {C_benchmark:.5f}")
    ax_est.errorbar(N, conv["estimate"], yerr=conv["estimate"] - conv["ci_lower"], fmt="o",
                    color=COLOR_MC, ms=6, capsize=4, lw=1.6, label="MC estimate, 95% CI")
    ax_est.set(xscale="log", xlabel="number of simulations N", ylabel="call price",
               title="MC estimate converges to the benchmark")
    ax_est.legend(loc="upper right")

    ax_se.loglog(N, conv["standard_error"], "o-", color=COLOR_MC, ms=6, label="reported s.e. (single run)")
    ax_se.loglog(rep["N"], rep["sd_of_estimates"], "s", color=COLOR_REP, ms=7, mfc="none", mew=1.6,
                 label=f"sd across {n_replications} runs")
    ref = conv["standard_error"].iloc[0] * np.sqrt(N[0] / N)
    ax_se.loglog(N, ref, ls="--", color=INK_MUTED, lw=1.2, label=r"$\propto 1/\sqrt{N}$")
    ax_se.set(xlabel="number of simulations N", ylabel="standard error",
              title=r"Standard error falls at the $1/\sqrt{N}$ rate")
    ax_se.legend(loc="lower left")

    fig.suptitle(rf"European call: $p_0={p0}$, $K={K}$, $T_{{option}}={T_option:.3f}$, $\kappa={kappa}$",
                 color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "C_convergence.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
