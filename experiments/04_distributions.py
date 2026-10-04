"""Part E: the distributional mechanism behind the premium sensitivities.

Run from the project root:
    .venv/bin/python experiments/04_distributions.py

Monte Carlo samples come from the direct T_option sampler (option_pricing.py);
exact quantities come from expiry_distribution.py and benchmark.py, which share
no code with the sampler.

Outputs (results/):
    E_baseline_expiry_distribution.png   E1 + E6: histogram vs exact, CDF vs exact
    E_conditional_distributions.png      E2: S | Y=1 and S | Y=0, calibration check
    E_payoff_distribution.png            E3: point mass at zero + continuous part
    E_kappa_distributions.png            E4: kappa = 0.5, 1.5, 3.0
    E_expiry_distributions.png           E5: T_option = 0.1, 2/3, 0.9
    E_distribution_summary.png           same-s collapse, dispersion and premium vs s
    E_distribution_summary.csv           scenario table (MC and exact)
    E_baseline_quantiles.csv             MC vs exact quantiles
    E_validation.csv                     z-scores / KS tests per scenario
    E_replication_diagnostic.csv         z / KS behaviour over many independent runs
    E_strike_below_p0.csv                exercise probability vs premium for K < p0
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import call_price_closed_form, exercise_probabilities  # noqa: E402
from src.expiry_distribution import (  # noqa: E402
    expiry_price_bin_probabilities,
    expiry_price_cdf,
    expiry_price_moments,
    expiry_price_quantile,
    information_ratio,
    premium_contribution_by_bin,
)
from src.option_pricing import call_payoff, simulate_outcome_and_price_at_option_expiry  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---- parameters ------------------------------------------------------------------
p0, T, K = 0.60, 1.0, 0.70
seed = 42
n_simulations = 1_000_000
# Replication diagnostic: are single-run z-scores / KS p-values behaving as noise should?
n_replications = 50
n_replication_simulations = 200_000

SCENARIOS = {  # name: (kappa, T_option)
    "Low kappa": (0.5, 2 / 3),
    "Baseline": (1.5, 2 / 3),
    "High kappa": (3.0, 2 / 3),
    "Early expiry": (1.5, 0.10),
    "Late expiry": (1.5, 0.90),
    # Same information ratio s as the baseline, reached with half the kappa and a later expiry.
    "Same s as baseline": (0.75, 8 / 9),
}
KAPPA_SCENARIOS = ["Low kappa", "Baseline", "High kappa"]
EXPIRY_SCENARIOS = ["Early expiry", "Baseline", "Late expiry"]

PRICE_EDGES = np.linspace(0.0, 1.0, 51)        # 0.02-wide price bins
PREMIUM_EDGES = np.linspace(K, 1.0, 16)        # 0.02-wide bins above the strike
PAYOFF_EDGES = np.linspace(0.0, 1.0 - K, 31)   # 0.01-wide payoff bins
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# ---- plot style (shared palette) -------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
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


# ---- statistics -------------------------------------------------------------------

def simulate(scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    kappa, T_option = SCENARIOS[scenario]
    return simulate_outcome_and_price_at_option_expiry(p0, T, T_option, kappa, n_simulations, rng)


def scenario_statistics(scenario: str, Y: np.ndarray, S: np.ndarray) -> dict:
    """MC statistics and their exact counterparts for one scenario."""
    kappa, T_option = SCENARIOS[scenario]
    N = S.size
    H = call_payoff(S, K)
    exercised = S > K

    mean, var = S.mean(), S.var(ddof=1)
    m4 = np.mean((S - mean) ** 4)
    prob_ex = exercised.mean()

    exact_moments = expiry_price_moments(p0, T, T_option, kappa)
    exact_ex = exercise_probabilities(p0, T, T_option, K, kappa)
    exact_C0 = call_price_closed_form(p0, T, T_option, K, kappa)

    return {
        "scenario": scenario,
        "kappa": kappa,
        "T_option": T_option,
        "information_ratio_s": information_ratio(T, T_option, kappa),
        "mc_mean_S": mean,
        "mc_mean_S_se": S.std(ddof=1) / np.sqrt(N),
        "exact_mean_S": exact_moments["mean"],
        "mc_var_S": var,
        "mc_var_S_se": np.sqrt((m4 - var**2) / N),
        "exact_var_S": exact_moments["variance"],
        "exact_var_fraction": exact_moments["variance_fraction"],
        "mc_prob_exercise": prob_ex,
        "mc_prob_exercise_se": np.sqrt(prob_ex * (1 - prob_ex) / N),
        "exact_prob_exercise": exact_ex["total"],
        "mc_prob_exercise_given_yes": exercised[Y == 1].mean(),
        "mc_prob_exercise_given_yes_se": np.sqrt(exact_ex["given_yes"] * (1 - exact_ex["given_yes"]) / (Y == 1).sum()),
        "exact_prob_exercise_given_yes": exact_ex["given_yes"],
        "mc_prob_exercise_given_no": exercised[Y == 0].mean(),
        "mc_prob_exercise_given_no_se": np.sqrt(exact_ex["given_no"] * (1 - exact_ex["given_no"]) / (Y == 0).sum()),
        "exact_prob_exercise_given_no": exact_ex["given_no"],
        "mc_fraction_zero_payoff": np.mean(H == 0.0),
        "mc_mean_payoff": H.mean(),
        "mc_mean_payoff_se": H.std(ddof=1) / np.sqrt(N),
        "exact_C0": exact_C0,
        "mc_mean_payoff_given_exercise": H[exercised].mean(),
        "exact_mean_payoff_given_exercise": exact_C0 / exact_ex["total"],
        "ks_pvalue": kstest(S, lambda x: expiry_price_cdf(x, p0, T, T_option, kappa)).pvalue,
    }


def validation_table(stats: pd.DataFrame) -> pd.DataFrame:
    z = lambda mc, ex, se: (stats[mc] - stats[ex]) / stats[se]  # noqa: E731
    return pd.DataFrame({
        "scenario": stats["scenario"],
        "z_mean_vs_p0": (stats["mc_mean_S"] - p0) / stats["mc_mean_S_se"],
        "z_var_vs_exact": z("mc_var_S", "exact_var_S", "mc_var_S_se"),
        "z_exercise_vs_benchmark": z("mc_prob_exercise", "exact_prob_exercise", "mc_prob_exercise_se"),
        "z_exercise_yes": z("mc_prob_exercise_given_yes", "exact_prob_exercise_given_yes", "mc_prob_exercise_given_yes_se"),
        "z_exercise_no": z("mc_prob_exercise_given_no", "exact_prob_exercise_given_no", "mc_prob_exercise_given_no_se"),
        "z_payoff_vs_C0": z("mc_mean_payoff", "exact_C0", "mc_mean_payoff_se"),
        "ks_pvalue_vs_exact_cdf": stats["ks_pvalue"],
        "exact_mean_minus_p0": stats["exact_mean_S"] - p0,
    })


def replication_diagnostic(seed_sequence: np.random.SeedSequence) -> pd.DataFrame:
    """Many independent smaller runs per scenario: z-scores should be ~N(0, 1) and
    KS p-values ~Uniform(0, 1) if the sampler and the exact distribution agree."""
    rows = []
    for name, ss in zip(SCENARIOS, seed_sequence.spawn(len(SCENARIOS))):
        kappa, T_option = SCENARIOS[name]
        exact_C0 = call_price_closed_form(p0, T, T_option, K, kappa)
        z_mean, z_payoff, ks_p = [], [], []
        for rs in ss.spawn(n_replications):
            _, S = simulate_outcome_and_price_at_option_expiry(
                p0, T, T_option, kappa, n_replication_simulations, np.random.default_rng(rs))
            H = call_payoff(S, K)
            z_mean.append((S.mean() - p0) / (S.std(ddof=1) / np.sqrt(S.size)))
            z_payoff.append((H.mean() - exact_C0) / (H.std(ddof=1) / np.sqrt(H.size)))
            ks_p.append(kstest(S, lambda x: expiry_price_cdf(x, p0, T, T_option, kappa)).pvalue)
        rows.append({"scenario": name, "mean_z_mean": np.mean(z_mean), "sd_z_mean": np.std(z_mean, ddof=1),
                     "mean_z_payoff": np.mean(z_payoff), "sd_z_payoff": np.std(z_payoff, ddof=1),
                     "ks_reject_rate_5pct": np.mean(np.array(ks_p) < 0.05)})
    return pd.DataFrame(rows)


def mc_bin_probabilities(S: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(S, bins=edges)[0] / S.size


def mc_premium_contribution(S: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(S, bins=edges, weights=call_payoff(S, K))[0] / S.size


# ---- figures ----------------------------------------------------------------------

def mark_p0_and_strike(ax, label: bool = True) -> None:
    ax.axvline(p0, color=INK_MUTED, ls="--", lw=1.1, label=rf"$p_0={p0}$" if label else None)
    ax.axvline(K, color=AQUA, lw=1.4, label=rf"strike $K={K}$" if label else None)


def plot_baseline(S: np.ndarray, st: dict) -> None:
    kappa, T_option = SCENARIOS["Baseline"]
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    mc = mc_bin_probabilities(S, PRICE_EDGES)
    exact = expiry_price_bin_probabilities(PRICE_EDGES, p0, T, T_option, kappa)
    ax_h.bar(PRICE_EDGES[:-1], mc, width=np.diff(PRICE_EDGES), align="edge", color=BLUE, alpha=0.35,
             label=f"Monte Carlo (N = {S.size:,})")
    ax_h.stairs(exact, PRICE_EDGES, color=INK, lw=1.3, label="exact (model)")
    mark_p0_and_strike(ax_h)
    ax_h.axvline(st["mc_mean_S"], color=BLUE, ls=":", lw=1.6, label=f"MC mean = {st['mc_mean_S']:.4f}")
    ax_h.text(0.22, 0.42,
              f"Var(S)    = {st['mc_var_S']:.4f}\nQ(S > K)  = {st['mc_prob_exercise']:.4f}",
              transform=ax_h.transAxes, va="top", color=INK, fontsize=9, family="monospace")
    ax_h.set(xlabel=r"expiry price $S_{T_{option}}$", ylabel="probability per 0.02 bin",
             title=r"E1  Distribution of $S_{T_{option}}$ (baseline)")
    ax_h.legend(loc="upper center", fontsize=8.5)

    x = np.linspace(0.0005, 0.9995, 800)
    S_sorted = np.sort(S)
    ax_c.plot(S_sorted[::200], np.arange(1, S.size + 1)[::200] / S.size, color=BLUE, lw=3, alpha=0.45,
              label="empirical CDF (MC)")
    ax_c.plot(x, expiry_price_cdf(x, p0, T, T_option, kappa), color=INK, lw=1.2, label="exact CDF")
    for q in QUANTILES:
        xq = expiry_price_quantile(q, p0, T, T_option, kappa)
        ax_c.plot([xq], [q], "o", color=ORANGE, ms=5)
        ax_c.annotate(f"{int(q * 100)}%: {xq:.3f}", (xq, q), textcoords="offset points", xytext=(6, -10),
                      fontsize=8, color=INK_MUTED)
    mark_p0_and_strike(ax_c, label=False)
    ax_c.set(xlabel=r"$x$", ylabel=r"$Q(S_{T_{option}} \leq x)$",
             title=f"E6  Exact CDF vs Monte Carlo (KS p-value = {st['ks_pvalue']:.2f})")
    ax_c.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "E_baseline_expiry_distribution.png")
    plt.close(fig)


def plot_conditional(Y: np.ndarray, S: np.ndarray, st: dict) -> None:
    kappa, T_option = SCENARIOS["Baseline"]
    fig, (ax_h, ax_cal) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})

    for y, color, name, key in ((1, BLUE, "YES worlds (Y = 1)", "given_yes"),
                                (0, ORANGE, "NO worlds (Y = 0)", "given_no")):
        mc = mc_bin_probabilities(S[Y == y], PRICE_EDGES)
        exact = expiry_price_bin_probabilities(PRICE_EDGES, p0, T, T_option, kappa, outcome=y)
        prob = st[f"exact_prob_exercise_{key}"]
        ax_h.bar(PRICE_EDGES[:-1], mc, width=np.diff(PRICE_EDGES), align="edge", color=color, alpha=0.3)
        ax_h.stairs(exact, PRICE_EDGES, color=color, lw=1.6, label=f"{name}:  Q(S > K | Y={y}) = {prob:.3f}")
    mark_p0_and_strike(ax_h)
    ax_h.axvspan(K, 1.0, color=AQUA, alpha=0.06, lw=0)
    ax_h.text(K + 0.01, 0.55, "exercised\n(S > K)", transform=ax_h.get_xaxis_transform(), color=AQUA, va="top",
              fontsize=9)
    ax_h.set(xlabel=r"expiry price $S_{T_{option}}$", ylabel="probability per 0.02 bin (within each outcome)",
             title="E2  Expiry price, split by the eventual outcome (bars MC, lines exact)")
    ax_h.legend(loc="upper center", bbox_to_anchor=(0.42, 1.0), fontsize=8.5)

    # Calibration: among simulated worlds whose expiry price is about x, how many resolve YES?
    cal_edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(S, cal_edges) - 1, 0, len(cal_edges) - 2)
    counts = np.bincount(idx, minlength=len(cal_edges) - 1)
    yes = np.bincount(idx, weights=Y, minlength=len(cal_edges) - 1)
    mean_S = np.bincount(idx, weights=S, minlength=len(cal_edges) - 1)
    ok = counts >= 200
    frac = yes[ok] / counts[ok]
    se = np.sqrt(frac * (1 - frac) / counts[ok])
    ax_cal.plot([0, 1], [0, 1], color=INK_MUTED, ls="--", lw=1.1, label="fraction YES = price")
    ax_cal.errorbar(mean_S[ok] / counts[ok], frac, yerr=1.96 * se, fmt="o", color=BLUE, ms=5, capsize=2,
                    label="simulated worlds")
    ax_cal.set(xlabel=r"expiry price $S_{T_{option}}$ (bin average)", ylabel="fraction that resolve YES",
               title="Prices are calibrated, not clairvoyant", xlim=(0, 1), ylim=(0, 1))
    ax_cal.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "E_conditional_distributions.png")
    plt.close(fig)


def plot_payoff(S: np.ndarray, st: dict) -> None:
    kappa, T_option = SCENARIOS["Baseline"]
    H = call_payoff(S, K)
    fig, (ax_m, ax_h) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.8]})

    p_zero, p_pos = st["mc_fraction_zero_payoff"], 1 - st["mc_fraction_zero_payoff"]
    bars = ax_m.bar(["H = 0\n(not exercised)", "H > 0\n(exercised)"], [p_zero, p_pos],
                    color=[INK_MUTED, BLUE], width=0.55)
    for b, v in zip(bars, (p_zero, p_pos)):
        ax_m.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", color=INK, fontsize=9)
    ax_m.set(ylabel="probability", ylim=(0, 1), title="E3  Point mass at zero payoff")

    pos = H[H > 0]
    mc = np.histogram(pos, bins=PAYOFF_EDGES)[0] / pos.size
    exact = (expiry_price_bin_probabilities(K + PAYOFF_EDGES, p0, T, T_option, kappa)
             / st["exact_prob_exercise"])
    ax_h.bar(PAYOFF_EDGES[:-1], mc, width=np.diff(PAYOFF_EDGES), align="edge", color=BLUE, alpha=0.35,
             label="Monte Carlo")
    ax_h.stairs(exact, PAYOFF_EDGES, color=INK, lw=1.3, label="exact")
    ax_h.axvline(st["mc_mean_payoff_given_exercise"], color=ORANGE, lw=1.4,
                 label=f"E[H | exercised] = {st['mc_mean_payoff_given_exercise']:.4f}")
    ax_h.text(0.02, 0.97,
              f"E[H] = Q(exercise) x E[H | exercised]\n"
              f"      = {st['mc_prob_exercise']:.4f} x {st['mc_mean_payoff_given_exercise']:.4f}"
              f" = {st['mc_mean_payoff']:.4f}\n"
              f"closed-form premium C0 = {st['exact_C0']:.4f}",
              transform=ax_h.transAxes, va="top", fontsize=8.5, color=INK, family="monospace")
    ax_h.set(xlabel=r"payoff $H=(S_{T_{option}}-K)^+$, given $H>0$",
             ylabel="probability per 0.01 bin (given exercise)",
             title=r"Continuous part: payoff when exercised (max $1-K=0.3$)")
    ax_h.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "E_payoff_distribution.png")
    plt.close(fig)


def plot_comparison(names: list[str], samples: dict, stats: pd.DataFrame, filename: str, title: str,
                    vary: str) -> None:
    """Row 1: expiry-price distributions. Row 2: where the premium comes from."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharey="row")
    for col, name in enumerate(names):
        kappa, T_option = SCENARIOS[name]
        _, S = samples[name]
        st = stats.set_index("scenario").loc[name]
        label = rf"$\kappa={kappa}$" if vary == "kappa" else rf"$T_{{option}}={T_option:.3g}$"

        ax = axes[0, col]
        ax.bar(PRICE_EDGES[:-1], mc_bin_probabilities(S, PRICE_EDGES), width=np.diff(PRICE_EDGES),
               align="edge", color=BLUE, alpha=0.35, label="Monte Carlo")
        ax.stairs(expiry_price_bin_probabilities(PRICE_EDGES, p0, T, T_option, kappa), PRICE_EDGES,
                  color=INK, lw=1.2, label="exact")
        mark_p0_and_strike(ax, label=(col == 0))
        ax.set(title=f"{name}: {label}  (s = {st['information_ratio_s']:.2f})",
               xlabel=r"expiry price $S_{T_{option}}$")
        ax.text(0.03, 0.97,
                f"mean      = {st['mc_mean_S']:.4f}\nVar(S)    = {st['mc_var_S']:.4f}\n"
                f"Q(S > K)  = {st['mc_prob_exercise']:.4f}",
                transform=ax.transAxes, va="top", fontsize=8.5, family="monospace", color=INK)

        ax = axes[1, col]
        ax.bar(PREMIUM_EDGES[:-1], mc_premium_contribution(S, PREMIUM_EDGES), width=np.diff(PREMIUM_EDGES),
               align="edge", color=ORANGE, alpha=0.4, label="Monte Carlo")
        ax.stairs(premium_contribution_by_bin(PREMIUM_EDGES, K, p0, T, T_option, kappa), PREMIUM_EDGES,
                  color=INK, lw=1.2, label="exact")
        ax.set(xlabel=r"expiry price $S_{T_{option}}$ (above $K$)", xlim=(K - 0.01, 1.0))
        ax.text(0.03, 0.97, f"sum of bars = C0 = {st['mc_mean_payoff']:.4f}\n(closed form {st['exact_C0']:.4f})",
                transform=ax.transAxes, va="top", fontsize=8.5, family="monospace", color=INK)
    axes[0, 0].set_ylabel("probability per 0.02 bin")
    axes[1, 0].set_ylabel(r"contribution to $C_0$: $E[(S-K)\,1\{S\in\mathrm{bin}\}]$")
    axes[0, 0].legend(loc="center left", fontsize=8)
    axes[1, 0].legend(loc="center left", fontsize=8)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / filename)
    plt.close(fig)


def plot_summary(samples: dict, stats: pd.DataFrame) -> None:
    fig, (ax_s, ax_v, ax_c) = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) same s, different (kappa, T_option): same distribution
    for name, color in (("Baseline", BLUE), ("Same s as baseline", ORANGE)):
        kappa, T_option = SCENARIOS[name]
        ax_s.stairs(mc_bin_probabilities(samples[name][1], PRICE_EDGES), PRICE_EDGES, color=color, lw=2.2,
                    alpha=0.8, label=rf"MC: $\kappa={kappa}$, $T_{{option}}={T_option:.3g}$")
    kappa, T_option = SCENARIOS["Baseline"]
    ax_s.stairs(expiry_price_bin_probabilities(PRICE_EDGES, p0, T, T_option, kappa), PRICE_EDGES,
                color=INK, lw=1.0, ls="--", label="exact, s = 2.12")
    mark_p0_and_strike(ax_s, label=False)
    ax_s.set(xlabel=r"expiry price $S_{T_{option}}$", ylabel="probability per 0.02 bin",
             title=r"Same $s$ $\Rightarrow$ same distribution")
    ax_s.legend(loc="upper center", fontsize=8)

    # (b), (c): exact curves along s (any (kappa, T_option) with that s gives the same values)
    s_grid = np.linspace(0.05, 8.0, 160)
    T_ref = 2 / 3
    kappa_grid = s_grid / np.sqrt(T_ref * T / (T - T_ref))
    var_frac = [expiry_price_moments(p0, T, T_ref, k)["variance_fraction"] for k in kappa_grid]
    premium = [call_price_closed_form(p0, T, T_ref, K, k) for k in kappa_grid]
    ax_v.plot(s_grid, var_frac, color=INK, lw=1.3)
    ax_c.plot(s_grid, premium, color=INK, lw=1.3)
    ax_c.axhline(p0 * (1 - K), color=INK_MUTED, ls="--", lw=1.0)
    ax_c.text(8.0, p0 * (1 - K), r"$p_0(1-K)$", ha="right", va="bottom", color=INK_MUTED, fontsize=8)

    markers = {"Low kappa": ("o", BLUE, (8, 2)), "Baseline": ("D", INK, (8, -12)),
               "High kappa": ("o", BLUE, (-62, 4)), "Early expiry": ("s", ORANGE, (8, -12)),
               "Late expiry": ("s", ORANGE, (6, -14))}
    for name, (m, color, offset) in markers.items():
        st = stats.set_index("scenario").loc[name]
        s = st["information_ratio_s"]
        ax_v.plot(s, st["mc_var_S"] / (p0 * (1 - p0)), m, color=color, ms=7, mfc="white", mew=1.8)
        ax_c.plot(s, st["mc_mean_payoff"], m, color=color, ms=7, mfc="white", mew=1.8)
        for ax, yv in ((ax_v, st["mc_var_S"] / (p0 * (1 - p0))), (ax_c, st["mc_mean_payoff"])):
            ax.annotate(name, (s, yv), textcoords="offset points", xytext=offset, fontsize=8, color=INK_MUTED)
    handles = [plt.Line2D([], [], color=INK, lw=1.3, label="exact"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, mfc="white", mew=1.8, label=r"vary $\kappa$ (MC)"),
               plt.Line2D([], [], marker="s", ls="", color=ORANGE, mfc="white", mew=1.8,
                          label=r"vary $T_{option}$ (MC)")]
    ax_v.legend(handles=handles, loc="lower right", fontsize=8)
    ax_v.set(xlabel="information ratio $s$", ylabel=r"Var$(S)$ / $p_0(1-p_0)$", ylim=(0, 1.05),
             title="Dispersion of expiry prices (mean fixed at 0.6)")
    ax_c.set(xlabel="information ratio $s$", ylabel=r"call premium $C_0$",
             title="Call premium")
    fig.suptitle(r"$\kappa$ and $T_{option}$ act through the same channel: "
                 r"$s=\kappa\sqrt{T_{option}T/(T-T_{option})}$", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "E_distribution_summary.png")
    plt.close(fig)


# ---- main -------------------------------------------------------------------------

def strike_below_p0_check() -> pd.DataFrame:
    """For K < p0 the exercise probability falls with information while the premium still rises."""
    rows = []
    for K_ in (0.5, K):
        for kappa in (0.25, 0.5, 1.5, 3.0, 6.0):
            ex = exercise_probabilities(p0, T, 2 / 3, K_, kappa)
            rows.append({"K": K_, "kappa": kappa, "s": information_ratio(T, 2 / 3, kappa),
                         "prob_exercise": ex["total"], "call_premium": call_price_closed_form(p0, T, 2 / 3, K_, kappa)})
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 160)
    streams = np.random.SeedSequence(seed).spawn(len(SCENARIOS))
    samples = {name: simulate(name, np.random.default_rng(ss)) for name, ss in zip(SCENARIOS, streams)}
    stats = pd.DataFrame([scenario_statistics(name, *samples[name]) for name in SCENARIOS])
    stats.to_csv(RESULTS / "E_distribution_summary.csv", index=False)

    base = stats.set_index("scenario").loc["Baseline"]
    Y_b, S_b = samples["Baseline"]

    # E1
    print(f"=== E1  Baseline expiry-price distribution (N = {n_simulations:,}) ===")
    print(f"E[S]      MC {base['mc_mean_S']:.5f} (s.e. {base['mc_mean_S_se']:.5f})   exact {base['exact_mean_S']:.5f}")
    print(f"Var(S)    MC {base['mc_var_S']:.5f} (s.e. {base['mc_var_S_se']:.5f})   exact {base['exact_var_S']:.5f}"
          f"   = {base['exact_var_fraction']:.1%} of the maximum p0(1-p0) = {p0 * (1 - p0):.2f}")
    print(f"Q(S > K)  MC {base['mc_prob_exercise']:.5f}   exact {base['exact_prob_exercise']:.5f}")
    for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)):
        print(f"  Q({lo:.1f} < S <= {hi:.1f}) = {np.mean((S_b > lo) & (S_b <= hi)):.4f}")
    quant = pd.DataFrame({"quantile": QUANTILES,
                          "mc": np.quantile(S_b, QUANTILES),
                          "exact": [expiry_price_quantile(q, p0, T, *SCENARIOS["Baseline"][::-1]) for q in QUANTILES]})
    quant.to_csv(RESULTS / "E_baseline_quantiles.csv", index=False)
    print(quant.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # E2
    print("\n=== E2  Conditional on the eventual outcome ===")
    print(f"Q(S > K | Y=1)  MC {base['mc_prob_exercise_given_yes']:.4f}   benchmark Phi(d1) {base['exact_prob_exercise_given_yes']:.4f}")
    print(f"Q(S > K | Y=0)  MC {base['mc_prob_exercise_given_no']:.4f}   benchmark Phi(d0) {base['exact_prob_exercise_given_no']:.4f}")
    print(f"E[S | Y=1] = {S_b[Y_b == 1].mean():.4f},  E[S | Y=0] = {S_b[Y_b == 0].mean():.4f}")
    print(f"Q(S < 0.5 | Y=1) = {np.mean(S_b[Y_b == 1] < 0.5):.4f}  (YES worlds that look like NO at expiry)")
    print(f"Q(S > 0.5 | Y=0) = {np.mean(S_b[Y_b == 0] > 0.5):.4f}  (NO worlds that look like YES at expiry)")

    # E3
    print("\n=== E3  Payoff distribution ===")
    print(f"fraction with zero payoff     {base['mc_fraction_zero_payoff']:.4f}")
    print(f"probability of exercise       {base['mc_prob_exercise']:.4f}")
    print(f"average payoff E[H]           {base['mc_mean_payoff']:.6f} (s.e. {base['mc_mean_payoff_se']:.6f})"
          f"   closed-form C0 {base['exact_C0']:.6f}")
    print(f"E[H | exercised]              {base['mc_mean_payoff_given_exercise']:.4f}"
          f"   exact {base['exact_mean_payoff_given_exercise']:.4f}")

    # E4, E5 and the summary table
    cols = ["scenario", "kappa", "T_option", "information_ratio_s", "mc_mean_S", "mc_var_S",
            "exact_var_fraction", "mc_prob_exercise", "mc_mean_payoff", "exact_C0", "mc_mean_payoff_given_exercise"]
    print("\n=== Scenario summary (MC, N = 1,000,000 each; exact C0 for reference) ===")
    print(stats[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Validation
    val = validation_table(stats)
    val.to_csv(RESULTS / "E_validation.csv", index=False)
    print("\n=== Validation: z = (MC - exact) / s.e.;  KS test of MC sample vs exact CDF ===")
    print(val.to_string(index=False, float_format=lambda x: f"{x:+.2f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"))
    zcols = [c for c in val.columns if c.startswith("z_")]
    print(f"max |z| over {val[zcols].size} comparisons = {val[zcols].abs().to_numpy().max():.2f};  "
          f"min KS p-value = {val['ks_pvalue_vs_exact_cdf'].min():.3f}")
    rep = replication_diagnostic(np.random.SeedSequence([seed, 1]))
    rep.to_csv(RESULTS / "E_replication_diagnostic.csv", index=False)
    print(f"\n=== Replication diagnostic: {n_replications} independent runs of N = {n_replication_simulations:,} "
          "per scenario (expect z mean ~0, sd ~1; KS reject rate ~0.05) ===")
    print(rep.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    s_idx = stats.set_index("scenario")
    for group, label in ((KAPPA_SCENARIOS, "kappa"), (EXPIRY_SCENARIOS, "T_option")):
        mc_var = s_idx.loc[group, "mc_var_S"].to_numpy()
        ex_var = s_idx.loc[group, "exact_var_S"].to_numpy()
        print(f"Var(S) increasing in {label}: MC {np.all(np.diff(mc_var) > 0)} {np.round(mc_var, 4).tolist()}, "
              f"exact {np.all(np.diff(ex_var) > 0)}")

    below = strike_below_p0_check()
    below.to_csv(RESULTS / "E_strike_below_p0.csv", index=False)
    print("\n=== Exercise probability vs premium as information grows (T_option = 2/3) ===")
    print(below.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_baseline(S_b, base)
    plot_conditional(Y_b, S_b, base)
    plot_payoff(S_b, base)
    plot_comparison(KAPPA_SCENARIOS, samples, stats, "E_kappa_distributions.png",
                    r"E4  Raising $\kappa$ with $T_{option}=2/3$ fixed: same mean, wider spread, higher premium",
                    vary="kappa")
    plot_comparison(EXPIRY_SCENARIOS, samples, stats, "E_expiry_distributions.png",
                    r"E5  Later option expiry with $\kappa=1.5$ fixed: same mean, wider spread, higher premium",
                    vary="T_option")
    plot_summary(samples, stats)
    print(f"\nFigures and tables written to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
