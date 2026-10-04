"""Monte Carlo pricing of a European call on a prediction-market YES contract.

Payoff at option expiry T_option < T:   H = max(S_{T_option} - K, 0)
Price with r = 0:                       C_0 = E^Q[H]

The payoff depends only on S_{T_option}, so instead of simulating whole paths we
sample the Brownian bridge directly at T_option:

    beta_{T_option, T} ~ N(0, T_option (T - T_option) / T),   independent of Y,

which is the exact marginal law of the bridge at that time. This is a
sampling shortcut, not a change of model: it gives the same distribution for
S_{T_option} as the full-path simulation in ``simulation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .information_model import (
    brownian_bridge_variance,
    check_model_parameters,
    check_positive_int,
    information_signal,
    signal_to_price,
)
from .simulation import simulate_terminal_outcome


@dataclass(frozen=True)
class MCPriceResult:
    """Monte Carlo price estimate with its sampling uncertainty."""

    option_price: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_simulations: int


def check_option_parameters(T: float, T_option: float, K: float) -> None:
    """T_option must lie strictly in (0, T); K must be finite.

    K outside [0, 1] is allowed (the payoff is still well defined: K <= 0 gives
    the linear payoff S - K, K >= 1 gives zero), but it is not economically
    interesting for an underlying that lives in [0, 1].
    """
    if not (np.isfinite(T_option) and 0.0 < T_option < T):
        raise ValueError(f"T_option must satisfy 0 < T_option < T = {T}; got {T_option}.")
    if not np.isfinite(K):
        raise ValueError(f"K must be finite; got {K}.")


def simulate_price_at_option_expiry(
    p0: float,
    T: float,
    T_option: float,
    kappa: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_simulations independent samples of S_{T_option} under Q.

    1. Y ~ Bernoulli(p0)
    2. beta_{T_option,T} ~ N(0, T_option (T - T_option) / T), independent of Y
    3. xi_{T_option} = kappa T_option Y + beta_{T_option,T}
    4. S_{T_option} from the same Bayesian price formula as the path simulation
    """
    check_model_parameters(p0, T, kappa)
    check_positive_int("n_simulations", n_simulations)
    check_option_parameters(T, T_option, K=0.0)

    Y = simulate_terminal_outcome(p0, n_simulations, rng)
    beta_sd = np.sqrt(brownian_bridge_variance(T_option, T))
    beta = beta_sd * rng.standard_normal(n_simulations)
    xi = information_signal(T_option, Y, beta, kappa)
    return signal_to_price(T_option, xi, p0, kappa, T)


def call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """H = max(S - K, 0)."""
    return np.maximum(S - K, 0.0)


def summarize_mc(samples: np.ndarray, confidence_level: float = 0.95) -> MCPriceResult:
    """Sample mean, standard error s/sqrt(N), and normal-approximation CI.

    With r = 0 the discount factor is 1, so the sample mean of payoffs is the
    price estimate directly.
    """
    n = samples.size
    mean = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = norm.ppf(0.5 + confidence_level / 2.0)
    return MCPriceResult(
        option_price=mean,
        standard_error=standard_error,
        ci_lower=mean - z * standard_error,
        ci_upper=mean + z * standard_error,
        confidence_level=confidence_level,
        n_simulations=n,
    )


def price_european_call_mc(
    p0: float,
    T: float,
    T_option: float,
    K: float,
    kappa: float,
    n_simulations: int,
    rng: np.random.Generator,
    confidence_level: float = 0.95,
) -> MCPriceResult:
    """Monte Carlo estimate of C_0 = E^Q[max(S_{T_option} - K, 0)] (r = 0)."""
    check_option_parameters(T, T_option, K)
    S_T_option = simulate_price_at_option_expiry(p0, T, T_option, kappa, n_simulations, rng)
    return summarize_mc(call_payoff(S_T_option, K), confidence_level)


def mc_convergence_study(
    p0: float,
    T: float,
    T_option: float,
    K: float,
    kappa: float,
    n_simulations_list: list[int],
    seed: int,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Price the same option at several sample sizes N.

    Each N gets its own independent child RNG stream (from one SeedSequence),
    so the estimates are independent of each other and the whole table is
    reproducible from ``seed``.
    """
    child_seeds = np.random.SeedSequence(seed).spawn(len(n_simulations_list))
    rows = []
    for n, child in zip(n_simulations_list, child_seeds):
        res = price_european_call_mc(
            p0, T, T_option, K, kappa, n, np.random.default_rng(child), confidence_level
        )
        rows.append({
            "N": n,
            "estimate": res.option_price,
            "standard_error": res.standard_error,
            "ci_lower": res.ci_lower,
            "ci_upper": res.ci_upper,
            "ci_width": res.ci_upper - res.ci_lower,
        })
    return pd.DataFrame(rows)
