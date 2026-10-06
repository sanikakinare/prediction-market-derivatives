"""Calendar time -> information time, and the coherent route for timing refinements.

Calibration (Part G) maps each market's calendar window [u_start, u_resolution]
onto the model's information time tau in [0, 1] (T = 1 in information time) and
runs the baseline model there:

    xi = kappa tau Y + beta_tau,   S = expit(logit p0 + A(tau, xi; kappa, T=1)),   S = Y at tau = 1.

``linear_clock`` (a constant rate) is exactly the constant-kappa baseline.

Timing refinements: why a clock and not xi_t = g(t) Y + beta_{tT} with non-linear g
----------------------------------------------------------------------------------
With calendar-time bridge noise, Girsanov gives the log-likelihood ratio of the
observed path as int_0^t [g'(s)(T-s) + g(s)] dM_s - (1/2) int_0^t h(s)^2 ds, with
M_s = xi_s/(T-s) and h = g' + g/(T-s). The weight g'(s)(T-s) + g(s) is constant
only for g(s) = kappa s, so for any other g the posterior depends on the whole
path, and "S_t = Q(Y=1 | xi_t)" would discard information and break
S_t = Q(Y=1 | F_t). A deterministic, increasing clock tau(t) avoids this: a
time-changed Markov process is still Markov and the filtrations correspond, so
every baseline formula holds with t -> tau(t), T -> 1. A non-linear clock was the
planned refinement; Part G's pre-specified goodness-of-fit test did not call for
it, so only the linear (baseline) clock is implemented.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logit
from scipy.stats import multivariate_normal

from .information_model import check_model_parameters, check_positive_int
from .simulation import MarketPaths, information_signal, prices_from_signal, simulate_brownian_bridge, \
    simulate_terminal_outcome


def linear_clock(u, u_start: float, u_resolution: float) -> np.ndarray:
    """Baseline clock: information time grows at a constant rate from u_start to u_resolution."""
    if not np.all(np.asarray(u_start) < u_resolution):
        raise ValueError("u_start must be before u_resolution.")
    return np.clip((np.asarray(u, dtype=float) - u_start) / (u_resolution - u_start), 0.0, 1.0)


def simulate_clock_paths(
    p0: float,
    kappa: float,
    tau_grid: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> MarketPaths:
    """Simulate paths on an information-time grid 0 = tau_0 < ... < tau_n = 1.

    Uses exactly the baseline simulation functions with time = tau and T = 1;
    the returned ``time`` field holds tau. Map back to calendar time with the
    calendar grid that generated tau_grid.
    """
    check_model_parameters(p0, 1.0, kappa)
    check_positive_int("n_paths", n_paths)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if tau_grid[0] != 0.0 or tau_grid[-1] != 1.0 or np.any(np.diff(tau_grid) <= 0):
        raise ValueError("tau_grid must increase strictly from 0 to 1.")
    Y = simulate_terminal_outcome(p0, n_paths, rng)
    beta = simulate_brownian_bridge(tau_grid, n_paths, rng)
    xi = information_signal(tau_grid[np.newaxis, :], Y[:, np.newaxis], beta, kappa)
    S = prices_from_signal(tau_grid, xi, Y, p0, kappa)
    return MarketPaths(tau_grid, Y, beta, xi, S)


def posterior_from_full_path(p0: float, kappa: float, tau: np.ndarray, xi_path: np.ndarray) -> float:
    """Brute-force Bayes: Q(Y=1 | xi at all of tau_1..tau_n), with 0 < tau_i < 1.

    Under Y = y the vector (xi_{tau_1}, ..., xi_{tau_n}) is Gaussian with mean
    kappa tau y and bridge covariance min(tau_i, tau_j) - tau_i tau_j. Used to
    check that the closed-form S (which uses only the latest xi) is the full posterior.
    """
    tau = np.asarray(tau, dtype=float)
    cov = np.minimum.outer(tau, tau) - np.outer(tau, tau)
    log_l1 = multivariate_normal.logpdf(xi_path, mean=kappa * tau, cov=cov)
    log_l0 = multivariate_normal.logpdf(xi_path, mean=np.zeros_like(tau), cov=cov)
    log_odds = logit(p0) + log_l1 - log_l0
    return float(1.0 / (1.0 + np.exp(-log_odds)))
