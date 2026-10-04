"""Monte Carlo simulation of full prediction-market paths under Q.

All randomness flows through a single ``np.random.Generator`` passed in by the
caller, so a whole experiment is reproducible from one seed. Arrays of paths
have shape (n_paths, n_steps + 1); column j corresponds to time[j].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .information_model import (
    check_model_parameters,
    check_positive_int,
    information_signal,
    signal_to_price,
)


@dataclass(frozen=True)
class MarketPaths:
    """Simulated paths. Path arrays have shape (n_paths, n_steps + 1)."""

    time: np.ndarray                # (n_steps + 1,)  t_0 = 0, ..., t_n = T
    Y: np.ndarray                   # (n_paths,)      terminal outcome in {0, 1}
    brownian_bridge: np.ndarray     # beta_{tT}
    information_signal: np.ndarray  # xi_t
    prediction_price: np.ndarray    # S_t


def time_grid(T: float, n_steps: int) -> np.ndarray:
    """Uniform grid 0 = t_0 < ... < t_n = T (last point is exactly T)."""
    check_positive_int("n_steps", n_steps)
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"T must be positive and finite; got {T}.")
    return np.linspace(0.0, T, n_steps + 1)


def simulate_terminal_outcome(p0: float, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Y ~ Bernoulli(p0) under Q, since Q(Y=1) = S_0 = p0."""
    check_positive_int("n_paths", n_paths)
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie strictly in (0, 1); got {p0}.")
    return (rng.random(n_paths) < p0).astype(np.int8)


def simulate_brownian_bridge(
    time: np.ndarray, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """beta_{tT} = W_t - (t/T) W_T on the grid, from a simulated Brownian motion W.

    No time-stepping approximation: W_t is a cumulative sum of independent
    N(0, dt) increments, which is exactly Brownian motion at the grid points, so
    the bridge values have exactly the right joint Gaussian law there (only
    Monte Carlo sampling error remains). The grid controls which times we
    observe, not the accuracy at those times. Between grid points nothing is
    simulated; plots simply join the dots.

    beta_0 = 0 and beta_T = 0 hold exactly in floating point because
    time[0] = 0 and time[-1] / T == 1.0.
    """
    check_positive_int("n_paths", n_paths)
    dt = np.diff(time)
    dW = rng.standard_normal((n_paths, dt.size)) * np.sqrt(dt)
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    T = time[-1]
    return W - (time / T) * W[:, [-1]]


def simulate_information_path(
    time: np.ndarray, Y: np.ndarray, beta: np.ndarray, kappa: float
) -> np.ndarray:
    """xi_t = kappa t Y + beta_{tT}, broadcast over paths (rows) and times (columns)."""
    return information_signal(time[np.newaxis, :], Y[:, np.newaxis], beta, kappa)


def prices_from_signal(
    time: np.ndarray, xi: np.ndarray, Y: np.ndarray, p0: float, kappa: float
) -> np.ndarray:
    """S_t from xi_t: Bayesian formula for t < T, and S_T = Y set explicitly."""
    T = time[-1]
    S = np.empty_like(xi, dtype=float)
    S[:, :-1] = signal_to_price(time[:-1], xi[:, :-1], p0, kappa, T)
    S[:, -1] = Y
    return S


def simulate_market_paths(
    p0: float,
    T: float,
    kappa: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> MarketPaths:
    """Simulate n_paths complete market paths.

    Order of operations mirrors the model: draw Y, draw independent bridge noise,
    form the information signal, then map information to price.

    Because S_t is a deterministic function of xi_t at the same time t, and xi_t
    is exact at grid points, the simulated prices carry no discretisation bias.
    """
    check_model_parameters(p0, T, kappa)
    check_positive_int("n_steps", n_steps)
    check_positive_int("n_paths", n_paths)
    time = time_grid(T, n_steps)
    Y = simulate_terminal_outcome(p0, n_paths, rng)
    beta = simulate_brownian_bridge(time, n_paths, rng)
    xi = simulate_information_path(time, Y, beta, kappa)
    S = prices_from_signal(time, xi, Y, p0, kappa)
    return MarketPaths(time, Y, beta, xi, S)


def simulate_market_path(
    p0: float, T: float, kappa: float, n_steps: int, rng: np.random.Generator
) -> MarketPaths:
    """Single-path convenience wrapper; arrays keep a leading axis of length 1."""
    return simulate_market_paths(p0, T, kappa, n_steps, 1, rng)
