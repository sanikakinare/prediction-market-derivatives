"""Tests for the baseline information-based model (Parts A and B)."""

from __future__ import annotations

import numpy as np
import pytest

from src.information_model import (
    bayesian_exponent,
    brownian_bridge_variance,
    signal_to_price,
)
from src.simulation import (
    simulate_brownian_bridge,
    simulate_market_paths,
    simulate_terminal_outcome,
    time_grid,
)

P0, T, KAPPA, N_STEPS = 0.60, 1.0, 1.5, 300


@pytest.fixture(scope="module")
def paths():
    rng = np.random.default_rng(12345)
    return simulate_market_paths(P0, T, KAPPA, N_STEPS, 20_000, rng)


# ---- pathwise properties ------------------------------------------------------

def test_initial_price_equals_p0(paths):
    assert np.all(paths.prediction_price[:, 0] == P0)


def test_terminal_price_equals_outcome(paths):
    assert np.array_equal(paths.prediction_price[:, -1], paths.Y.astype(float))


def test_prices_bounded(paths):
    S = paths.prediction_price
    assert np.all(np.isfinite(S))
    assert np.all((S >= 0.0) & (S <= 1.0))


def test_bridge_pinned_at_both_ends(paths):
    assert np.allclose(paths.brownian_bridge[:, 0], 0.0, atol=1e-12)
    assert np.allclose(paths.brownian_bridge[:, -1], 0.0, atol=1e-12)


def test_information_signal_reveals_outcome_at_T(paths):
    assert np.allclose(paths.information_signal[:, -1], KAPPA * T * paths.Y)


# ---- distributional properties -----------------------------------------------

def test_bridge_variance_matches_theory():
    rng = np.random.default_rng(1)
    time = time_grid(T, 50)
    n = 200_000
    beta = simulate_brownian_bridge(time, n, rng)
    var_emp = beta.var(axis=0, ddof=1)
    var_theory = brownian_bridge_variance(time, T)
    # sample-variance s.e. for a Gaussian is var * sqrt(2/(n-1)); allow 5 s.e.
    tol = 5 * var_theory * np.sqrt(2 / (n - 1)) + 1e-12
    assert np.all(np.abs(var_emp - var_theory) <= tol)


def test_terminal_outcome_frequency():
    rng = np.random.default_rng(2)
    n = 200_000
    Y = simulate_terminal_outcome(P0, n, rng)
    assert set(np.unique(Y)) <= {0, 1}
    assert abs(Y.mean() - P0) < 5 * np.sqrt(P0 * (1 - P0) / n)


def test_martingale_property(paths):
    """E^Q[S_t] = p0 at every t, within Monte Carlo error."""
    S = paths.prediction_price
    n = S.shape[0]
    for t_check in (0.1, 0.25, 0.5, 2 / 3, 0.9):
        j = int(round(t_check / T * N_STEPS))
        se = S[:, j].std(ddof=1) / np.sqrt(n)
        assert abs(S[:, j].mean() - P0) < 4 * se, f"martingale check failed at t={t_check}"


# ---- formula-level checks -----------------------------------------------------

def test_stable_sigmoid_matches_naive_formula():
    """logit/expit form equals p0 e^A / ((1-p0) + p0 e^A) where the naive form is safe."""
    t = np.array([0.1, 0.5, 0.9])
    xi = np.array([0.3, -0.2, 1.1])
    A = bayesian_exponent(t, xi, KAPPA, T)
    naive = P0 * np.exp(A) / ((1 - P0) + P0 * np.exp(A))
    assert np.allclose(signal_to_price(t, xi, P0, KAPPA, T), naive, rtol=1e-12)


def test_no_overflow_for_extreme_signals():
    with np.errstate(over="raise", invalid="raise"):
        S = signal_to_price(np.array([0.999999, 0.999999]), np.array([50.0, -50.0]), P0, KAPPA, T)
    assert S[0] == pytest.approx(1.0) and S[1] == pytest.approx(0.0)


def test_exponent_rejects_t_equal_T():
    with pytest.raises(ValueError):
        bayesian_exponent(T, 0.0, KAPPA, T)


def test_reproducible_with_same_seed():
    a = simulate_market_paths(P0, T, KAPPA, 50, 10, np.random.default_rng(7))
    b = simulate_market_paths(P0, T, KAPPA, 50, 10, np.random.default_rng(7))
    assert np.array_equal(a.prediction_price, b.prediction_price)
