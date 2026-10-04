"""Tests for Part C: Monte Carlo call pricing and the independent benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark import call_price_closed_form, call_price_quadrature
from src.option_pricing import (
    call_payoff,
    mc_convergence_study,
    price_european_call_mc,
    simulate_price_at_option_expiry,
)

P0, T, T_OPTION, K, KAPPA = 0.60, 1.0, 2 / 3, 0.70, 1.5

PARAM_SETS = [
    (0.60, 1.0, 2 / 3, 0.70, 1.5),
    (0.60, 1.0, 2 / 3, 0.40, 1.5),
    (0.30, 1.0, 0.20, 0.50, 3.0),
    (0.90, 1.0, 0.95, 0.95, 0.5),
    (0.50, 2.0, 1.00, 0.50, 1.0),
]


# ---- payoff and basic properties ----------------------------------------------

def test_payoff_nonnegative():
    rng = np.random.default_rng(0)
    S = simulate_price_at_option_expiry(P0, T, T_OPTION, KAPPA, 50_000, rng)
    assert np.all(call_payoff(S, K) >= 0.0)
    assert np.all((S >= 0.0) & (S <= 1.0))


def test_direct_sampling_is_martingale():
    rng = np.random.default_rng(1)
    S = simulate_price_at_option_expiry(P0, T, T_OPTION, KAPPA, 400_000, rng)
    se = S.std(ddof=1) / np.sqrt(S.size)
    assert abs(S.mean() - P0) < 4 * se


def test_call_price_decreases_with_strike_mc():
    """Common random numbers (same seed per strike) make the MC prices monotone
    pathwise; the CIs also have to be consistent with the benchmark ordering."""
    strikes = [0.4, 0.5, 0.6, 0.7, 0.8]
    prices = [price_european_call_mc(P0, T, T_OPTION, k, KAPPA, 100_000, np.random.default_rng(3)).option_price
              for k in strikes]
    assert all(a > b for a, b in zip(prices, prices[1:]))


def test_call_price_decreases_with_strike_independent_streams():
    """Without common random numbers, monotonicity must hold within MC tolerance."""
    strikes = [0.4, 0.5, 0.6, 0.7, 0.8]
    seeds = np.random.SeedSequence(4).spawn(len(strikes))
    res = [price_european_call_mc(P0, T, T_OPTION, k, KAPPA, 100_000, np.random.default_rng(s))
           for k, s in zip(strikes, seeds)]
    for a, b in zip(res, res[1:]):
        tol = 4 * np.hypot(a.standard_error, b.standard_error)
        assert a.option_price > b.option_price - tol


# ---- MC vs independent benchmark ---------------------------------------------

@pytest.mark.parametrize("params", PARAM_SETS)
def test_closed_form_matches_quadrature(params):
    assert call_price_closed_form(*params) == pytest.approx(call_price_quadrature(*params), abs=1e-9)


@pytest.mark.parametrize("params", PARAM_SETS)
def test_mc_matches_benchmark(params):
    p0, T_, T_opt, K_, kappa = params
    res = price_european_call_mc(p0, T_, T_opt, K_, kappa, 400_000, np.random.default_rng(5))
    benchmark = call_price_closed_form(*params)
    assert abs(res.option_price - benchmark) < 4 * res.standard_error


def test_ci_is_consistent_with_se():
    res = price_european_call_mc(P0, T, T_OPTION, K, KAPPA, 10_000, np.random.default_rng(6))
    assert res.ci_lower < res.option_price < res.ci_upper
    assert (res.ci_upper - res.ci_lower) == pytest.approx(2 * 1.959964 * res.standard_error, rel=1e-6)


# ---- benchmark limiting cases ------------------------------------------------

def test_benchmark_limits():
    # deep in the money: E[S - K] = p0 - K
    assert call_price_closed_form(P0, T, T_OPTION, -0.2, KAPPA) == pytest.approx(P0 + 0.2)
    assert call_price_closed_form(P0, T, T_OPTION, 1.0, KAPPA) == 0.0
    # T_option -> 0: no information yet, price -> max(p0 - K, 0)
    assert call_price_closed_form(P0, T, 1e-8, 0.5, KAPPA) == pytest.approx(0.1, abs=1e-3)
    assert call_price_closed_form(P0, T, 1e-8, K, KAPPA) == pytest.approx(0.0, abs=1e-3)
    # T_option -> T: S is (almost) Y, price -> p0 (1 - K)
    assert call_price_closed_form(P0, T, 1 - 1e-8, K, KAPPA) == pytest.approx(P0 * (1 - K), abs=1e-3)


# ---- Monte Carlo convergence ---------------------------------------------------

def test_standard_error_scales_as_inverse_sqrt_n():
    N = [1_000, 5_000, 10_000, 50_000, 100_000]
    table = mc_convergence_study(P0, T, T_OPTION, K, KAPPA, N, seed=7)
    slope = np.polyfit(np.log(table["N"]), np.log(table["standard_error"]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05)
    assert np.all(np.diff(table["standard_error"]) < 0)
    assert np.all(np.diff(table["ci_width"]) < 0)


def test_reported_se_matches_spread_of_repeated_estimates():
    rng = np.random.default_rng(8)
    reps = [price_european_call_mc(P0, T, T_OPTION, K, KAPPA, 5_000, rng) for _ in range(300)]
    sd = np.std([r.option_price for r in reps], ddof=1)
    mean_se = np.mean([r.standard_error for r in reps])
    assert sd == pytest.approx(mean_se, rel=0.15)


# ---- parameter validation ------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(T_option=0.0), dict(T_option=1.0), dict(T_option=1.5), dict(K=np.nan),
    dict(p0=0.0), dict(p0=1.0), dict(kappa=0.0), dict(n_simulations=0), dict(n_simulations=10.5),
])
def test_invalid_option_inputs_raise(bad):
    kwargs = dict(p0=P0, T=T, T_option=T_OPTION, K=K, kappa=KAPPA, n_simulations=100,
                  rng=np.random.default_rng(0))
    kwargs.update(bad)
    with pytest.raises(ValueError):
        price_european_call_mc(**kwargs)
