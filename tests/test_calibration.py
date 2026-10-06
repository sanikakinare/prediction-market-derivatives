"""Tests for Part G: calibration protocol, information-time model, pricing consistency.

No test asserts that the empirical fit is good: a poor fit is a valid result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import multivariate_normal

from src.benchmark import call_price_closed_form
from src.calibration import (
    KAPPA_BOUNDS,
    binned_rmse,
    chronological_split,
    constant_kappa_values,
    effective_resolution_time,
    expected_abs_error,
    fit_constant_kappa,
    information_ratio_from_tau,
    prepare_observations,
    simulate_observations_from_model,
)
from src.empirical_analysis import path_statistics
from src.expiry_distribution import expiry_price_moments
from src.information_clock import linear_clock, posterior_from_full_path, simulate_clock_paths
from src.option_pricing import simulate_outcome_and_price_at_option_expiry


# ---- synthetic market data in the Part F format --------------------------------------

def make_markets(n_markets: int = 12, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly paths from the constant-kappa model (u_eff = 0.75), Part F column layout."""
    rng = np.random.default_rng(seed)
    u = np.arange(1, 40) / 39
    rows, sample = [], []
    for i in range(n_markets):
        p0 = float(rng.uniform(0.3, 0.6))
        tau = linear_clock(u, u[0], 0.75)
        grid = np.unique(np.r_[0.0, tau[tau < 1], 1.0])
        path = simulate_clock_paths(p0, 0.8, grid, 1, rng)
        S = np.interp(tau, grid, path.prediction_price[0])
        ticker = f"M{i:02d}"
        open_time = (pd.Timestamp("2026-08-01T14:00:00Z") + pd.Timedelta(days=i)).isoformat()
        sample.append({"ticker": ticker, "open_time": open_time, "excluded": False})
        rows.append(pd.DataFrame({"ticker": ticker, "normalized_time": u, "price": S, "Y": int(path.Y[0])}))
    # shuffle the sample order so the split must sort by time itself
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sample).sample(frac=1, random_state=1)


# ---- protocol ---------------------------------------------------------------------------------

def test_chronological_split_is_disjoint_ordered_and_75_25():
    _, sample = make_markets()
    train, test = chronological_split(sample, 0.75)
    assert len(train) == 9 and len(test) == 3
    assert not set(train) & set(test)
    t = sample.set_index("ticker")["open_time"]
    assert t.loc[train].max() < t.loc[test].min()


def test_effective_resolution_uses_training_markets_only():
    prices, sample = make_markets()
    train, test = chronological_split(sample)
    expected = path_statistics(prices[prices["ticker"].isin(train)])["learning_time"].median()
    assert effective_resolution_time(prices, train) == pytest.approx(expected)
    tampered = prices.copy()
    tampered.loc[tampered["ticker"].isin(test), "price"] = 0.5  # never learns
    assert effective_resolution_time(tampered, train) == effective_resolution_time(prices, train)


def test_test_markets_never_enter_the_fit():
    prices, sample = make_markets()
    train, test = chronological_split(sample)
    tampered = prices.copy()
    tampered.loc[tampered["ticker"].isin(test), "price"] = 1 - tampered.loc[tampered["ticker"].isin(test), "price"]
    a = fit_constant_kappa(prepare_observations(prices, train), 0.75)
    b = fit_constant_kappa(prepare_observations(tampered, train), 0.75)
    assert a == b
    assert set(prepare_observations(prices, train).ticker) == set(train)


def test_each_market_uses_its_own_initial_price():
    prices, sample = make_markets()
    obs = prepare_observations(prices, sample["ticker"].tolist())
    first = prices.sort_values("normalized_time").groupby("ticker")["price"].first()
    for ticker in first.index:
        assert np.all(obs.p0[obs.ticker == ticker] == first[ticker])
    assert len(np.unique(obs.p0)) > 1


def test_fit_is_deterministic_and_within_bounds():
    prices, sample = make_markets()
    train, _ = chronological_split(sample)
    obs = prepare_observations(prices, train)
    k1, r1 = fit_constant_kappa(obs, 0.75)
    k2, r2 = fit_constant_kappa(obs, 0.75)
    assert (k1, r1) == (k2, r2)
    assert KAPPA_BOUNDS[0] <= k1 <= KAPPA_BOUNDS[1] and k1 > 0 and r1 >= 0
    # the fit is a minimum of the stated objective
    for k in (0.5 * k1, 2 * k1):
        assert binned_rmse(obs, constant_kappa_values(obs, k, 0.75)) >= r1 - 1e-12


def test_bootstrap_simulation_is_reproducible_and_keeps_structure():
    prices, sample = make_markets()
    obs = prepare_observations(prices, sample["ticker"].tolist())
    a = simulate_observations_from_model(obs, 0.8, 0.75, np.random.default_rng(3))
    b = simulate_observations_from_model(obs, 0.8, 0.75, np.random.default_rng(3))
    assert np.array_equal(a.abs_error, b.abs_error)
    assert np.array_equal(a.u, obs.u) and np.array_equal(a.p0, obs.p0)
    assert np.all((a.abs_error >= 0) & (a.abs_error <= 1))


# ---- expected-absolute-error identity ---------------------------------------------------------

@pytest.mark.parametrize("p0, T_option, kappa", [(0.6, 2 / 3, 1.5), (0.3, 0.5, 0.48), (0.85, 0.9, 1.2)])
def test_expected_abs_error_matches_simulation_and_variance_identity(p0, T_option, kappa):
    s = kappa * np.sqrt(T_option / (1 - T_option))
    exact = float(expected_abs_error(p0, s))
    var = expiry_price_moments(p0, 1.0, T_option, kappa)["variance"]
    assert exact == pytest.approx(2 * (p0 * (1 - p0) - var), abs=1e-10)
    Y, S = simulate_outcome_and_price_at_option_expiry(p0, 1.0, T_option, kappa, 400_000, np.random.default_rng(7))
    err = np.abs(Y - S)
    assert abs(err.mean() - exact) < 4 * err.std(ddof=1) / np.sqrt(err.size)


def test_expected_abs_error_limits():
    p0 = np.array([0.2, 0.5, 0.9])
    assert np.allclose(expected_abs_error(p0, 0.0), 2 * p0 * (1 - p0))
    assert np.all(expected_abs_error(p0, np.inf) == 0.0)
    assert np.all(np.isinf(information_ratio_from_tau(1.0, [1.0, 1.2])))


# ---- information-time model ---------------------------------------------------------------------

def test_clock_paths_are_probabilities_and_resolve_to_outcome():
    tau = linear_clock(np.linspace(0.0, 0.8, 41), 0.0, 0.8)
    paths = simulate_clock_paths(0.4, 1.2, tau, 2000, np.random.default_rng(1))
    S = paths.prediction_price
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(S[:, 0] == 0.4)
    assert np.array_equal(S[:, -1], paths.Y.astype(float))


def test_closed_form_posterior_equals_full_path_bayes_on_nonuniform_clock():
    """S at the latest time uses only xi_latest; brute-force Bayes uses the whole path."""
    rng = np.random.default_rng(2)
    tau = np.r_[0.0, np.sort(rng.uniform(0.02, 0.97, 8)), 1.0]  # irregular information times
    p0, kappa = 0.35, 1.3
    paths = simulate_clock_paths(p0, kappa, tau, 25, rng)
    for i in range(25):
        S_closed = paths.prediction_price[i, -2]
        S_bayes = posterior_from_full_path(p0, kappa, tau[1:-1], paths.information_signal[i, 1:-1])
        assert S_closed == pytest.approx(S_bayes, abs=1e-9)


def test_naive_nonlinear_loading_with_calendar_bridge_is_not_markov():
    """Why the refinement must be a time change: with xi_t = g(t) Y + beta_tT and non-linear g,
    the latest xi alone does not give the full-path posterior (it does for linear g)."""
    t = np.array([0.2, 0.4, 0.6, 0.8])
    cov = np.minimum.outer(t, t) - np.outer(t, t)
    xi = np.array([0.05, -0.1, 0.3, 0.9])
    p0 = 0.5

    def posteriors(g):
        full = multivariate_normal.logpdf(xi, g, cov) - multivariate_normal.logpdf(xi, 0 * g, cov)
        marginal = (multivariate_normal.logpdf(xi[-1], g[-1], cov[-1, -1])
                    - multivariate_normal.logpdf(xi[-1], 0.0, cov[-1, -1]))
        return expit(logit(p0) + full), expit(logit(p0) + marginal)

    full_lin, last_lin = posteriors(1.5 * t)
    full_nl, last_nl = posteriors(1.5 * t**3)
    assert full_lin == pytest.approx(last_lin, abs=1e-12)
    assert abs(full_nl - last_nl) > 1e-3


# ---- pricing consistency ------------------------------------------------------------------------

def test_calendar_path_pricing_matches_information_time_closed_form():
    """Option expiring at calendar time u_opt: full calendar-grid simulation through the
    clock must reproduce the closed form evaluated at tau(u_opt)."""
    u_start, u_eff, kappa, p0, K = 0.0256, 0.769, 0.48, 0.6, 0.7
    u_grid = np.linspace(u_start, u_eff, 30)
    tau_grid = linear_clock(u_grid, u_start, u_eff)
    j = 19
    paths = simulate_clock_paths(p0, kappa, tau_grid, 200_000, np.random.default_rng(11))
    payoff = np.maximum(paths.prediction_price[:, j] - K, 0.0)
    exact = call_price_closed_form(p0, 1.0, float(tau_grid[j]), K, kappa)
    assert abs(payoff.mean() - exact) < 4 * payoff.std(ddof=1) / np.sqrt(payoff.size)
