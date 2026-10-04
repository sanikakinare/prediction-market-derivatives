"""Tests for Part E: the exact expiry-price distribution and its agreement with MC.

Exact-distribution properties are tested deterministically; Monte Carlo is only
required to agree within statistical tolerance (no histogram-bin comparisons).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from src.benchmark import call_price_closed_form, exercise_probabilities
from src.expiry_distribution import (
    expiry_price_bin_probabilities,
    expiry_price_cdf,
    expiry_price_moments,
    expiry_price_pdf,
    expiry_price_quantile,
    information_ratio,
    premium_contribution_by_bin,
)
from src.option_pricing import (
    call_payoff,
    simulate_outcome_and_price_at_option_expiry,
    simulate_price_at_option_expiry,
)

P0, T, T_OPTION, K, KAPPA = 0.60, 1.0, 2 / 3, 0.70, 1.5

PARAM_SETS = [  # (p0, T_option, kappa)
    (0.60, 2 / 3, 0.5), (0.60, 2 / 3, 1.5), (0.60, 2 / 3, 3.0), (0.60, 0.10, 1.5), (0.60, 0.90, 1.5),
    (0.20, 0.50, 2.0), (0.90, 0.30, 0.8), (0.50, 2 / 3, 10.0),
]


# ---- exact distribution: internal consistency --------------------------------

@pytest.mark.parametrize("p0, T_option, kappa", PARAM_SETS)
def test_exact_mean_is_p0(p0, T_option, kappa):
    assert expiry_price_moments(p0, T, T_option, kappa)["mean"] == pytest.approx(p0, abs=1e-10)


@pytest.mark.parametrize("p0, T_option, kappa", PARAM_SETS)
def test_cdf_is_a_mixture_of_conditionals_and_bins_sum_to_one(p0, T_option, kappa):
    x = np.linspace(0.01, 0.99, 50)
    F = expiry_price_cdf(x, p0, T, T_option, kappa)
    F1 = expiry_price_cdf(x, p0, T, T_option, kappa, outcome=1)
    F0 = expiry_price_cdf(x, p0, T, T_option, kappa, outcome=0)
    assert np.allclose(F, p0 * F1 + (1 - p0) * F0)
    assert np.all(np.diff(F) >= 0)
    assert np.all(F1 <= F0 + 1e-15)  # YES worlds put more mass on high prices
    edges = np.linspace(0, 1, 51)
    assert expiry_price_bin_probabilities(edges, p0, T, T_option, kappa).sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("p0, T_option, kappa", PARAM_SETS[:6])
def test_pdf_matches_derivative_of_cdf(p0, T_option, kappa):
    x = np.linspace(0.05, 0.95, 19)
    h = 1e-6
    fd = (expiry_price_cdf(x + h, p0, T, T_option, kappa) - expiry_price_cdf(x - h, p0, T, T_option, kappa)) / (2 * h)
    assert np.allclose(expiry_price_pdf(x, p0, T, T_option, kappa), fd, rtol=1e-5)


def test_prices_are_calibrated():
    """p0 f(x|Y=1) / ((1-p0) f(x|Y=0)) = x / (1-x): a price of x means a fraction x resolve YES."""
    x = np.linspace(0.02, 0.98, 25)
    f1 = expiry_price_pdf(x, P0, T, T_OPTION, KAPPA, outcome=1)
    f0 = expiry_price_pdf(x, P0, T, T_OPTION, KAPPA, outcome=0)
    assert np.allclose(P0 * f1 / ((1 - P0) * f0), x / (1 - x), rtol=1e-10)


@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_quantile_inverts_cdf(q):
    x = expiry_price_quantile(q, P0, T, T_OPTION, KAPPA)
    assert expiry_price_cdf(x, P0, T, T_OPTION, KAPPA) == pytest.approx(q, abs=1e-9)


# ---- exact distribution vs the Part C benchmark ------------------------------

@pytest.mark.parametrize("p0, T_option, kappa", PARAM_SETS)
@pytest.mark.parametrize("K_", [0.3, 0.7])
def test_exercise_probabilities_match_benchmark_thresholds(p0, T_option, kappa, K_):
    ex = exercise_probabilities(p0, T, T_option, K_, kappa)
    assert 1 - expiry_price_cdf(K_, p0, T, T_option, kappa) == pytest.approx(ex["total"], abs=1e-12)
    assert 1 - expiry_price_cdf(K_, p0, T, T_option, kappa, outcome=1) == pytest.approx(ex["given_yes"], abs=1e-12)
    assert 1 - expiry_price_cdf(K_, p0, T, T_option, kappa, outcome=0) == pytest.approx(ex["given_no"], abs=1e-12)


@pytest.mark.parametrize("p0, T_option, kappa", PARAM_SETS)
def test_premium_contributions_sum_to_closed_form(p0, T_option, kappa):
    edges = np.linspace(K, 1.0, 16)
    total = premium_contribution_by_bin(edges, K, p0, T, T_option, kappa).sum()
    assert total == pytest.approx(call_price_closed_form(p0, T, T_option, K, kappa), abs=1e-12)


# ---- dispersion ----------------------------------------------------------------

def test_variance_increases_with_kappa_and_expiry_and_is_bounded():
    v_kappa = [expiry_price_moments(P0, T, T_OPTION, k)["variance"] for k in (0.25, 0.5, 1.5, 3.0, 6.0)]
    v_expiry = [expiry_price_moments(P0, T, t, KAPPA)["variance"] for t in (0.05, 0.1, 2 / 3, 0.9, 0.99)]
    for v in (v_kappa, v_expiry):
        assert np.all(np.diff(v) > 0)
        assert 0 < min(v) and max(v) < P0 * (1 - P0)


def test_variance_depends_only_on_information_ratio():
    s = information_ratio(T, T_OPTION, KAPPA)
    T_other = 8 / 9
    kappa_other = s / np.sqrt(T_other * T / (T - T_other))
    a = expiry_price_moments(P0, T, T_OPTION, KAPPA)["variance"]
    b = expiry_price_moments(P0, T, T_other, kappa_other)["variance"]
    assert a == pytest.approx(b, abs=1e-12)


def test_premium_rises_with_information_even_when_exercise_probability_falls():
    """K < p0: more information lowers Q(S > K) towards p0 but still raises the premium."""
    kappas = [0.25, 0.5, 1.5, 3.0, 6.0]
    probs = [exercise_probabilities(P0, T, T_OPTION, 0.5, k)["total"] for k in kappas]
    prices = [call_price_closed_form(P0, T, T_OPTION, 0.5, k) for k in kappas]
    assert np.all(np.diff(probs) < 0)
    assert np.all(np.diff(prices) > 0)


# ---- Monte Carlo agreement (statistical tolerance) ----------------------------

@pytest.mark.parametrize("p0, T_option, kappa", PARAM_SETS[:5])
def test_mc_expiry_sample_matches_exact(p0, T_option, kappa):
    rng = np.random.default_rng(np.random.SeedSequence([99, int(1000 * T_option), int(100 * kappa)]))
    Y, S = simulate_outcome_and_price_at_option_expiry(p0, T, T_option, kappa, 200_000, rng)
    n = S.size
    exact = expiry_price_moments(p0, T, T_option, kappa)
    ex = exercise_probabilities(p0, T, T_option, K, kappa)
    C0 = call_price_closed_form(p0, T, T_option, K, kappa)

    assert set(np.unique(Y)) <= {0, 1}
    assert abs(S.mean() - p0) < 4 * S.std(ddof=1) / np.sqrt(n)
    var = S.var(ddof=1)
    var_se = np.sqrt((np.mean((S - S.mean()) ** 4) - var**2) / n)
    assert abs(var - exact["variance"]) < 4 * var_se
    prob = np.mean(S > K)
    assert abs(prob - ex["total"]) < 4 * np.sqrt(ex["total"] * (1 - ex["total"]) / n)
    H = call_payoff(S, K)
    assert abs(H.mean() - C0) < 4 * H.std(ddof=1) / np.sqrt(n)
    assert np.mean(H == 0) == pytest.approx(1 - prob)
    assert kstest(S, lambda x: expiry_price_cdf(x, p0, T, T_option, kappa)).pvalue > 1e-4


def test_sampler_refactor_preserves_draws():
    a = simulate_price_at_option_expiry(P0, T, T_OPTION, KAPPA, 1000, np.random.default_rng(5))
    _, b = simulate_outcome_and_price_at_option_expiry(P0, T, T_OPTION, KAPPA, 1000, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_exact_distribution_rejects_bad_inputs():
    with pytest.raises(ValueError):
        expiry_price_cdf(0.5, 1.0, T, T_OPTION, KAPPA)
    with pytest.raises(ValueError):
        expiry_price_moments(P0, T, 1.0, KAPPA)
    with pytest.raises(ValueError):
        expiry_price_cdf(0.5, P0, T, T_OPTION, KAPPA, outcome=2)
