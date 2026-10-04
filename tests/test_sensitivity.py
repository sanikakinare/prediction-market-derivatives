"""Tests for Part D: comparative statics of the call premium.

Monotonicity and limits are tested on the exact closed form (robust); Monte
Carlo is only required to agree with it within statistical tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark import call_price_closed_form
from src.sensitivity import (
    BASELINE,
    call_information_vega,
    call_price_information_form,
    closed_form_sensitivity,
    information_ratio,
    mc_sensitivity_points,
    static_bounds,
)

P0, K = BASELINE["p0"], BASELINE["K"]


def price_with(**kw):
    params = dict(BASELINE)
    params.update(kw)
    return call_price_closed_form(**params)


# ---- monotonicity on the exact price ------------------------------------------

def test_price_strictly_decreasing_in_strike():
    df = closed_form_sensitivity("K", np.linspace(0.01, 0.99, 197))
    assert np.all(np.diff(df["call_price"]) < 0)


@pytest.mark.parametrize("name, grid", [
    ("T_option", np.linspace(0.01, 0.99, 197)),
    ("kappa", np.linspace(0.05, 6.0, 200)),
    ("p0", np.linspace(0.01, 0.99, 197)),
])
def test_price_increasing(name, grid):
    df = closed_form_sensitivity(name, grid)
    assert np.all(np.diff(df["call_price"]) >= -1e-14)


@pytest.mark.parametrize("K_", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("p0", [0.1, 0.5, 0.9])
def test_price_nondecreasing_in_expiry_and_kappa_for_other_strikes(p0, K_):
    T_grid = [price_with(p0=p0, K=K_, T_option=t) for t in np.linspace(0.01, 0.99, 99)]
    k_grid = [price_with(p0=p0, K=K_, kappa=k) for k in np.linspace(0.05, 6.0, 100)]
    assert np.all(np.diff(T_grid) >= -1e-14)
    assert np.all(np.diff(k_grid) >= -1e-14)


# ---- bounds and limits ---------------------------------------------------------

def test_price_within_static_bounds_and_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p0, K_ = rng.uniform(0.01, 0.99, 2)
        t = rng.uniform(0.01, 0.99)
        kappa = rng.uniform(0.05, 6.0)
        c = call_price_closed_form(p0, 1.0, t, K_, kappa)
        lower, upper = static_bounds(p0, K_)
        assert c >= 0.0
        assert lower - 1e-12 <= c <= upper + 1e-12


def test_strike_limits():
    assert price_with(K=1e-6) == pytest.approx(P0, abs=1e-5)       # ~ owning the contract
    assert price_with(K=1 - 1e-6) == pytest.approx(0.0, abs=1e-6)


def test_expiry_limits():
    assert price_with(T_option=1e-6) == pytest.approx(max(P0 - K, 0.0), abs=1e-6)
    assert price_with(T_option=1e-6, K=0.5) == pytest.approx(P0 - 0.5, abs=1e-4)
    assert price_with(T_option=1 - 1e-6) == pytest.approx(P0 * (1 - K), abs=1e-6)


def test_kappa_limits():
    assert price_with(kappa=0.01) == pytest.approx(max(P0 - K, 0.0), abs=1e-6)
    assert price_with(kappa=0.01, K=0.5) == pytest.approx(P0 - 0.5, abs=1e-6)
    assert price_with(kappa=20.0) == pytest.approx(P0 * (1 - K), abs=1e-9)


# ---- information-ratio representation -----------------------------------------

def test_price_depends_on_kappa_and_expiry_only_through_s():
    rng = np.random.default_rng(1)
    for _ in range(500):
        p0, K_ = rng.uniform(0.02, 0.98, 2)
        t = rng.uniform(0.01, 0.99)
        kappa = rng.uniform(0.05, 5.0)
        s = information_ratio(1.0, t, kappa)
        assert call_price_closed_form(p0, 1.0, t, K_, kappa) == pytest.approx(
            call_price_information_form(p0, K_, s), abs=1e-12)


def test_information_vega_positive_and_matches_finite_difference():
    for s in (0.3, 1.0, 2.12, 4.0):
        h = 1e-6
        fd = (call_price_information_form(P0, K, s + h) - call_price_information_form(P0, K, s - h)) / (2 * h)
        vega = call_information_vega(P0, K, s)
        assert vega > 0
        assert fd == pytest.approx(vega, rel=1e-6)


# ---- Monte Carlo agreement and setup ----------------------------------------------

@pytest.mark.parametrize("name, values", [
    ("K", [0.3, 0.9]), ("T_option", [0.1, 0.9]), ("p0", [0.2, 0.8]), ("kappa", [0.5, 3.0]),
])
def test_mc_consistent_with_closed_form(name, values):
    df = mc_sensitivity_points(name, values, 200_000, np.random.SeedSequence(2024))
    assert np.all(df["mc_price"] >= 0)
    assert np.all(np.abs(df["z"]) < 4.0)


def test_non_varied_parameters_stay_at_baseline():
    df = closed_form_sensitivity("kappa", [BASELINE["kappa"]])
    assert df["call_price"].iloc[0] == call_price_closed_form(**BASELINE)
    assert df["change_vs_baseline"].iloc[0] == 0.0


def test_unknown_parameter_raises():
    with pytest.raises(ValueError):
        closed_form_sensitivity("sigma", [0.1])
