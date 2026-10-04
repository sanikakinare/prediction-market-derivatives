"""Comparative statics of the European call premium (Part D).

Two pricing routes are kept deliberately separate:

* ``closed_form_sensitivity`` evaluates the exact benchmark
  ``benchmark.call_price_closed_form`` on a dense grid (the smooth curves);
* ``mc_sensitivity_points`` runs the general Monte Carlo engine
  ``option_pricing.price_european_call_mc`` at a few representative points and
  reports how far each estimate is from the benchmark in standard errors.

Two derived quantities help interpret the results; neither changes the model.

Information ratio
    With sigma^2 = Var(beta_{tT}) = t (T - t) / T at t = T_option, the expiry
    signal is xi_t = kappa t Y + sigma Z. Its signal-to-noise ratio is

        s = kappa t / sigma = kappa * sqrt(T_option * T / (T - T_option)).

    Substituting into the closed form gives, with L = logit K - logit p0,

        C_0 = p0 (1 - K) Phi(s/2 - L/s) - (1 - p0) K Phi(-s/2 - L/s),

    so kappa, T_option and T affect the price only through s.

Information "vega"
    Using p0 (1 - K) phi(d1) = (1 - p0) K phi(d0) (which follows from
    d0^2 - d1^2 = 2L), differentiating gives dC_0/ds = p0 (1 - K) phi(d1) > 0
    for 0 < K < 1.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm

from .benchmark import call_price_closed_form
from .option_pricing import price_european_call_mc

BASELINE: dict[str, float] = {
    "p0": 0.60,
    "T": 1.0,
    "T_option": 2 / 3,
    "K": 0.70,
    "kappa": 1.5,
}

SENSITIVITY_PARAMETERS = ("K", "T_option", "p0", "kappa")


def _parameters_with(name: str, value: float, baseline: dict[str, float]) -> dict[str, float]:
    """Baseline parameters with exactly one entry replaced."""
    if name not in SENSITIVITY_PARAMETERS:
        raise ValueError(f"Unknown sensitivity parameter {name!r}; expected one of {SENSITIVITY_PARAMETERS}.")
    params = dict(baseline)
    params[name] = float(value)
    return params


def information_ratio(T: float, T_option: float, kappa: float) -> float:
    """s = kappa * sqrt(T_option * T / (T - T_option)): signal-to-noise of xi at T_option."""
    return kappa * np.sqrt(T_option * T / (T - T_option))


def call_price_information_form(p0: float, K: float, s: float) -> float:
    """C_0 written as a function of the information ratio s only (0 < K < 1)."""
    L = logit(K) - logit(p0)
    return float(p0 * (1 - K) * norm.cdf(s / 2 - L / s) - (1 - p0) * K * norm.cdf(-s / 2 - L / s))


def call_information_vega(p0: float, K: float, s: float) -> float:
    """dC_0/ds = p0 (1 - K) phi(d1), with d1 = s/2 - L/s (0 < K < 1)."""
    L = logit(K) - logit(p0)
    return float(p0 * (1 - K) * norm.pdf(s / 2 - L / s))


def static_bounds(p0: float, K: float) -> tuple[float, float]:
    """Model-implied bounds (p0 - K)^+ <= C_0 <= p0 (1 - K).

    Lower: Jensen, since E[S_{T_option}] = p0 and (x - K)^+ is convex.
    Upper: S_{T_option} = E[Y | F_{T_option}], so again by Jensen
    (S - K)^+ <= E[(Y - K)^+ | F], whose expectation is p0 (1 - K) for 0 <= K <= 1.
    These are the no-information and full-information limits respectively.
    """
    return max(p0 - K, 0.0), p0 * (1.0 - K)


def closed_form_sensitivity(
    name: str,
    values: Iterable[float],
    baseline: dict[str, float] = BASELINE,
) -> pd.DataFrame:
    """Exact call premium on a grid of one parameter, all others at baseline."""
    base_price = call_price_closed_form(**baseline)
    rows = []
    for value in values:
        params = _parameters_with(name, value, baseline)
        price = call_price_closed_form(**params)
        lower, upper = static_bounds(params["p0"], params["K"])
        rows.append({
            name: float(value),
            "call_price": price,
            "change_vs_baseline": price - base_price,
            "lower_bound": lower,
            "upper_bound": upper,
            "information_ratio": information_ratio(params["T"], params["T_option"], params["kappa"]),
        })
    return pd.DataFrame(rows)


def mc_sensitivity_points(
    name: str,
    values: Iterable[float],
    n_simulations: int,
    seed_sequence: np.random.SeedSequence,
    baseline: dict[str, float] = BASELINE,
) -> pd.DataFrame:
    """Monte Carlo prices at selected points, each from its own independent stream.

    The benchmark is attached only for comparison: z = (MC - exact) / s.e.
    """
    values = list(values)
    streams = seed_sequence.spawn(len(values))
    rows = []
    for value, stream in zip(values, streams):
        params = _parameters_with(name, value, baseline)
        res = price_european_call_mc(
            n_simulations=n_simulations, rng=np.random.default_rng(stream), **params
        )
        exact = call_price_closed_form(**params)
        rows.append({
            "parameter": name,
            "value": float(value),
            "mc_price": res.option_price,
            "standard_error": res.standard_error,
            "ci_lower": res.ci_lower,
            "ci_upper": res.ci_upper,
            "closed_form": exact,
            "z": (res.option_price - exact) / res.standard_error,
            "closed_form_in_ci": res.ci_lower <= exact <= res.ci_upper,
        })
    return pd.DataFrame(rows)
