"""Independent (non-Monte-Carlo) benchmark prices for the European call.

This module intentionally imports nothing from the rest of ``src`` and re-derives
everything it needs from the model definition, so that it can serve as an
independent check on the Monte Carlo implementation.

Derivation (exercise-threshold representation)
----------------------------------------------
Write t = T_option and sigma^2 = Var(beta_{tT}) = t (T - t) / T. Given Y,

    xi_t | Y=1 ~ N(kappa t, sigma^2),        xi_t | Y=0 ~ N(0, sigma^2).

S_t is strictly increasing in xi_t (the coefficient kappa T / (T - t) > 0), so for
0 < K < 1 the call is exercised exactly when xi_t exceeds a threshold xi*:

    S_t > K  <=>  logit(p0) + A_t > logit(K)
             <=>  xi_t > xi* = (T - t) / (kappa T) * (logit K - logit p0) + kappa t / 2.

Then C_0 = E[(S_t - K) 1{xi_t > xi*}] = E[S_t 1{xi_t > xi*}] - K Q(xi_t > xi*).
Because S_t = Q(Y=1 | F_t) and {xi_t > xi*} is F_t-measurable, the tower property
gives E[S_t 1{xi_t > xi*}] = Q(Y=1, xi_t > xi*). With

    d1 = (kappa t - xi*) / sigma,      d0 = -xi* / sigma,

    Q(Y=1, xi_t > xi*) = p0 Phi(d1),   Q(Y=0, xi_t > xi*) = (1 - p0) Phi(d0),

    C_0 = p0 (1 - K) Phi(d1) - (1 - p0) K Phi(d0).

Interpretation: a YES-world exercise pays (1 - K) "on average" because S_t is the
posterior probability of YES; a NO-world exercise costs K.

A second, more brute-force benchmark integrates the payoff numerically against
the two conditional Gaussian densities. It does not use the tower-property step,
so agreement between the two cross-checks that step as well.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import expit, logit
from scipy.stats import norm


def _validate(p0: float, T: float, T_option: float, K: float, kappa: float) -> None:
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie strictly in (0, 1); got {p0}.")
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive; got {T}.")
    if not (np.isfinite(kappa) and kappa > 0.0):
        raise ValueError(f"kappa must be positive; got {kappa}.")
    if not (np.isfinite(T_option) and 0.0 < T_option < T):
        raise ValueError(f"T_option must satisfy 0 < T_option < T; got {T_option}.")
    if not np.isfinite(K):
        raise ValueError(f"K must be finite; got {K}.")


def exercise_threshold(p0: float, T: float, T_option: float, K: float, kappa: float) -> float:
    """xi* such that S_{T_option} > K  <=>  xi_{T_option} > xi*  (requires 0 < K < 1)."""
    t = T_option
    return (T - t) / (kappa * T) * (logit(K) - logit(p0)) + kappa * t / 2.0


def call_price_closed_form(p0: float, T: float, T_option: float, K: float, kappa: float) -> float:
    """C_0 = p0 (1 - K) Phi(d1) - (1 - p0) K Phi(d0)."""
    _validate(p0, T, T_option, K, kappa)
    if K <= 0.0:   # always exercised: E[S_t - K] = p0 - K by the martingale property
        return p0 - K
    if K >= 1.0:   # S_t < 1 for t < T, so never exercised
        return 0.0
    t = T_option
    sigma = np.sqrt(t * (T - t) / T)
    xi_star = exercise_threshold(p0, T, T_option, K, kappa)
    d1 = (kappa * t - xi_star) / sigma
    d0 = -xi_star / sigma
    return float(p0 * (1.0 - K) * norm.cdf(d1) - (1.0 - p0) * K * norm.cdf(d0))


def call_price_quadrature(p0: float, T: float, T_option: float, K: float, kappa: float) -> float:
    """C_0 = p0 * E[H | Y=1] + (1 - p0) * E[H | Y=0], each a 1-D Gaussian integral.

    Integrates over xi from the exercise threshold upwards (so the integrand is
    smooth, without the kink at S = K). Price written as
    S(xi) = p0 e^A / ((1 - p0) + p0 e^A), evaluated via a stable sigmoid.
    """
    _validate(p0, T, T_option, K, kappa)
    if K >= 1.0:
        return 0.0
    t = T_option
    sigma = np.sqrt(t * (T - t) / T)

    def price(xi: float) -> float:
        A = kappa * T / (T - t) * xi - kappa**2 * t * T / (2.0 * (T - t))
        return float(expit(np.log(p0) - np.log(1.0 - p0) + A))

    lower = -np.inf if K <= 0.0 else exercise_threshold(p0, T, T_option, K, kappa)

    def conditional_expectation(mean: float) -> float:
        integrand = lambda xi: (price(xi) - K) * norm.pdf(xi, loc=mean, scale=sigma)
        value, _ = integrate.quad(integrand, lower, np.inf, epsabs=1e-13, epsrel=1e-11, limit=200)
        return value

    return p0 * conditional_expectation(kappa * t) + (1.0 - p0) * conditional_expectation(0.0)
