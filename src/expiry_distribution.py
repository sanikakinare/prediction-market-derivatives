"""Exact distribution of the expiry price S_{T_option} implied by the baseline model.

Like ``benchmark.py``, this module imports nothing from the rest of ``src`` so
that it can independently validate the Monte Carlo samples.

Derivation
----------
At t = T_option the signal is xi_t = kappa t Y + sigma Z with Z ~ N(0, 1) and
sigma^2 = t (T - t) / T. Define the information ratio

    s = kappa t / sigma = kappa * sqrt(T_option * T / (T - T_option)).

Substituting xi_t / sigma = s Y + Z into A_t gives A_t = s (s Y + Z) - s^2 / 2, so

    logit S = logit p0 + s Z + s^2 / 2    if Y = 1,
    logit S = logit p0 + s Z - s^2 / 2    if Y = 0.

S is strictly increasing in Z, so for x in (0, 1), with L(x) = logit x - logit p0,

    Q(S <= x | Y=1) = Phi(L/s - s/2),     Q(S <= x | Y=0) = Phi(L/s + s/2),
    Q(S <= x)       = p0 Phi(L/s - s/2) + (1 - p0) Phi(L/s + s/2).

Differentiating (dL/dx = 1 / (x (1 - x))) gives the densities

    f(x | Y=1) = phi(L/s - s/2) / (s x (1 - x)),
    f(x | Y=0) = phi(L/s + s/2) / (s x (1 - x)).

They satisfy p0 f(x|Y=1) / ((1 - p0) f(x|Y=0)) = x / (1 - x): among the worlds in
which the market price is x, a fraction x resolve YES. The price is "calibrated"
under Q, which is what S_t = Q(Y=1 | F_t) means.

Numerical note: the density is evaluated in log space and is exact, but for large
s it has tall, integrable spikes near 0 and 1 (e.g. f(1e-12) ~ 1.6e6 when
kappa = 3), so naive plotting or quadrature of f on [0, 1] is unreliable there.
Bin probabilities and bin contributions to the premium are therefore computed
from CDF differences (``expiry_price_bin_probabilities``,
``premium_contribution_by_bin``), which are stable for every s.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from scipy.special import expit, logit
from scipy.stats import norm


def _validate(p0: float, T: float, T_option: float, kappa: float) -> None:
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie strictly in (0, 1); got {p0}.")
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive; got {T}.")
    if not (np.isfinite(kappa) and kappa > 0.0):
        raise ValueError(f"kappa must be positive; got {kappa}.")
    if not (np.isfinite(T_option) and 0.0 < T_option < T):
        raise ValueError(f"T_option must satisfy 0 < T_option < T; got {T_option}.")


def information_ratio(T: float, T_option: float, kappa: float) -> float:
    """s = kappa * sqrt(T_option * T / (T - T_option))."""
    return kappa * np.sqrt(T_option * T / (T - T_option))


def _standardised_arguments(x, p0: float, s: float):
    """(a1, a0) = (L/s - s/2, L/s + s/2) with L = logit x - logit p0."""
    L = logit(np.asarray(x, dtype=float)) - logit(p0)
    return L / s - s / 2.0, L / s + s / 2.0


def expiry_price_cdf(x, p0: float, T: float, T_option: float, kappa: float, outcome: int | None = None):
    """Q(S_{T_option} <= x), optionally conditional on Y = outcome (1 or 0)."""
    _validate(p0, T, T_option, kappa)
    s = information_ratio(T, T_option, kappa)
    x = np.asarray(x, dtype=float)
    a1, a0 = _standardised_arguments(np.clip(x, 1e-300, 1 - 1e-16), p0, s)
    if outcome == 1:
        F = norm.cdf(a1)
    elif outcome == 0:
        F = norm.cdf(a0)
    elif outcome is None:
        F = p0 * norm.cdf(a1) + (1 - p0) * norm.cdf(a0)
    else:
        raise ValueError("outcome must be None, 0 or 1.")
    return np.where(x <= 0.0, 0.0, np.where(x >= 1.0, 1.0, F))


def expiry_price_pdf(x, p0: float, T: float, T_option: float, kappa: float, outcome: int | None = None):
    """Density of S_{T_option} on (0, 1), optionally conditional on Y = outcome."""
    _validate(p0, T, T_option, kappa)
    s = information_ratio(T, T_option, kappa)
    x = np.asarray(x, dtype=float)
    inside = (x > 0.0) & (x < 1.0)
    xs = np.where(inside, x, 0.5)
    a1, a0 = _standardised_arguments(xs, p0, s)
    log_jacobian = -np.log(s) - np.log(xs) - np.log1p(-xs)
    if outcome == 1:
        log_f = norm.logpdf(a1)
    elif outcome == 0:
        log_f = norm.logpdf(a0)
    elif outcome is None:
        log_f = np.logaddexp(np.log(p0) + norm.logpdf(a1), np.log(1 - p0) + norm.logpdf(a0))
    else:
        raise ValueError("outcome must be None, 0 or 1.")
    return np.where(inside, np.exp(log_f + log_jacobian), 0.0)


def expiry_price_quantile(q: float, p0: float, T: float, T_option: float, kappa: float) -> float:
    """Inverse of the unconditional CDF, solved in log-odds space for stability."""
    _validate(p0, T, T_option, kappa)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie in (0, 1); got {q}.")
    s = information_ratio(T, T_option, kappa)

    def cdf_minus_q(L: float) -> float:
        return p0 * norm.cdf(L / s - s / 2) + (1 - p0) * norm.cdf(L / s + s / 2) - q

    span = 40.0 + 2.0 * s * s
    L = optimize.brentq(cdf_minus_q, -span, span, xtol=1e-12)
    return float(expit(logit(p0) + L))


def expiry_price_moments(p0: float, T: float, T_option: float, kappa: float) -> dict[str, float]:
    """Exact E[S], Var(S) and E[S | Y] by 1-D Gaussian quadrature in Z.

    E[S | Y=y] = E[expit(logit p0 + s Z +/- s^2/2)]. The unconditional mean
    p0 E[S|Y=1] + (1-p0) E[S|Y=0] should equal p0 (martingale property), and
    since E[S^2] = E[S Y] = p0 E[S | Y=1],

        Var(S) = p0 E[S | Y=1] - p0^2,

    which lies between 0 (no information) and p0 (1 - p0) (outcome revealed).
    """
    _validate(p0, T, T_option, kappa)
    s = information_ratio(T, T_option, kappa)

    def conditional_mean(sign: float) -> float:
        value, _ = integrate.quad(
            lambda z: expit(logit(p0) + s * z + sign * s * s / 2) * norm.pdf(z),
            -40.0, 40.0, epsabs=1e-14, epsrel=1e-12, limit=200,
        )
        return value

    mean_yes, mean_no = conditional_mean(+1.0), conditional_mean(-1.0)
    variance = p0 * mean_yes - p0**2
    return {
        "information_ratio": s,
        "mean": p0 * mean_yes + (1 - p0) * mean_no,
        "variance": variance,
        "variance_fraction": variance / (p0 * (1 - p0)),
        "mean_given_yes": mean_yes,
        "mean_given_no": mean_no,
    }


def expiry_price_bin_probabilities(
    edges: np.ndarray, p0: float, T: float, T_option: float, kappa: float, outcome: int | None = None
) -> np.ndarray:
    """Exact Q(S in [edges[i], edges[i+1]]) from CDF differences (stable for any s)."""
    return np.diff(expiry_price_cdf(np.asarray(edges, dtype=float), p0, T, T_option, kappa, outcome))


def premium_contribution_by_bin(
    edges: np.ndarray, K: float, p0: float, T: float, T_option: float, kappa: float
) -> np.ndarray:
    """Exact E[(S - K) 1{S in bin}] for bins lying in [K, 1].

    Each bin event is known at T_option, so by the tower property
    E[S 1{S in bin}] = Q(Y=1, S in bin) = p0 * Q(S in bin | Y=1). Hence

        contribution = p0 * dF(. | Y=1) - K * dF(.),

    and the contributions over all bins above K sum to the call premium C_0.
    """
    edges = np.asarray(edges, dtype=float)
    if np.any(edges < K - 1e-15):
        raise ValueError("premium_contribution_by_bin expects bins at or above the strike K.")
    joint_yes = p0 * expiry_price_bin_probabilities(edges, p0, T, T_option, kappa, outcome=1)
    total = expiry_price_bin_probabilities(edges, p0, T, T_option, kappa)
    return joint_yes - K * total
