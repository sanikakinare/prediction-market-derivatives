"""Information-based Bayesian model for a binary prediction-market contract.

Pure, deterministic functions only (no random numbers). Each function maps to
one equation of the model:

    Information process   xi_t = kappa * t * Y + beta_{tT}
    Bayesian exponent     A_t  = kappa*T/(T-t) * xi_t - kappa^2 * t*T / (2*(T-t))
    Market price          S_t  = Q(Y=1 | F_t) = sigmoid(logit(p0) + A_t),   t < T
                          S_T  = Y

Throughout, r = 0 so no discounting appears anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def brownian_bridge_variance(t: np.ndarray | float, T: float) -> np.ndarray | float:
    """Var(beta_{tT}) = t (T - t) / T for a Brownian bridge pinned to 0 at 0 and T."""
    return t * (T - t) / T


def information_signal(
    t: np.ndarray | float,
    Y: np.ndarray | int,
    beta: np.ndarray | float,
    kappa: float,
) -> np.ndarray:
    """xi_t = kappa * t * Y + beta_{tT}.

    The first term is the "true signal" (it grows linearly at rate kappa only if
    the event occurs); the bridge is noise that vanishes at t = T, so
    xi_T = kappa * T * Y reveals Y exactly.
    """
    return kappa * t * Y + beta


def bayesian_exponent(
    t: np.ndarray | float,
    xi: np.ndarray | float,
    kappa: float,
    T: float,
) -> np.ndarray:
    """A_t = kappa*T/(T-t) * xi_t - kappa^2 * t * T / (2 (T-t)),  valid for t < T only.

    A_t is the log-likelihood ratio of observing xi_t under Y=1 versus Y=0, i.e.
    the amount by which the observed information shifts the log-odds of YES.
    """
    t = np.asarray(t, dtype=float)
    if np.any(t >= T):
        raise ValueError("A_t is only defined for t < T; at t = T set S_T = Y directly.")
    return kappa * T / (T - t) * xi - kappa**2 * t * T / (2.0 * (T - t))


def signal_to_price(
    t: np.ndarray | float,
    xi: np.ndarray | float,
    p0: float,
    kappa: float,
    T: float,
) -> np.ndarray:
    """S_t = Q(Y=1 | F_t) for t < T, computed in log-odds space.

    Uses  logit(S_t) = logit(p0) + A_t  and  S_t = expit(logit(S_t)).
    scipy's expit is numerically stable: it never overflows for large |A_t| and
    saturates cleanly at 0.0 / 1.0, which is algebraically identical to
    p0 e^A / ((1-p0) + p0 e^A) but avoids computing e^A directly.
    """
    _check_p0(p0)
    A_t = bayesian_exponent(t, xi, kappa, T)
    return expit(logit(p0) + A_t)


def _check_p0(p0: float) -> None:
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie strictly in (0, 1); got {p0}.")
