"""Information-based Bayesian model for a binary prediction-market contract.

Pure, deterministic functions only (no random numbers). Each function maps to
one equation of the model:

    Information process   xi_t = kappa * t * Y + beta_{tT}
    Bayesian exponent     A_t  = kappa*T/(T-t) * xi_t - kappa^2 * t*T / (2*(T-t))
    Market price          S_t  = Q(Y=1 | F_t) = sigmoid(logit(p0) + A_t),   t < T
                          S_T  = Y

Throughout, r = 0 so no discounting appears anywhere.

Floating-point note: S_t is mathematically strictly inside (0, 1) for t < T, but
once |logit(p0) + A_t| exceeds roughly 37 (towards 1) or 745 (towards 0), the
float64 sigmoid returns exactly 1.0 or 0.0. Exact 0/1 prices before T are
therefore numerical saturation, not early resolution of the contract; the
model's resolution happens only at T, where S_T = Y is set explicitly.
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

    For strongly informative signals the result can saturate to exactly 0.0 or
    1.0 in float64 (see module docstring); this is rounding, not resolution.
    """
    check_model_parameters(p0, T, kappa)
    A_t = bayesian_exponent(t, xi, kappa, T)
    return expit(logit(p0) + A_t)


def check_model_parameters(p0: float, T: float, kappa: float) -> None:
    """Validate the model parameters: p0 in (0, 1), T > 0, kappa > 0, all finite.

    p0 = 0 or 1 would make logit(p0) infinite (the event is already certain), and
    kappa <= 0 would make the information signal uninformative or reversed.
    """
    for name, value in (("p0", p0), ("T", T), ("kappa", kappa)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value}.")
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie strictly in (0, 1); got {p0}.")
    if T <= 0.0:
        raise ValueError(f"T must be positive; got {T}.")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive; got {kappa}.")


def check_positive_int(name: str, value: int) -> None:
    """Validate a count such as n_steps or n_paths (a positive integer)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
