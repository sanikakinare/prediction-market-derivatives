"""Calibration of the information model to real price paths (Part G).

Fit target: the mean forecast error |Y - S_t| in normalized-time bins (Part F).
Model counterpart, for a market starting at p0 with information ratio s:

    E|Y - S_t| = 2 E[S_t (1 - S_t)] = 2 [p0 (1 - p0) - Var(S_t)] = 2 p0 E[1 - S_t | Y = 1],

the last form because E[S (1 - Y)] = E[(1 - S) Y] (both equal p0 - E[S^2]).
Here s = kappa sqrt(tau / (1 - tau)) with information time tau in [0, 1]
(T = 1 in information time), and E|Y - S| = 0 once tau = 1 (S = Y).

These identities hold whenever S_t = Pr(Y = 1 | F_t) under the measure used,
so comparing them with observed |Y - S_t| assumes observed prices are
calibrated probabilities. That is a statement about the real-world measure P;
see the P-vs-Q caveat in the experiment.

Each market is started from its own first observed midpoint p0^(i) at its first
observation time u_start^(i), and information time is measured from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, logit
from scipy.stats import norm

from .empirical_analysis import absolute_error, path_statistics, time_bins
from .information_clock import linear_clock

# Trapezoid rule on a fixed z-grid: E[g(Z)] for Z ~ N(0, 1). The integrand
# expit(c - s z) phi(z) is analytic, so the error decays like exp(-2 pi^2 / (s h));
# tests check agreement with adaptive quadrature to < 1e-8.
_Z = np.linspace(-9.0, 9.0, 1201)
_W = np.full(_Z.size, _Z[1] - _Z[0])
_W[[0, -1]] *= 0.5
_W *= norm.pdf(_Z)


def expected_abs_error(p0, s) -> np.ndarray:
    """E|Y - S| = 2 p0 E[1 - S | Y=1] with logit S | Y=1 = logit p0 + s Z + s^2 / 2.

    s = 0 gives 2 p0 (1 - p0) (no information yet); s = inf gives 0 (resolved).
    """
    p0, s = np.broadcast_arrays(np.asarray(p0, dtype=float), np.asarray(s, dtype=float))
    out = np.zeros(p0.shape)
    finite = np.isfinite(s)
    if np.any(finite):
        a = logit(p0[finite])[:, None] + 0.5 * s[finite][:, None] ** 2
        one_minus_S = expit(-(a + s[finite][:, None] * _Z[None, :]))
        out[finite] = 2.0 * p0[finite] * (one_minus_S * _W).sum(axis=1)  # explicit sum: no BLAS
    return out


def information_ratio_from_tau(kappa: float, tau) -> np.ndarray:
    """s = kappa sqrt(tau / (1 - tau)); inf at tau = 1."""
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(tau >= 1.0, np.inf, kappa * np.sqrt(tau / np.maximum(1.0 - tau, 0.0)))


# ---- data preparation --------------------------------------------------------------

def chronological_split(sample: pd.DataFrame, train_fraction: float = 0.75) -> tuple[list[str], list[str]]:
    """Earliest markets (by open time) for training, latest for testing; whole markets only."""
    used = sample[~sample["excluded"].astype(bool)].sort_values("open_time")
    n_train = int(round(train_fraction * len(used)))
    tickers = used["ticker"].tolist()
    return tickers[:n_train], tickers[n_train:]


def effective_resolution_time(prices: pd.DataFrame, tickers: list[str], threshold: float = 0.1) -> float:
    """Median Part F learning time over the given markets only."""
    stats = path_statistics(prices[prices["ticker"].isin(tickers)], threshold)
    return float(stats["learning_time"].median())


@dataclass(frozen=True)
class Observations:
    """Per-observation arrays for a set of markets."""

    ticker: np.ndarray
    u: np.ndarray
    p0: np.ndarray        # each market's own first observed midpoint
    u_start: np.ndarray   # each market's first observation time
    abs_error: np.ndarray
    bin: np.ndarray


def prepare_observations(prices: pd.DataFrame, tickers: list[str], n_bins: int = 10) -> Observations:
    df = prices[prices["ticker"].isin(tickers)].sort_values(["ticker", "normalized_time"])
    first = df.groupby("ticker").first()
    return Observations(
        ticker=df["ticker"].to_numpy(),
        u=df["normalized_time"].to_numpy(),
        p0=df["ticker"].map(first["price"]).to_numpy(),
        u_start=df["ticker"].map(first["normalized_time"]).to_numpy(),
        abs_error=absolute_error(df["price"], df["Y"]),
        bin=time_bins(df["normalized_time"], n_bins),
    )


# ---- model profiles ----------------------------------------------------------------------

def constant_kappa_values(obs: Observations, kappa: float, u_resolution: float) -> np.ndarray:
    """Model E|Y - S| at every observation: constant information rate from u_start to u_resolution."""
    tau = linear_clock(obs.u, obs.u_start, u_resolution)
    return expected_abs_error(obs.p0, information_ratio_from_tau(kappa, tau))


def binned_comparison(obs: Observations, model_values: np.ndarray | None = None) -> pd.DataFrame:
    """Per bin: empirical mean |Y - S|, its market-clustered s.e., and the model mean."""
    df = pd.DataFrame({"ticker": obs.ticker, "bin": obs.bin, "empirical": obs.abs_error})
    if model_values is not None:
        df["model"] = model_values
    per_market = df.groupby(["bin", "ticker"])["empirical"].mean()
    out = df.groupby("bin").agg(n_observations=("empirical", "size"), empirical=("empirical", "mean"))
    out["n_markets"] = per_market.groupby("bin").size()
    out["empirical_se"] = per_market.groupby("bin").std(ddof=1) / np.sqrt(out["n_markets"])
    if model_values is not None:
        out["model"] = df.groupby("bin")["model"].mean()
        out["residual"] = out["model"] - out["empirical"]
    return out.reset_index()


def binned_rmse(obs: Observations, model_values: np.ndarray) -> float:
    """sqrt(mean over bins of (model bin mean - empirical bin mean)^2); bins weighted equally."""
    b = binned_comparison(obs, model_values)
    return float(np.sqrt(np.mean(b["residual"] ** 2)))


# ---- fitting -------------------------------------------------------------------------------

KAPPA_BOUNDS = (0.01, 20.0)


def _minimize_on_log_grid(objective: Callable[[float], float], bounds: tuple[float, float],
                          n_grid: int = 121) -> tuple[float, float]:
    """Deterministic 1-D minimisation: log-spaced grid, then bounded refinement around the best point."""
    grid = np.geomspace(*bounds, n_grid)
    values = np.array([objective(k) for k in grid])
    i = int(np.argmin(values))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    res = optimize.minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    return (float(res.x), float(res.fun)) if res.fun <= values[i] else (float(grid[i]), float(values[i]))


def fit_constant_kappa(obs: Observations, u_resolution: float) -> tuple[float, float]:
    """kappa minimising the binned RMSE; returns (kappa_hat, training RMSE)."""
    return _minimize_on_log_grid(lambda k: binned_rmse(obs, constant_kappa_values(obs, k, u_resolution)),
                                 KAPPA_BOUNDS)


# ---- goodness of fit -------------------------------------------------------------------------

def simulate_observations_from_model(obs: Observations, kappa: float, u_resolution: float,
                                     rng: np.random.Generator) -> Observations:
    """Replace every market's observed path by one simulated from the constant-kappa model,
    at the same observation times and from the same p0 (Y drawn from Bernoulli(p0))."""
    from .information_clock import simulate_clock_paths

    abs_error = np.empty_like(obs.abs_error)
    for ticker in np.unique(obs.ticker):
        idx = np.nonzero(obs.ticker == ticker)[0]
        tau = linear_clock(obs.u[idx], obs.u_start[idx], u_resolution)
        grid = np.unique(np.r_[tau[tau < 1.0], 1.0])
        if grid[0] > 0.0:
            grid = np.r_[0.0, grid]
        path = simulate_clock_paths(float(obs.p0[idx[0]]), kappa, grid, 1, rng)
        S = np.interp(tau, grid, path.prediction_price[0])  # tau values are grid points: exact lookup
        abs_error[idx] = np.abs(path.Y[0] - S)
    return Observations(obs.ticker, obs.u, obs.p0, obs.u_start, abs_error, obs.bin)


def parametric_bootstrap_rmse(obs: Observations, kappa: float, u_resolution: float, n_replications: int,
                              seed_sequence: np.random.SeedSequence) -> np.ndarray:
    """Training RMSE (with kappa refitted) on data simulated from the fitted model itself.

    The observed training RMSE is compared with this distribution: if it is
    typical, the constant-kappa misfit is within what sampling noise from
    this many markets produces.
    """
    out = []
    for ss in seed_sequence.spawn(n_replications):
        sim = simulate_observations_from_model(obs, kappa, u_resolution, np.random.default_rng(ss))
        out.append(fit_constant_kappa(sim, u_resolution)[1])
    return np.asarray(out)
