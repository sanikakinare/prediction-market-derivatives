"""Descriptive statistics for real prediction-market paths (Part F).

Everything here is descriptive and measured under the real-world measure P; none
of it tests the risk-neutral (Q) martingale assumption of the pricing model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_error(price, Y) -> np.ndarray:
    """|Y - S_t|: distance between the market price and the outcome that occurred."""
    return np.abs(np.asarray(Y, dtype=float) - np.asarray(price, dtype=float))


def time_bins(normalized_time, n_bins: int = 10) -> np.ndarray:
    """Bin index 0..n_bins-1 for u in [0, 1]; u = 1 goes into the last bin."""
    u = np.asarray(normalized_time, dtype=float)
    return np.minimum((u * n_bins).astype(int), n_bins - 1)


def _with_bins(prices: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    df = prices.copy()
    df["bin"] = time_bins(df["normalized_time"], n_bins)
    df["bin_label"] = [f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}" for b in df["bin"]]
    df["abs_error"] = absolute_error(df["price"], df["Y"])
    return df


def error_by_time_bin(prices: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Per normalized-time bin: observation and market counts, mean/median/quartiles of |Y - S|."""
    df = _with_bins(prices, n_bins)
    g = df.groupby(["bin", "bin_label"])
    out = g["abs_error"].agg(n_observations="size", mean_abs_error="mean", median_abs_error="median",
                             q25_abs_error=lambda x: x.quantile(0.25), q75_abs_error=lambda x: x.quantile(0.75))
    out["n_markets"] = g["ticker"].nunique()
    return out.reset_index()


def price_by_outcome(prices: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Per bin and outcome: median, mean and quartiles of the YES price."""
    df = _with_bins(prices, n_bins)
    g = df.groupby(["Y", "bin", "bin_label"])["price"]
    return g.agg(n_observations="size", mean_price="mean", median_price="median",
                 q25_price=lambda x: x.quantile(0.25), q75_price=lambda x: x.quantile(0.75)).reset_index()


def learning_time(path: pd.DataFrame, threshold: float = 0.1) -> float:
    """First normalized time after which |Y - S_t| stays below ``threshold`` until the
    last observation. NaN if the path never settles that close to the outcome."""
    err = absolute_error(path["price"], path["Y"])
    u = path["normalized_time"].to_numpy()
    above = np.nonzero(err >= threshold)[0]
    if above.size == 0:
        return float(u[0])
    last_above = above[-1]
    return float(u[last_above + 1]) if last_above + 1 < len(u) else np.nan


def path_statistics(prices: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Per-market jump and learning-speed descriptors.

    largest_move           max |S_{k+1} - S_k| between consecutive clean observations
    largest_move_share     largest_move / sum of |moves| (1 = a single step does everything)
    largest_move_time      normalized time at the end of the largest move
    learning_time          see ``learning_time``
    """
    rows = []
    for ticker, path in prices.sort_values(["ticker", "normalized_time"]).groupby("ticker"):
        moves = np.abs(np.diff(path["price"].to_numpy()))
        total = moves.sum()
        k = int(np.argmax(moves)) if moves.size else 0
        rows.append({
            "ticker": ticker,
            "Y": int(path["Y"].iloc[0]),
            "n_observations": len(path),
            "largest_move": moves.max() if moves.size else np.nan,
            "largest_move_share": moves.max() / total if total > 0 else np.nan,
            "largest_move_time": float(path["normalized_time"].iloc[k + 1]) if moves.size else np.nan,
            "learning_time": learning_time(path, threshold),
            "final_abs_error": float(absolute_error(path["price"].iloc[-1], path["Y"].iloc[-1])),
        })
    return pd.DataFrame(rows)
