"""Tests for Part F data handling and descriptive statistics (offline, no network).

These check the data pipeline only; they do not require the empirical results to
agree with the theoretical model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.empirical_analysis import absolute_error, error_by_time_bin, learning_time, path_statistics, time_bins
from src.kalshi_data import (
    candles_to_frame,
    clean_prices,
    market_price_frame,
    normalized_time,
    parse_outcome,
    select_most_uncertain_market,
    to_unix,
)

OPEN, CLOSE = "2026-10-03T14:00:00Z", "2026-10-05T05:00:00Z"


def candle(ts: int, bid, ask, last=None, volume="1.00") -> dict:
    q = lambda v: None if v is None else {"close_dollars": f"{v:.4f}"}  # noqa: E731
    return {"end_period_ts": ts, "yes_bid": q(bid), "yes_ask": q(ask),
            "price": {} if last is None else {"close_dollars": f"{last:.4f}"}, "volume_fp": volume}


def market(result: str = "yes") -> dict:
    return {"ticker": "TEST-1", "open_time": OPEN, "close_time": CLOSE, "result": result}


# ---- outcome mapping ------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [("yes", 1), ("no", 0), ("YES", 1), (" No ", 0)])
def test_outcome_mapping(raw, expected):
    assert parse_outcome(raw) == expected


@pytest.mark.parametrize("raw", ["", "void", "scalar", None, "all_no"])
def test_non_binary_outcomes_rejected(raw):
    with pytest.raises(ValueError):
        parse_outcome(raw)


# ---- cleaning ------------------------------------------------------------------------

def test_midpoint_and_invalid_quotes_dropped():
    t0 = to_unix(OPEN)
    raw = [
        candle(t0 + 3600, 0.40, 0.44),
        candle(t0 + 7200, 0.00, 1.00),       # empty book: spread 1.0 > 0.5
        candle(t0 + 10800, 0.50, 0.45),      # crossed quotes
        candle(t0 + 14400, None, 0.30),      # missing bid
        candle(t0 + 18000, 0.98, 0.99),
    ]
    df = clean_prices(candles_to_frame(raw))
    assert df["price"].tolist() == pytest.approx([0.42, 0.985])
    assert np.all((df["price"] >= 0) & (df["price"] <= 1))


def test_duplicates_and_ordering_handled_consistently():
    t0 = to_unix(OPEN)
    raw = [candle(t0 + 7200, 0.50, 0.52), candle(t0 + 3600, 0.40, 0.42), candle(t0 + 7200, 0.60, 0.62)]
    df = clean_prices(candles_to_frame(raw))
    assert df["end_period_ts"].is_monotonic_increasing
    assert df["end_period_ts"].is_unique
    assert df["price"].tolist() == pytest.approx([0.41, 0.61])  # duplicate: last one kept


def test_market_frame_normalized_time_ordering_and_outcome():
    t0, t1 = to_unix(OPEN), to_unix(CLOSE)
    raw = [candle(t0 + h * 3600, 0.30 + 0.01 * h, 0.32 + 0.01 * h) for h in range(1, 40)]
    raw.append(candle(t1 + 3600, 0.98, 0.99))  # after close: excluded
    df = market_price_frame(market("no"), list(reversed(raw)))
    assert df["normalized_time"].between(0, 1).all()
    assert df["normalized_time"].is_monotonic_increasing
    assert df["timestamp"].is_monotonic_increasing
    assert (df["Y"] == 0).all()
    assert df["normalized_time"].iloc[-1] <= 1.0
    # last observed price is NOT forced to the outcome
    assert df["price"].iloc[-1] != 0.0


def test_missing_hours_are_not_filled():
    t0 = to_unix(OPEN)
    raw = [candle(t0 + 3600, 0.4, 0.42), candle(t0 + 5 * 3600, 0.5, 0.52)]
    df = market_price_frame(market(), raw)
    assert len(df) == 2


def test_normalized_time_bounds():
    u = normalized_time([0, 50, 100, 150, -10], 0, 100)
    assert u.tolist() == [0.0, 0.5, 1.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        normalized_time([1], 5, 5)


def test_selection_is_outcome_blind_closest_to_half():
    frames = {"A": pd.DataFrame({"price": [0.20, 0.9]}), "B": pd.DataFrame({"price": [0.45, 0.0]}),
              "C": pd.DataFrame({"price": []})}
    assert select_most_uncertain_market(frames) == "B"


# ---- statistics -------------------------------------------------------------------------

def test_absolute_error():
    assert absolute_error([0.6, 0.9, 0.2], [1, 1, 0]).tolist() == pytest.approx([0.4, 0.1, 0.2])


def test_time_bins_include_endpoint():
    assert time_bins([0.0, 0.05, 0.1, 0.99, 1.0], 10).tolist() == [0, 0, 1, 9, 9]


def _toy_prices() -> pd.DataFrame:
    u = np.linspace(0, 1, 11)
    yes = pd.DataFrame({"ticker": "Y1", "normalized_time": u, "price": np.r_[[0.5] * 7, 0.8, 0.95, 0.99, 0.99], "Y": 1})
    no = pd.DataFrame({"ticker": "N1", "normalized_time": u, "price": np.r_[[0.4] * 6, 0.9, 0.05, 0.02, 0.01, 0.01], "Y": 0})
    return pd.concat([yes, no], ignore_index=True)


def test_error_by_time_bin_counts_and_values():
    err = error_by_time_bin(_toy_prices(), n_bins=10)
    assert err["n_observations"].sum() == 22
    assert err["n_markets"].max() == 2
    first = err.iloc[0]
    assert first["mean_abs_error"] == pytest.approx((0.5 + 0.4) / 2)


def test_learning_time_and_jump_statistics():
    stats = path_statistics(_toy_prices(), threshold=0.1).set_index("ticker")
    assert stats.loc["Y1", "learning_time"] == pytest.approx(0.8)    # 0.95 at u=0.8 and stays
    assert stats.loc["N1", "learning_time"] == pytest.approx(0.7)    # reversal at u=0.6, then 0.05
    assert stats.loc["N1", "largest_move"] == pytest.approx(0.85)    # 0.9 -> 0.05
    assert stats.loc["N1", "largest_move_time"] == pytest.approx(0.7)


def test_learning_time_nan_when_never_settles():
    path = pd.DataFrame({"normalized_time": [0, 0.5, 1.0], "price": [0.5, 0.6, 0.7], "Y": 1})
    assert np.isnan(learning_time(path, 0.1))
