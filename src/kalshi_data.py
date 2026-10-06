"""Download and clean historical Kalshi market data (Part F).

Public market-data endpoints (no authentication) of the Kalshi Trade API v2:

    GET /events?series_ticker=...&status=settled&with_nested_markets=true
    GET /series/{series_ticker}/markets/{ticker}/candlesticks
        ?start_ts&end_ts&period_interval=60          (hourly candles)

Markets settled before Kalshi's historical cutoff (GET /historical/cutoff) are
served only by GET /historical/markets/{ticker}/candlesticks; this module uses
the live endpoint and therefore restricts itself to recently settled events.

Prices are US-dollar fixed-point strings (e.g. "0.4200"), i.e. already in [0, 1].

Price definition: S_t = (yes_bid_close + yes_ask_close) / 2 at the end of each
hourly candle. Cleaning rules (deliberately simple):
    * drop a candle if either quote is missing, if ask < bid, or if the spread
      exceeds ``max_spread`` (this removes the empty book, bid 0 / ask 1, that
      appears after trading closes);
    * drop duplicate timestamps (keep the last), sort by time;
    * hours without a candle are left missing, not forward-filled.
The final observed price is never forced to 0 or 1; the outcome Y is stored
separately.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_PAUSE_SECONDS = 0.15  # stay well inside public rate limits


# ---- HTTP + raw caching ----------------------------------------------------------

def _get(path: str, params: dict | None = None, retries: int = 4) -> dict:
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            time.sleep(REQUEST_PAUSE_SECONDS)
            return payload
        except Exception:  # network hiccup or 429: back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _cached(path: Path, fetch) -> dict:
    """Load raw JSON from disk if present; otherwise fetch it and save it unchanged."""
    if path.exists():
        return json.loads(path.read_text())
    payload = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return payload


def fetch_settled_events(series_ticker: str, n_events: int, raw_dir: Path) -> list[dict]:
    """Most recent ``n_events`` settled events of a series, with nested markets."""
    def fetch() -> dict:
        events, cursor = [], None
        while len(events) < n_events:
            params = {"series_ticker": series_ticker, "status": "settled",
                      "with_nested_markets": "true", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            page = _get("/events", params)
            events.extend(page.get("events", []))
            cursor = page.get("cursor")
            if not cursor or not page.get("events"):
                break
        return {"series_ticker": series_ticker, "events": events[:n_events]}

    return _cached(raw_dir / f"{series_ticker}_events.json", fetch)["events"]


def fetch_hourly_candles(series_ticker: str, market: dict, raw_dir: Path) -> list[dict]:
    """Hourly candlesticks over the market's full open -> close window."""
    start_ts = to_unix(market["open_time"])
    end_ts = to_unix(market["close_time"]) + 3600

    def fetch() -> dict:
        return _get(f"/series/{series_ticker}/markets/{market['ticker']}/candlesticks",
                    {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60})

    return _cached(raw_dir / "candles" / f"{market['ticker']}.json", fetch).get("candlesticks", [])


# ---- parsing and cleaning (pure functions, tested offline) ------------------------

def to_unix(iso_timestamp: str) -> int:
    return int(datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")).timestamp())


def target_day_midnight_et(close_time: str) -> pd.Timestamp:
    """00:00 America/New_York on a KXHIGHNY event's target day.

    Trading closes at 05:00 UTC after the target day, i.e. 00:00 EST or 01:00 EDT
    (midnight local standard time), so stepping back two hours from the close in
    ET always lands on the target day, in or out of daylight saving time.
    """
    close_et = pd.Timestamp(close_time).tz_convert("America/New_York")
    return (close_et - pd.Timedelta(hours=2)).normalize()


def normalized_time_of_clock(open_time: str, close_time: str, hours_after_target_midnight: float) -> float:
    """Normalized market time u of a wall-clock time (ET) on the target day."""
    t = target_day_midnight_et(close_time).timestamp() + 3600.0 * hours_after_target_midnight
    return (t - to_unix(open_time)) / (to_unix(close_time) - to_unix(open_time))


def parse_outcome(result: str) -> int:
    """Kalshi settlement result -> Y. Only 'yes'/'no' are valid binary outcomes."""
    mapping = {"yes": 1, "no": 0}
    key = str(result).strip().lower()
    if key not in mapping:
        raise ValueError(f"Not a binary YES/NO settlement: {result!r}")
    return mapping[key]


def _dollars(block: dict | None, field: str = "close_dollars") -> float:
    if not block or block.get(field) in (None, ""):
        return np.nan
    return float(block[field])


def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """Raw candle list -> one row per candle with close quotes, last trade and volume."""
    rows = [{
        "end_period_ts": int(c["end_period_ts"]),
        "yes_bid": _dollars(c.get("yes_bid")),
        "yes_ask": _dollars(c.get("yes_ask")),
        "last_trade": _dollars(c.get("price")),
        "volume": float(c.get("volume_fp") or 0.0),
    } for c in candles]
    return pd.DataFrame(rows, columns=["end_period_ts", "yes_bid", "yes_ask", "last_trade", "volume"])


def clean_prices(frame: pd.DataFrame, max_spread: float = 0.5) -> pd.DataFrame:
    """Bid-ask midpoints with the cleaning rules from the module docstring."""
    df = frame.drop_duplicates(subset="end_period_ts", keep="last").sort_values("end_period_ts")
    valid = (
        df["yes_bid"].notna() & df["yes_ask"].notna()
        & (df["yes_ask"] >= df["yes_bid"])
        & (df["yes_ask"] - df["yes_bid"] <= max_spread)
        & df["yes_bid"].between(0.0, 1.0) & df["yes_ask"].between(0.0, 1.0)
    )
    df = df[valid].copy()
    df["spread"] = df["yes_ask"] - df["yes_bid"]
    df["price"] = (df["yes_bid"] + df["yes_ask"]) / 2.0
    return df.reset_index(drop=True)


def normalized_time(ts, start_ts: float, end_ts: float) -> np.ndarray:
    """u = (t - t_start) / (t_end - t_start), clipped to [0, 1]."""
    if end_ts <= start_ts:
        raise ValueError("end_ts must be after start_ts.")
    u = (np.asarray(ts, dtype=float) - start_ts) / (end_ts - start_ts)
    return np.clip(u, 0.0, 1.0)


def market_price_frame(market: dict, candles: list[dict], max_spread: float = 0.5) -> pd.DataFrame:
    """Long-format clean prices for one market: ticker, timestamp, normalized_time, price, Y, ..."""
    start_ts, end_ts = to_unix(market["open_time"]), to_unix(market["close_time"])
    df = clean_prices(candles_to_frame(candles), max_spread)
    df = df[df["end_period_ts"] <= end_ts].copy()  # trading-period observations only
    df.insert(0, "ticker", market["ticker"])
    df.insert(1, "timestamp", pd.to_datetime(df["end_period_ts"], unit="s", utc=True))
    df.insert(2, "normalized_time", normalized_time(df["end_period_ts"], start_ts, end_ts))
    df["Y"] = parse_outcome(market["result"])
    return df.drop(columns="end_period_ts").reset_index(drop=True)


def select_most_uncertain_market(markets_with_prices: dict[str, pd.DataFrame]) -> str | None:
    """Outcome-blind choice of one market per event: the one whose *first* clean
    midpoint is closest to 0.5 (uses only information available at the start)."""
    best, best_dist = None, np.inf
    for ticker, df in markets_with_prices.items():
        if df.empty:
            continue
        dist = abs(df["price"].iloc[0] - 0.5)
        if dist < best_dist:
            best, best_dist = ticker, dist
    return best


def build_dataset(
    series_ticker: str,
    n_events: int,
    raw_dir: Path,
    min_observations: int = 10,
    max_spread: float = 0.5,
    log=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or load cached) data and return (long price data, market sample table).

    One market per event (``select_most_uncertain_market``) so that the sample
    consists of distinct, non-overlapping days rather than correlated strike
    buckets of the same event.
    """
    events = fetch_settled_events(series_ticker, n_events, raw_dir)
    frames, rows = [], []
    for event in events:
        candidates = {}
        markets = {m["ticker"]: m for m in event.get("markets", [])}
        for ticker, market in markets.items():
            if str(market.get("result", "")).lower() not in ("yes", "no"):
                continue
            candles = fetch_hourly_candles(series_ticker, market, raw_dir)
            candidates[ticker] = market_price_frame(market, candles, max_spread)
        chosen = select_most_uncertain_market(candidates)
        if chosen is None:
            log(f"  {event['event_ticker']}: no usable market, skipped")
            continue
        df, market = candidates[chosen], markets[chosen]
        excluded = len(df) < min_observations
        rows.append({
            "ticker": chosen,
            "event_ticker": event["event_ticker"],
            "title": event.get("title", ""),
            "yes_sub_title": market.get("yes_sub_title", ""),
            "open_time": market["open_time"],
            "close_time": market["close_time"],
            "Y": parse_outcome(market["result"]),
            "n_observations": len(df),
            "first_price": df["price"].iloc[0] if len(df) else np.nan,
            "last_price": df["price"].iloc[-1] if len(df) else np.nan,
            "median_spread": df["spread"].median() if len(df) else np.nan,
            "total_volume": float(market.get("volume_fp") or 0.0),
            "excluded": excluded,
        })
        if not excluded:
            frames.append(df.assign(event_ticker=event["event_ticker"]))
    sample = pd.DataFrame(rows)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return prices, sample
