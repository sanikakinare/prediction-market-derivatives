"""Part F: qualitative sanity check of the information model on real Kalshi data.

Run from the project root (downloads on first run, then uses the cache in data/raw):
    .venv/bin/python experiments/05_real_data.py

Data: Kalshi series KXHIGHNY ("Highest temperature in NYC"), the most recent
N_EVENTS settled daily events, one market per event (the strike bucket whose
first clean midpoint is closest to 0.5, i.e. chosen without using the outcome).

Outputs:
    data/raw/                         raw API responses (events, hourly candles)
    data/processed/F_prices_long.csv  ticker, timestamp, normalized_time, price, Y, ...
    results/F_market_sample.csv       one row per market (+ path statistics)
    results/F_error_by_time.csv       |Y - S| by normalized-time bin
    results/F_price_by_outcome.csv    price by bin and outcome
    results/F_summary.csv             headline numbers
    results/F_example_market_paths.png
    results/F_uncertainty_vs_time.png
    results/F_paths_by_outcome.png
    results/F_learning_and_jumps.png

All statistics are real-world (P) descriptions of price behaviour; they say
nothing directly about the risk-neutral (Q) martingale assumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.empirical_analysis import (  # noqa: E402
    absolute_error,
    error_by_time_bin,
    path_statistics,
    price_by_outcome,
)
from src.kalshi_data import build_dataset, to_unix  # noqa: E402

SERIES = "KXHIGHNY"
N_EVENTS = 40
N_BINS = 10
MIN_OBSERVATIONS = 10
MAX_SPREAD = 0.5
LEARNING_THRESHOLD = 0.1
N_EXAMPLES_PER_OUTCOME = 6

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for d in (RAW, PROCESSED, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# ---- plot style (shared palette) -------------------------------------------------
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
})


ET = "America/New_York"


def target_day_midnight_et(close_time: str) -> pd.Timestamp:
    """00:00 America/New_York on the event's target day.

    Trading closes at 05:00 UTC after the target day, i.e. 00:00 EST or 01:00 EDT
    (midnight local standard time), so stepping back two hours from the close in
    ET always lands on the target day, in or out of daylight saving time.
    """
    close_et = pd.Timestamp(close_time).tz_convert(ET)
    return (close_et - pd.Timedelta(hours=2)).normalize()


def hours_after_target_midnight_et(row: pd.Series, u: float) -> float:
    """Normalized time -> wall-clock hours (ET) after midnight starting the target day."""
    open_ts, close_ts = to_unix(row["open_time"]), to_unix(row["close_time"])
    t = open_ts + u * (close_ts - open_ts)
    return (t - target_day_midnight_et(row["close_time"]).timestamp()) / 3600.0


def clock_reference_points(sample: pd.DataFrame) -> dict[str, float]:
    """Median normalized time of 00:00 and 12:00 ET on the target day."""
    refs = {}
    for hour in (0, 12):
        us = []
        for _, row in sample.iterrows():
            open_ts, close_ts = to_unix(row["open_time"]), to_unix(row["close_time"])
            t = target_day_midnight_et(row["close_time"]).timestamp() + hour * 3600
            us.append((t - open_ts) / (close_ts - open_ts))
        if np.ptp(us) > 1e-9:
            print(f"note: {hour:02d}:00 ET maps to slightly different u across markets; using the median")
        refs[f"target day {hour:02d}:00 ET"] = float(np.median(us))
    return refs


def add_clock_lines(ax, refs: dict[str, float], labels: bool = True) -> None:
    for name, u in refs.items():
        ax.axvline(u, color=INK_MUTED, ls=":", lw=1.0)
        if labels:
            ax.text(u, 1.0, " " + name.replace("target day ", "target day\n "), transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7.5, color=INK_MUTED)


def plot_example_paths(prices: pd.DataFrame, sample: pd.DataFrame, refs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, y, title in ((axes[0], 1, "Resolved YES (Y = 1)"), (axes[1], 0, "Resolved NO (Y = 0)")):
        subset = sample[(sample["Y"] == y) & ~sample["excluded"]].sort_values("open_time")
        idx = np.unique(np.linspace(0, len(subset) - 1, N_EXAMPLES_PER_OUTCOME).round().astype(int))
        for color, (_, row) in zip(SERIES_COLORS, subset.iloc[idx].iterrows()):
            path = prices[prices["ticker"] == row["ticker"]]
            date = row["event_ticker"].split("-")[-1]
            ax.plot(path["normalized_time"], path["price"], drawstyle="steps-post", color=color, lw=1.5,
                    marker="o", ms=2.5, label=f"{date}  {row['yes_sub_title']}")
        add_clock_lines(ax, refs)
        ax.set(title=title, xlabel="normalized time $u$ (0 = market open, 1 = trading close)",
               ylim=(-0.03, 1.03), xlim=(0, 1))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5,
                  title="event date / YES bucket", title_fontsize=8)
    axes[0].set_ylabel("YES price (bid-ask midpoint)")
    fig.suptitle(f"Kalshi {SERIES}: example hourly paths (evenly spaced by date, not hand-picked)", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "F_example_market_paths.png")
    plt.close(fig)


def plot_uncertainty(prices: pd.DataFrame, err: pd.DataFrame, refs: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for _, path in prices.groupby("ticker"):
        ax.plot(path["normalized_time"], absolute_error(path["price"], path["Y"]), drawstyle="steps-post",
                color=FAINT, lw=0.7, alpha=0.6)
    centers = (err["bin"] + 0.5) / N_BINS
    ax.fill_between(centers, err["q25_abs_error"], err["q75_abs_error"], color=BLUE, alpha=0.15, lw=0,
                    label="interquartile range (per bin)")
    ax.plot(centers, err["mean_abs_error"], "o-", color=BLUE, label=r"mean $|Y-S_t|$")
    ax.plot(centers, err["median_abs_error"], "s--", color=ORANGE, ms=5, label=r"median $|Y-S_t|$")
    ax.plot([], [], color=FAINT, lw=1, label="individual markets")
    add_clock_lines(ax, refs)
    n_markets = prices["ticker"].nunique()
    ax.set(xlabel="normalized time $u$", ylabel=r"forecast error $|Y - S_t|$", ylim=(-0.02, 1.02), xlim=(0, 1),
           title=f"Distance from the eventual outcome ({n_markets} markets, {len(prices):,} hourly observations)")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "F_uncertainty_vs_time.png")
    plt.close(fig)


def plot_by_outcome(prices: pd.DataFrame, by_outcome: pd.DataFrame, refs: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for y, color, name in ((1, BLUE, "resolved YES"), (0, ORANGE, "resolved NO")):
        for _, path in prices[prices["Y"] == y].groupby("ticker"):
            ax.plot(path["normalized_time"], path["price"], drawstyle="steps-post", color=color, lw=0.6, alpha=0.18)
        b = by_outcome[by_outcome["Y"] == y]
        centers = (b["bin"] + 0.5) / N_BINS
        n = prices.loc[prices["Y"] == y, "ticker"].nunique()
        ax.fill_between(centers, b["q25_price"], b["q75_price"], color=color, alpha=0.18, lw=0)
        ax.plot(centers, b["median_price"], "o-", color=color, lw=2.2, label=f"{name}: median, IQR band ({n} markets)")
    add_clock_lines(ax, refs)
    ax.set(xlabel="normalized time $u$", ylabel="YES price", ylim=(-0.03, 1.03), xlim=(0, 1),
           title="YES and NO markets: typical price path (thin lines: individual markets)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "F_paths_by_outcome.png")
    plt.close(fig)


def plot_learning_and_jumps(stats: pd.DataFrame) -> None:
    fig, (ax_l, ax_j) = plt.subplots(1, 2, figsize=(12, 4.3))
    learned = stats["learning_hour_et"].dropna()
    ax_l.hist(learned, bins=np.arange(np.floor(learned.min()) - 0.5, np.ceil(learned.max()) + 1.5, 1.0),
              color=BLUE, alpha=0.75, edgecolor="white")
    ax_l.axvline(learned.median(), color=ORANGE, lw=1.6, label=f"median {learned.median():.0f}:00 ET")
    ax_l.set(xlabel="hour (ET) on the target day", ylabel="number of markets",
             title=rf"When $|Y-S_t|$ falls below {LEARNING_THRESHOLD} for good" + f"\n({len(learned)} of {len(stats)} markets)")
    ax_l.legend(loc="upper left", fontsize=8.5)

    ax_j.hist(stats["largest_move"], bins=np.linspace(0, 1, 21), color=ORANGE, alpha=0.75, edgecolor="white")
    share = stats["largest_move_share"].median()
    ax_j.set(xlabel=r"largest single-hour price change $\max_k |S_{k+1}-S_k|$", ylabel="number of markets",
             title="Largest single-hour move per market\n"
                   f"(median: {stats['largest_move'].median():.2f}; median share of total movement: {share:.0%})")
    fig.tight_layout()
    fig.savefig(RESULTS / "F_learning_and_jumps.png")
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 160)
    prices, sample = build_dataset(SERIES, N_EVENTS, RAW, MIN_OBSERVATIONS, MAX_SPREAD)
    prices.to_csv(PROCESSED / "F_prices_long.csv", index=False)

    used = sample[~sample["excluded"]]
    print(f"=== Dataset: Kalshi {SERIES} ({sample['title'].iloc[0].split(' on ')[0]}) ===")
    print(f"events downloaded: {len(sample)}, markets used: {len(used)} "
          f"(YES {int(used['Y'].sum())}, NO {int((1 - used['Y']).sum())}), excluded: {int(sample['excluded'].sum())}")
    print(f"date range: {used['open_time'].min()[:10]} to {used['close_time'].max()[:10]}")
    print(f"hourly observations: {len(prices):,} (median {int(used['n_observations'].median())} per market); "
          f"median spread {prices['spread'].median():.3f}")
    print(f"mean first price {used['first_price'].mean():.3f}; YES rate {used['Y'].mean():.3f}")

    refs = clock_reference_points(used)

    err = error_by_time_bin(prices, N_BINS)
    err.to_csv(RESULTS / "F_error_by_time.csv", index=False)
    print("\n=== |Y - S_t| by normalized-time bin ===")
    print(err.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    by_outcome = price_by_outcome(prices, N_BINS)
    by_outcome.to_csv(RESULTS / "F_price_by_outcome.csv", index=False)
    wide = by_outcome.pivot(index="bin_label", columns="Y", values="median_price").rename(
        columns={0: "median_price_NO", 1: "median_price_YES"})
    wide["gap"] = wide["median_price_YES"] - wide["median_price_NO"]
    print("\n=== Median YES price by outcome ===")
    print(wide.to_string(float_format=lambda x: f"{x:.3f}"))

    stats = path_statistics(prices, LEARNING_THRESHOLD)
    by_ticker = used.set_index("ticker")
    stats["learning_hour_et"] = [
        hours_after_target_midnight_et(by_ticker.loc[t], u) if np.isfinite(u) else np.nan
        for t, u in zip(stats["ticker"], stats["learning_time"])]
    stats["largest_move_hour_et"] = [
        hours_after_target_midnight_et(by_ticker.loc[t], u)
        for t, u in zip(stats["ticker"], stats["largest_move_time"])]
    sample.merge(stats.drop(columns=["Y", "n_observations"]), on="ticker", how="left").to_csv(
        RESULTS / "F_market_sample.csv", index=False)

    lt = stats["learning_time"]
    print("\n=== Jumps and learning speed (per market) ===")
    print(f"largest hourly move: median {stats['largest_move'].median():.2f}, "
          f"> 0.3 in {np.mean(stats['largest_move'] > 0.3):.0%} of markets, "
          f"> 0.5 in {np.mean(stats['largest_move'] > 0.5):.0%}")
    print(f"largest move as share of total absolute movement: median {stats['largest_move_share'].median():.0%}")
    print(f"largest move happens at (ET hour on target day): median {stats['largest_move_hour_et'].median():.1f}, "
          f"IQR [{stats['largest_move_hour_et'].quantile(0.25):.1f}, {stats['largest_move_hour_et'].quantile(0.75):.1f}]")
    print(f"learning time u (|Y-S| < {LEARNING_THRESHOLD} for good): reached by {lt.notna().sum()} of {len(lt)}; "
          f"median u = {lt.median():.2f}, IQR [{lt.quantile(0.25):.2f}, {lt.quantile(0.75):.2f}], "
          f"range [{lt.min():.2f}, {lt.max():.2f}]")
    lh = stats["learning_hour_et"]
    print(f"  in ET clock time on the target day: median {lh.median():.1f}h, "
          f"IQR [{lh.quantile(0.25):.1f}, {lh.quantile(0.75):.1f}], range [{lh.min():.1f}, {lh.max():.1f}]")
    print(f"final observed |Y - S|: median {stats['final_abs_error'].median():.3f}, max {stats['final_abs_error'].max():.3f}")

    first, last = err.iloc[0], err.iloc[-1]
    summary = pd.DataFrame([
        ("series", SERIES), ("markets_used", len(used)), ("markets_yes", int(used["Y"].sum())),
        ("markets_no", int((1 - used["Y"]).sum())), ("markets_excluded", int(sample["excluded"].sum())),
        ("hourly_observations", len(prices)), ("median_spread", prices["spread"].median()),
        ("mean_abs_error_first_bin", first["mean_abs_error"]), ("mean_abs_error_last_bin", last["mean_abs_error"]),
        ("median_abs_error_first_bin", first["median_abs_error"]), ("median_abs_error_last_bin", last["median_abs_error"]),
        ("median_largest_hourly_move", stats["largest_move"].median()),
        ("share_markets_largest_move_gt_0.3", np.mean(stats["largest_move"] > 0.3)),
        ("median_largest_move_share_of_total", stats["largest_move_share"].median()),
        ("median_learning_time_u", lt.median()), ("iqr_learning_time_u", lt.quantile(0.75) - lt.quantile(0.25)),
        ("median_learning_hour_et_target_day", lh.median()),
    ], columns=["metric", "value"])
    summary.to_csv(RESULTS / "F_summary.csv", index=False)

    plot_example_paths(prices, used, refs)
    plot_uncertainty(prices, err, refs)
    plot_by_outcome(prices, by_outcome, refs)
    plot_learning_and_jumps(stats)
    print(f"\nreference lines: {', '.join(f'{k} at u = {v:.3f}' for k, v in refs.items())}")
    print(f"Figures and tables written to {RESULTS.relative_to(ROOT)}/ and data to {PROCESSED.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
