"""Part G: calibrate the information model to Kalshi KXHIGHNY, test it out of sample,
and bridge back to option pricing.

Run from the project root (uses Part F's processed data; no network needed):
    .venv/bin/python experiments/06_calibration.py

Protocol (fixed before looking at any test-set number):
    1. Chronological split: earliest 75% of markets train, latest 25% test.
    2. T_effective = median Part F learning time over TRAINING markets.
    3. Each market starts from its own first observed midpoint p0^(i).
    4. One common kappa fitted on TRAINING by binned squared error of mean |Y - S_t|.
    5. Goodness of fit on TRAINING: parametric bootstrap of the training RMSE under
       the fitted model (kappa refitted on each replicate). Refine the information
       timing only if the observed RMSE exceeds the bootstrap 95th percentile.
    6. Evaluate the selected specification(s) once on the TEST markets.

Outputs (results/):
    G_calibration_summary.csv  headline numbers
    G_bin_profiles.csv         empirical vs model by normalized-time bin (train and test)
    G_bootstrap_rmse.csv       goodness-of-fit replicates
    G_pricing_bridge.csv       illustrative option premiums
    G_constant_kappa_fit.png   training fit, residuals, goodness of fit
    G_model_comparison.png     train vs held-out test (main final figure)
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

from src.benchmark import call_price_closed_form  # noqa: E402
from src.calibration import (  # noqa: E402
    binned_comparison,
    binned_rmse,
    chronological_split,
    constant_kappa_values,
    effective_resolution_time,
    fit_constant_kappa,
    information_ratio_from_tau,
    parametric_bootstrap_rmse,
    prepare_observations,
    simulate_observations_from_model,
)
from src.information_clock import linear_clock  # noqa: E402
from src.kalshi_data import normalized_time_of_clock, target_day_midnight_et, to_unix  # noqa: E402
from src.option_pricing import price_european_call_mc  # noqa: E402

RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

seed = 42
TRAIN_FRACTION = 0.75
N_BINS = 10
LEARNING_THRESHOLD = 0.1
N_BOOTSTRAP = 200
N_BAND = 300
GOF_QUANTILE = 0.95

# Illustrative option (Parts C-E): p0 = 0.6, K = 0.7, expiry at 2/3 of the information horizon.
P0_OPTION, K_OPTION, RELATIVE_EXPIRY = 0.60, 0.70, 2 / 3
SYNTHETIC_KAPPA = 1.5
N_MC = 400_000

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "lines.linewidth": 1.8, "legend.frameon": False,
})


def hours_et(sample_row: pd.Series, u: float) -> float:
    """Normalized time -> hours after midnight ET starting the target day."""
    open_ts, close_ts = to_unix(sample_row["open_time"]), to_unix(sample_row["close_time"])
    t = open_ts + u * (close_ts - open_ts)
    return (t - target_day_midnight_et(sample_row["close_time"]).timestamp()) / 3600.0


def clock_label(hours: float) -> str:
    day = "day before" if hours < 0 else ("target day" if hours < 24 else "day after")
    h = hours % 24
    return f"{day} {int(h):02d}:{int(round((h % 1) * 60)):02d} ET"


def model_band(obs, kappa: float, u_resolution: float, seed_sequence, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-95% range of bin means, and RMSEs at fixed kappa, for data simulated from the model."""
    model_values = constant_kappa_values(obs, kappa, u_resolution)
    means, rmses = [], []
    for ss in seed_sequence.spawn(n):
        sim = simulate_observations_from_model(obs, kappa, u_resolution, np.random.default_rng(ss))
        means.append(binned_comparison(sim)["empirical"].to_numpy())
        rmses.append(binned_rmse(sim, model_values))
    means = np.array(means)
    return np.quantile(means, 0.05, axis=0), np.quantile(means, 0.95, axis=0), np.array(rmses)


def main() -> None:
    pd.set_option("display.width", 160)
    sample = pd.read_csv(RESULTS / "F_market_sample.csv")
    prices = pd.read_csv(PROCESSED / "F_prices_long.csv")
    ss_boot, ss_band_train, ss_band_test, ss_price = np.random.SeedSequence(seed).spawn(4)

    # ---- G1 split ---------------------------------------------------------------------------
    train, test = chronological_split(sample, TRAIN_FRACTION)
    s_idx = sample.set_index("ticker")
    print("=== G1  Chronological split ===")
    print(f"train: {len(train)} markets, {s_idx.loc[train, 'event_ticker'].iloc[0]} .. {s_idx.loc[train, 'event_ticker'].iloc[-1]}")
    print(f"test:  {len(test)} markets, {s_idx.loc[test, 'event_ticker'].iloc[0]} .. {s_idx.loc[test, 'event_ticker'].iloc[-1]}")

    # ---- G2 effective resolution time (training only) ----------------------------------------
    u_eff = effective_resolution_time(prices, train, LEARNING_THRESHOLD)
    ref_row = s_idx.loc[train[0]]
    print("\n=== G2  Effective resolution time (training markets only) ===")
    print(f"T_effective: u = {u_eff:.4f}  (= {clock_label(hours_et(ref_row, u_eff))}); formal trading close: u = 1 "
          f"(= {clock_label(hours_et(ref_row, 1.0))})")

    # ---- G3/G4 constant-kappa fits -------------------------------------------------------------
    obs_train = prepare_observations(prices, train, N_BINS)
    obs_test = prepare_observations(prices, test, N_BINS)
    kappa_eff, rmse_train_eff = fit_constant_kappa(obs_train, u_eff)
    kappa_close, rmse_train_close = fit_constant_kappa(obs_train, 1.0)
    print("\n=== G4  Constant-kappa calibration (training) ===")
    print(f"per-market p0: first observed midpoint, range [{obs_train.p0.min():.3f}, {obs_train.p0.max():.3f}], "
          f"mean {np.mean(np.unique(obs_train.p0)):.3f}")
    print(f"T = T_effective:  kappa_hat = {kappa_eff:.4f},  training RMSE = {rmse_train_eff:.4f}")
    print(f"T = formal close: kappa_hat = {kappa_close:.4f},  training RMSE = {rmse_train_close:.4f}")

    fit_train = binned_comparison(obs_train, constant_kappa_values(obs_train, kappa_eff, u_eff))
    fit_train_close = binned_comparison(obs_train, constant_kappa_values(obs_train, kappa_close, 1.0))
    print("\nTraining profile, T = T_effective (residual = model - empirical):")
    print(fit_train.assign(z=fit_train["residual"] / fit_train["empirical_se"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- G5 goodness of fit and decision (training only) ----------------------------------------
    boot = parametric_bootstrap_rmse(obs_train, kappa_eff, u_eff, N_BOOTSTRAP, ss_boot)
    pd.DataFrame({"bootstrap_training_rmse": boot}).to_csv(RESULTS / "G_bootstrap_rmse.csv", index=False)
    q_gof = float(np.quantile(boot, GOF_QUANTILE))
    p_gof = float(np.mean(boot >= rmse_train_eff))
    refine = rmse_train_eff > q_gof
    print("\n=== G5  Goodness of fit (parametric bootstrap, training) ===")
    print(f"observed training RMSE {rmse_train_eff:.4f}; under the fitted model: median {np.median(boot):.4f}, "
          f"{GOF_QUANTILE:.0%} quantile {q_gof:.4f}; p-value {p_gof:.3f}")
    print("decision: " + ("REFINE information timing (observed RMSE above the pre-specified threshold)" if refine else
                          "KEEP constant kappa (misfit within the range produced by the model's own sampling noise)"))
    if refine:
        raise SystemExit("Pre-specified rule calls for the information-clock refinement (G6), which is not "
                         "implemented in this run; see src/information_clock.py for the coherent construction.")

    # ---- G8/G9 held-out test (parameters frozen) ------------------------------------------------
    rmse_test_eff = binned_rmse(obs_test, constant_kappa_values(obs_test, kappa_eff, u_eff))
    rmse_test_close = binned_rmse(obs_test, constant_kappa_values(obs_test, kappa_close, 1.0))
    fit_test = binned_comparison(obs_test, constant_kappa_values(obs_test, kappa_eff, u_eff))
    fit_test_close = binned_comparison(obs_test, constant_kappa_values(obs_test, kappa_close, 1.0))
    lo_tr, hi_tr, _ = model_band(obs_train, kappa_eff, u_eff, ss_band_train, N_BAND)
    lo_te, hi_te, rmse_noise_test = model_band(obs_test, kappa_eff, u_eff, ss_band_test, N_BAND)
    print("\n=== G8/G9  Held-out test (10 latest markets, parameters frozen) ===")
    print(f"T = T_effective:  test RMSE = {rmse_test_eff:.4f}")
    print(f"T = formal close: test RMSE = {rmse_test_close:.4f}")
    print(f"reference: RMSE from sampling noise alone with {len(test)} markets under the fitted model: "
          f"median {np.median(rmse_noise_test):.4f}, 95% quantile {np.quantile(rmse_noise_test, 0.95):.4f}; "
          f"p-value of observed {np.mean(rmse_noise_test >= rmse_test_eff):.3f}")
    print(f"test bins inside the model's 5-95% noise band: {int(np.sum((fit_test['empirical'] >= lo_te) & (fit_test['empirical'] <= hi_te)))} of {N_BINS}")
    print(fit_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    profiles = pd.concat([
        fit_train.assign(sample="train", specification="T_effective", band_lo=lo_tr, band_hi=hi_tr),
        fit_train_close.assign(sample="train", specification="formal_close"),
        fit_test.assign(sample="test", specification="T_effective", band_lo=lo_te, band_hi=hi_te),
        fit_test_close.assign(sample="test", specification="formal_close"),
    ], ignore_index=True)
    profiles.to_csv(RESULTS / "G_bin_profiles.csv", index=False)

    # ---- G11 pricing bridge -----------------------------------------------------------------
    u_start = float(np.median(obs_train.u_start))
    u_opt_eff = u_start + RELATIVE_EXPIRY * (u_eff - u_start)
    u_opt_close = u_start + RELATIVE_EXPIRY * (1.0 - u_start)
    u_14 = normalized_time_of_clock(ref_row["open_time"], ref_row["close_time"], 14.0)
    specs = [
        ("synthetic baseline (Parts C-E)", SYNTHETIC_KAPPA, "T = 1 (synthetic)", RELATIVE_EXPIRY, "2/3 of horizon"),
        ("calibrated constant kappa, T_effective", kappa_eff, clock_label(hours_et(ref_row, u_eff)),
         float(linear_clock(u_opt_eff, u_start, u_eff)), clock_label(hours_et(ref_row, u_opt_eff))),
        ("calibrated constant kappa, formal close", kappa_close, clock_label(hours_et(ref_row, 1.0)),
         float(linear_clock(u_opt_close, u_start, 1.0)), clock_label(hours_et(ref_row, u_opt_close))),
        ("calibrated constant kappa, T_effective, expiry 14:00 ET", kappa_eff, clock_label(hours_et(ref_row, u_eff)),
         float(linear_clock(u_14, u_start, u_eff)), clock_label(14.0)),
        ("calibrated constant kappa, formal close, expiry 14:00 ET", kappa_close, clock_label(hours_et(ref_row, 1.0)),
         float(linear_clock(u_14, u_start, 1.0)), clock_label(14.0)),
    ]
    rows = []
    for (name, kappa, horizon, tau_opt, expiry), ss in zip(specs, ss_price.spawn(len(specs))):
        exact = call_price_closed_form(P0_OPTION, 1.0, tau_opt, K_OPTION, kappa)
        mc = price_european_call_mc(P0_OPTION, 1.0, tau_opt, K_OPTION, kappa, N_MC, np.random.default_rng(ss))
        rows.append({"specification": name, "kappa": kappa, "information_horizon": horizon,
                     "option_expiry": expiry, "tau_option": tau_opt,
                     "information_ratio_s": float(information_ratio_from_tau(kappa, tau_opt)),
                     "premium_closed_form": exact, "premium_mc": mc.option_price, "mc_standard_error": mc.standard_error,
                     "z_mc_vs_closed_form": (mc.option_price - exact) / mc.standard_error})
    bridge = pd.DataFrame(rows)
    bridge["ratio_to_synthetic"] = bridge["premium_closed_form"] / bridge["premium_closed_form"].iloc[0]
    bridge.to_csv(RESULTS / "G_pricing_bridge.csv", index=False)
    print("\n=== G11  Illustrative pricing bridge (p0 = 0.6, K = 0.7; NOT an observed option price) ===")
    print(bridge.drop(columns=["information_horizon"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary = pd.DataFrame([
        ("n_train_markets", len(train)), ("n_test_markets", len(test)),
        ("u_effective", u_eff), ("T_effective_clock", clock_label(hours_et(ref_row, u_eff))),
        ("kappa_constant_T_effective", kappa_eff), ("train_rmse_T_effective", rmse_train_eff),
        ("test_rmse_T_effective", rmse_test_eff),
        ("kappa_constant_formal_close", kappa_close), ("train_rmse_formal_close", rmse_train_close),
        ("test_rmse_formal_close", rmse_test_close),
        ("gof_bootstrap_q95_train_rmse", q_gof), ("gof_p_value", p_gof),
        ("refinement_selected", bool(refine)),
        ("test_noise_rmse_median", float(np.median(rmse_noise_test))),
        ("illustrative_premium_calibrated", bridge["premium_closed_form"].iloc[1]),
        ("illustrative_premium_synthetic", bridge["premium_closed_form"].iloc[0]),
    ], columns=["metric", "value"])
    summary.to_csv(RESULTS / "G_calibration_summary.csv", index=False)

    refs = {"target-day noon": normalized_time_of_clock(ref_row["open_time"], ref_row["close_time"], 12.0),
            r"$T_{eff}$ (16:00)": u_eff}
    plot_fit(fit_train, fit_train_close, lo_tr, hi_tr, boot, rmse_train_eff, q_gof, kappa_eff, kappa_close,
             rmse_train_close, refs)
    plot_comparison(fit_train, fit_test, fit_train_close, fit_test_close, (lo_tr, hi_tr), (lo_te, hi_te),
                    (rmse_train_eff, rmse_test_eff), (rmse_train_close, rmse_test_close), kappa_eff, kappa_close, refs)
    print(f"\nFigures and tables written to {RESULTS.relative_to(ROOT)}/")


def _reference_lines(ax, refs: dict) -> None:
    """Vertical reference lines; labels right-aligned left of the first line and left-aligned right of the last."""
    items = sorted(refs.items(), key=lambda kv: kv[1])
    for i, (name, u) in enumerate(items):
        ax.axvline(u, color=INK_MUTED, ls=":", lw=1.0)
        left = i == 0 and len(items) > 1
        ax.text(u, 0.995, (name + " ") if left else (" " + name), transform=ax.get_xaxis_transform(), va="top",
                ha="right" if left else "left", fontsize=7.5, color=INK_MUTED)


def _profile_panel(ax, fit, fit_close, band, rmse_eff, rmse_close, kappa_eff, kappa_close, title, refs) -> None:
    x = (fit["bin"] + 0.5) / N_BINS
    ax.fill_between(x, band[0], band[1], color=BLUE, alpha=0.15, lw=0,
                    label="5-95% range of bin means if the model were true")
    ax.plot(x, fit["model"], "-", color=BLUE, label=rf"constant $\kappa$={kappa_eff:.2f}, $T=T_{{eff}}$ (RMSE {rmse_eff:.3f})")
    ax.plot(x, fit_close["model"], "--", color=ORANGE, lw=1.4,
            label=rf"constant $\kappa$={kappa_close:.2f}, $T$ = formal close (RMSE {rmse_close:.3f})")
    ax.errorbar(x, fit["empirical"], yerr=1.96 * fit["empirical_se"], fmt="o", color=INK, ms=5, capsize=3,
                label=r"empirical mean $|Y-S_t|$ ($\pm$1.96 market-clustered s.e.)")
    _reference_lines(ax, refs)
    ax.set(xlabel="normalized time $u$ (bin centres)", ylabel=r"mean $|Y - S_t|$", ylim=(-0.02, 0.62), xlim=(0, 1),
           title=title)


def plot_fit(fit, fit_close, lo, hi, boot, rmse_obs, q_gof, kappa_eff, kappa_close, rmse_close, refs) -> None:
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.1, 0.9])
    ax_p, ax_r, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    _profile_panel(ax_p, fit, fit_close, (lo, hi), rmse_obs, rmse_close, kappa_eff, kappa_close,
                   "G5  Training markets: empirical vs fitted model", refs)
    ax_p.legend(loc="lower left", fontsize=7.2)

    x = (fit["bin"] + 0.5) / N_BINS
    ax_r.fill_between(x, -1.96 * fit["empirical_se"], 1.96 * fit["empirical_se"], color=INK_MUTED, alpha=0.15, lw=0,
                      label=r"$\pm$1.96 s.e. of empirical mean")
    ax_r.axhline(0, color=INK_MUTED, lw=1)
    ax_r.plot(x, fit["residual"], "o-", color=BLUE, label=r"$T=T_{eff}$")
    ax_r.plot(x, fit_close["residual"], "s--", color=ORANGE, ms=5, lw=1.4, label="T = formal close")
    ax_r.text(0.02, 0.97, "above 0: model too uncertain\n(learns too slowly)", transform=ax_r.transAxes,
              va="top", fontsize=8, color=INK_MUTED)
    ax_r.text(0.02, 0.03, "below 0: model too resolved (learns too quickly)", transform=ax_r.transAxes,
              va="bottom", fontsize=8, color=INK_MUTED)
    _reference_lines(ax_r, refs)
    ax_r.set(xlabel="normalized time $u$", ylabel="model - empirical", ylim=(-0.13, 0.13), xlim=(0, 1),
             title="Residuals by time bin")
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.5)

    ax_b.hist(boot, bins=25, color=BLUE, alpha=0.6, edgecolor="white")
    ax_b.axvline(q_gof, color=INK_MUTED, ls="--", lw=1.2, label=f"95% quantile {q_gof:.4f}")
    ax_b.axvline(rmse_obs, color=ORANGE, lw=2, label=f"observed {rmse_obs:.4f}")
    ax_b.set(xlabel="training RMSE (kappa refitted)", ylabel="bootstrap replicates",
             title="Goodness of fit: RMSE if the\nfitted model generated the data")
    ax_b.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "G_constant_kappa_fit.png")
    plt.close(fig)


def plot_comparison(fit_tr, fit_te, fit_tr_c, fit_te_c, band_tr, band_te, rmse_eff, rmse_close,
                    kappa_eff, kappa_close, refs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    _profile_panel(axes[0], fit_tr, fit_tr_c, band_tr, rmse_eff[0], rmse_close[0], kappa_eff, kappa_close,
                   f"Training: {int(fit_tr['n_markets'].max())} earlier markets (parameters fitted here)", refs)
    _profile_panel(axes[1], fit_te, fit_te_c, band_te, rmse_eff[1], rmse_close[1], kappa_eff, kappa_close,
                   f"Held-out test: {int(fit_te['n_markets'].max())} later markets (parameters frozen)", refs)
    for ax in axes:
        ax.legend(loc="lower left", fontsize=7.2)
    fig.suptitle("Calibrated constant-information model vs real KXHIGHNY forecast error", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "G_model_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
