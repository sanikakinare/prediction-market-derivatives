"""Part D: comparative statics of the call premium.

Run from the project root:
    .venv/bin/python experiments/03_sensitivity.py

Outputs (results/):
    D_sensitivity_{strike,expiry,p0,kappa}.csv   full closed-form grids
    D_mc_validation.csv                          MC points vs closed form
    D_summary.csv                                compact representative table
    D_sensitivities.png                          2x2 sensitivity figure
    D_mc_validation.png                          MC z-scores at every overlay point
    D_information_ratio.png                      kappa and T_option curves vs s
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import call_price_closed_form  # noqa: E402
from src.sensitivity import (  # noqa: E402
    BASELINE,
    call_price_information_form,
    closed_form_sensitivity,
    information_ratio,
    mc_sensitivity_points,
)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

seed = 42
n_simulations = 100_000
# CI-calibration diagnostic: many smaller independent runs at every overlay point.
n_coverage_replications = 200
n_coverage_simulations = 10_000

# Dense grids for the closed-form curves (strictly inside the admissible ranges).
GRIDS = {
    "K": np.linspace(0.01, 0.99, 99),
    "T_option": np.linspace(0.01, 0.99, 99),
    "p0": np.linspace(0.01, 0.99, 99),
    "kappa": np.linspace(0.05, 6.0, 120),
}
# Representative points for Monte Carlo overlays and the summary table.
MC_POINTS = {
    "K": [0.30, 0.50, 0.70, 0.90],
    "T_option": [0.10, 1 / 3, 2 / 3, 0.90],
    "p0": [0.20, 0.40, 0.60, 0.80],
    "kappa": [0.50, 1.50, 3.00, 4.00],
}
FILE_TAGS = {"K": "strike", "T_option": "expiry", "p0": "p0", "kappa": "kappa"}
LABELS = {"K": "strike $K$", "T_option": r"option expiry $T_{option}$",
          "p0": r"initial YES price $p_0$", "kappa": r"information flow $\kappa$"}
TITLES = {"K": "D1  Strike", "T_option": "D2  Option expiry",
          "p0": "D3  Initial market price", "kappa": "D4  Information flow"}

# ---- plot style (shared palette) ---------------------------------------------
COLOR_CURVE = "#2a78d6"
COLOR_MC = "#eb6834"
COLOR_ALT = "#1baf7a"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
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
    "lines.linewidth": 1.8,
    "legend.frameon": False,
})


def price_with(**overrides: float) -> float:
    params = dict(BASELINE)
    params.update(overrides)
    return call_price_closed_form(**params)


def monotonicity_report(name: str, curve: pd.DataFrame) -> str:
    """Classify the grid curve without imposing anything on it."""
    diffs = np.diff(curve["call_price"].to_numpy())
    tol = 1e-14
    if np.all(diffs < -tol):
        return "strictly decreasing"
    if np.all(diffs > tol):
        return "strictly increasing"
    if np.all(diffs >= -tol):
        flat = int(np.sum(np.abs(diffs) <= tol))
        return f"non-decreasing ({flat} of {diffs.size} steps flat to within {tol:g}: saturated)"
    if np.all(diffs <= tol):
        return "non-increasing"
    return f"NOT monotone: {int(np.sum(diffs > tol))} up-steps, {int(np.sum(diffs < -tol))} down-steps"


def limit_checks() -> pd.DataFrame:
    p0, K = BASELINE["p0"], BASELINE["K"]
    eps = 1e-6
    rows = [
        ("K -> 0", "p0 - K (owning the YES contract)", price_with(K=eps), p0 - eps),
        ("K -> 1", "0", price_with(K=1 - eps), 0.0),
        ("T_option -> 0", "(p0 - K)^+", price_with(T_option=eps), max(p0 - K, 0.0)),
        ("T_option -> T", "p0 (1 - K)", price_with(T_option=BASELINE["T"] * (1 - eps)), p0 * (1 - K)),
        ("kappa -> 0", "(p0 - K)^+", price_with(kappa=0.01), max(p0 - K, 0.0)),
        ("kappa large", "p0 (1 - K)", price_with(kappa=20.0), p0 * (1 - K)),
        ("kappa -> 0, K = 0.5 < p0", "(p0 - K)^+ = 0.1", price_with(kappa=0.01, K=0.5), max(p0 - 0.5, 0.0)),
    ]
    return pd.DataFrame(rows, columns=["limit", "expected", "closed_form", "expected_value"]).assign(
        abs_error=lambda d: (d["closed_form"] - d["expected_value"]).abs()
    )


def ci_coverage_diagnostic(seed_sequence: np.random.SeedSequence) -> pd.DataFrame:
    """Fraction of independent MC 95% CIs that contain the closed form, per overlay point."""
    points = [(name, v) for name in SENSITIVITY_ORDER for v in MC_POINTS[name]]
    rows = []
    for (name, v), ss in zip(points, seed_sequence.spawn(len(points))):
        rep_streams = ss.spawn(n_coverage_replications)
        runs = pd.concat([mc_sensitivity_points(name, [v], n_coverage_simulations, rs) for rs in rep_streams])
        rows.append({"parameter": name, "value": v, "coverage": runs["closed_form_in_ci"].mean(),
                     "mean_z": runs["z"].mean(), "sd_z": runs["z"].std(ddof=1)})
    return pd.DataFrame(rows)


def summary_table() -> pd.DataFrame:
    base = call_price_closed_form(**BASELINE)
    rows = []
    for name, values in MC_POINTS.items():
        for v in values:
            price = price_with(**{name: v})
            rows.append({
                "parameter": name,
                "value": round(v, 4),
                "is_baseline": bool(np.isclose(v, BASELINE[name])),
                "call_price": price,
                "change_vs_baseline": price - base,
                "pct_change_vs_baseline": 100 * (price - base) / base,
            })
    return pd.DataFrame(rows)


def plot_sensitivities(curves: dict[str, pd.DataFrame], mc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    for ax, name in zip(axes.ravel(), SENSITIVITY_ORDER):
        df = curves[name]
        x = df[name]
        ax.plot(x, df["upper_bound"], color=INK_MUTED, ls="--", lw=1.1,
                label=r"full information $p_0(1-K)$")
        ax.plot(x, df["lower_bound"], color=INK_MUTED, ls=":", lw=1.3,
                label=r"no information $(p_0-K)^+$")
        ax.plot(x, df["call_price"], color=COLOR_CURVE, label="closed form")
        pts = mc[mc["parameter"] == name]
        ax.errorbar(pts["value"], pts["mc_price"], yerr=pts["mc_price"] - pts["ci_lower"], fmt="o",
                    color=COLOR_MC, ms=6, mfc="white", mew=1.6, capsize=3, label="Monte Carlo (95% CI narrower than marker; see D_mc_validation.png)")
        ax.axvline(BASELINE[name], color=INK_MUTED, lw=0.9, alpha=0.6)
        ax.text(BASELINE[name], 1.0, " baseline", transform=ax.get_xaxis_transform(),
                color=INK_MUTED, fontsize=8, va="top")
        if name == "p0":
            ax.axvline(BASELINE["K"], color=COLOR_ALT, lw=0.9, ls="-.")
            ax.text(BASELINE["K"], 0.9, r" $p_0=K$", transform=ax.get_xaxis_transform(),
                    color=COLOR_ALT, fontsize=8, va="top")
        ax.set(title=TITLES[name], xlabel=LABELS[name], ylabel=r"call premium $C_0$")
        ax.set_ylim(bottom=-0.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(r"Call premium sensitivities (others at baseline: $p_0=0.6$, $K=0.7$, "
                 r"$T_{option}=2/3$, $T=1$, $\kappa=1.5$)", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS / "D_sensitivities.png")
    plt.close(fig)


def plot_mc_validation(mc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = [f"{p}={v:.3g}" for p, v in zip(mc["parameter"], mc["value"])]
    xpos = np.arange(len(mc))
    ax.axhspan(-1.96, 1.96, color=COLOR_CURVE, alpha=0.10, lw=0, label=r"$|z|<1.96$")
    ax.axhline(0, color=INK_MUTED, lw=1)
    ax.plot(xpos, mc["z"], "o", color=COLOR_MC, ms=7)
    ax.set_xticks(xpos, labels, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="(MC - closed form) / s.e.", ylim=(-3.5, 3.5),
           title=f"Monte Carlo vs closed form at every overlay point (N = {n_simulations:,} each)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "D_mc_validation.png")
    plt.close(fig)


def plot_information_ratio(curves: dict[str, pd.DataFrame]) -> None:
    p0, K = BASELINE["p0"], BASELINE["K"]
    s_grid = np.linspace(0.02, 8, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(s_grid, [call_price_information_form(p0, K, s) for s in s_grid], color=INK, lw=1.2,
            label=r"$C_0(s)$ with $s=\kappa\sqrt{T_{option}T/(T-T_{option})}$")
    for name, color, marker in (("kappa", COLOR_CURVE, "o"), ("T_option", COLOR_MC, "s")):
        df = curves[name][curves[name]["information_ratio"] <= 8]
        ax.plot(df["information_ratio"], df["call_price"], marker, color=color, ms=4.5, mfc="none",
                label=f"{name} grid (D{4 if name == 'kappa' else 2})")
    ax.axhline(p0 * (1 - K), color=INK_MUTED, ls="--", lw=1.1)
    ax.set(xlabel="information ratio $s$", ylabel=r"call premium $C_0$",
           title=r"$\kappa$ and $T_{option}$ act on the price only through $s$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "D_information_ratio.png")
    plt.close(fig)


SENSITIVITY_ORDER = ("K", "T_option", "p0", "kappa")


def main() -> None:
    pd.set_option("display.width", 140)
    base_price = call_price_closed_form(**BASELINE)
    print(f"Baseline: {BASELINE}")
    print(f"Baseline call premium (closed form): {base_price:.6f}")
    print(f"Baseline information ratio s = {information_ratio(BASELINE['T'], BASELINE['T_option'], BASELINE['kappa']):.4f}")

    # 1-5: closed-form curves
    curves = {}
    print("\n=== Monotonicity of the closed-form curves (dense grids) ===")
    for name in SENSITIVITY_ORDER:
        curves[name] = closed_form_sensitivity(name, GRIDS[name])
        curves[name].to_csv(RESULTS / f"D_sensitivity_{FILE_TAGS[name]}.csv", index=False)
        below = (curves[name]["call_price"] < curves[name]["lower_bound"] - 1e-12).sum()
        above = (curves[name]["call_price"] > curves[name]["upper_bound"] + 1e-12).sum()
        print(f"{name:>9}: {monotonicity_report(name, curves[name])};  "
              f"bound violations: {below} below, {above} above")

    print("\n=== Limiting values ===")
    print(limit_checks().to_string(index=False, float_format=lambda x: f"{x:.3e}" if abs(x) < 1e-3 and x else f"{x:.6f}"))

    # p0 around K: time value C - (p0 - K)^+
    dfp = curves["p0"]
    time_value = dfp["call_price"] - dfp["lower_bound"]
    delta = np.gradient(dfp["call_price"], dfp["p0"])
    print(f"\np0 sensitivity: time value C - (p0-K)^+ peaks at p0 = {dfp['p0'][time_value.idxmax()]:.2f} "
          f"(K = {BASELINE['K']}); dC/dp0 ranges over [{delta.min():.3f}, {delta.max():.3f}]")

    # 6: Monte Carlo validation at representative points
    streams = np.random.SeedSequence(seed).spawn(len(SENSITIVITY_ORDER))
    mc = pd.concat([mc_sensitivity_points(name, MC_POINTS[name], n_simulations, ss)
                    for name, ss in zip(SENSITIVITY_ORDER, streams)], ignore_index=True)
    mc.to_csv(RESULTS / "D_mc_validation.csv", index=False)
    print(f"\n=== Monte Carlo vs closed form (N = {n_simulations:,}, independent stream per point) ===")
    print(mc.drop(columns=["parameter"]).assign(parameter=mc["parameter"])
          [["parameter", "value", "mc_price", "standard_error", "ci_lower", "ci_upper",
            "closed_form", "z", "closed_form_in_ci"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    chi_sq = float((mc["z"] ** 2).sum())
    print(f"\nclosed form inside 95% CI at {int(mc['closed_form_in_ci'].sum())} of {len(mc)} points "
          f"(expected about {0.95 * len(mc):.1f})")
    print(f"joint check: sum z^2 = {chi_sq:.2f} on {len(mc)} d.o.f., p-value = {chi2.sf(chi_sq, len(mc)):.3f}")

    coverage = ci_coverage_diagnostic(np.random.SeedSequence([seed, 1]))
    coverage.to_csv(RESULTS / "D_ci_coverage.csv", index=False)
    pooled = coverage["coverage"].mean()
    n_total = n_coverage_replications * len(coverage)
    print(f"\n=== CI calibration: {n_coverage_replications} independent runs of N = {n_coverage_simulations:,} "
          f"at each of the {len(coverage)} points ===")
    print(coverage.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"pooled coverage = {pooled:.4f} over {n_total:,} intervals "
          f"(nominal 0.95, s.e. {np.sqrt(0.95 * 0.05 / n_total):.4f}); "
          f"per-point s.e. {np.sqrt(0.95 * 0.05 / n_coverage_replications):.3f}")

    # summary
    summary = summary_table()
    summary.to_csv(RESULTS / "D_summary.csv", index=False)
    print("\n=== Summary (closed form) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 7: figures
    plot_sensitivities(curves, mc)
    plot_mc_validation(mc)
    plot_information_ratio(curves)
    print(f"\nFigures and tables written to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
