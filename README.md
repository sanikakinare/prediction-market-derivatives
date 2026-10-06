# prediction-market-derivatives

Pricing European options whose underlying is a binary prediction-market contract
(e.g. a Polymarket/Kalshi YES share, price `S_t ∈ [0, 1]`, resolving to `Y ∈ {0, 1}` at `T`).

## Baseline model (information-based, r = 0)

| Quantity | Definition | Code |
|---|---|---|
| Outcome | `Y ~ Bernoulli(p0)` under Q | `simulation.simulate_terminal_outcome` |
| Noise | Brownian bridge `β_tT = W_t − (t/T) W_T`, `Var = t(T−t)/T` | `simulation.simulate_brownian_bridge` |
| Information | `ξ_t = κ t Y + β_tT` | `information_model.information_signal` |
| Bayes exponent | `A_t = κT ξ_t/(T−t) − κ² t T / (2(T−t))`, `t < T` | `information_model.bayesian_exponent` |
| Price | `logit S_t = logit p0 + A_t`; `S_T = Y` | `information_model.signal_to_price`, `simulation.prices_from_signal` |
| Call price | `C_0 = E^Q[max(S_{T_option} − K, 0)]` by Monte Carlo | `option_pricing.price_european_call_mc` |
| Benchmark | `C_0 = p0 (1−K) Φ(d1) − (1−p0) K Φ(d0)` (see `src/benchmark.py`) | `benchmark.call_price_closed_form`, `benchmark.call_price_quadrature` |

## Layout

```
src/information_model.py   pure model equations (no randomness)
src/simulation.py          path simulation driven by one np.random.Generator
src/option_pricing.py      MC call pricer (bridge sampled directly at T_option), s.e., CI, convergence
src/benchmark.py           independent closed-form + quadrature prices (imports nothing from src/)
src/sensitivity.py         comparative statics: closed-form grids, MC overlay points, information ratio s
src/expiry_distribution.py exact CDF / density / quantiles / moments of S_{T_option} (imports nothing from src/)
src/kalshi_data.py         Kalshi public API download (cached), cleaning, normalized time
src/empirical_analysis.py  |Y - S_t| by time bin, prices by outcome, jump / learning-time statistics
data/raw/                  raw Kalshi API responses (events, hourly candles)
data/processed/            cleaned long-format price data
experiments/               scripts that produce figures/tables in results/
tests/                     model checks, MC-vs-benchmark, convergence, input validation
```

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python experiments/01_paths_and_validation.py
.venv/bin/python experiments/02_option_pricing.py
.venv/bin/python experiments/03_sensitivity.py
.venv/bin/python experiments/04_distributions.py
.venv/bin/python experiments/05_real_data.py   # downloads Kalshi data on first run, then uses data/raw
```
