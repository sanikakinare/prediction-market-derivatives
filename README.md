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

## Layout

```
src/information_model.py   pure model equations (no randomness)
src/simulation.py          path simulation driven by one np.random.Generator
experiments/               scripts that produce figures/tables in results/
tests/test_model.py        pathwise and distributional checks
```

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python experiments/01_paths_and_validation.py
```
