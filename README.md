# Supply Chain Intelligence

> **Cuts simulated inventory cost 5.0% while raising fill rate 1.0 point, across
> 30,490 retail SKU-store series** — by choosing a service level per series
> rather than one target for the whole catalogue.
>
> 95% bootstrap CI on the cost delta: −8.5% to −1.9%.
> The baseline is *reconstructed*, not observed — [read why that matters](docs/03-assumptions-and-limitations.md#a5--the-observed-baseline-is-reconstructed-not-observed).

**Live demo:** _not yet deployed — `docker compose up` runs it locally._

---

## The business problem

A retailer holds inventory at thousands of store-SKU combinations. Hold too much
and capital sits on shelves; hold too little and sales are lost. The decision is
how much to stock at each store-SKU, and at what service level.

This project answers that end to end: it forecasts demand as a **distribution**,
converts those distributions into a replenishment policy, and simulates that
policy against reconstructed observed behaviour to put a defended cost number on
the difference.

The deliverable is not a forecast. It is a replenishment decision with a cost
attached and its assumptions written down.

## Architecture

```
sources/     M5 (Kaggle), Open-Meteo, FRED
   |
ingest/      idempotent, checksummed pulls -> parquet
   |
warehouse/   DuckDB star schema  (59.2M rows)
   |
transform/   availability filter, stockout proxy, segmentation  (46.9M-row panel)
   |
features/    point-in-time-correct feature store  (33 features, every lag >= 28d)
   |
models/      baselines -> global LightGBM -> quantiles -> calibration
   |
optimize/    (s,S) policy, expected-cost allocation, constrained LP + shadow prices
   |
app/         FastAPI + Streamlit
```

## Results

Test window d1914–1941, scored once. Full detail in [docs/04-results.md](docs/04-results.md).

### The decision

| Policy | Fill rate | Cost (28d) | vs baseline | 95% CI |
|---|---:|---:|---:|---|
| Reconstructed baseline | 94.05% | $127,930 | — | — |
| Better forecast only | 94.03% | $129,254 | +1.0% | −1.3% to +3.5% |
| Empirical safety stock, uniform level | 94.02% | $133,976 | +4.7% | +1.4% to +8.3% |
| **Per-series cost-optimal allocation** | **95.06%** | **$121,500** | **−5.0%** | **−8.5% to −1.9%** |

The recommended policy is cheaper **and** serves more demand — dominance, not a
trade-off.

**The decomposition is the interesting part.** A better forecast alone bought
nothing (CI spans zero). Replacing normal safety stock with an empirical
bootstrap at a uniform service level was actively *worse*. The entire gain came
from **deciding which series deserve the stock** — the allocation step. That is
not the result the project was hoping for, and it is the one it found.

### The forecast

| Model | CV WRMSSE | Test WRMSSE |
|---|---:|---:|
| **Global LightGBM (Tweedie)** | **0.8559** | **0.8446** |
| TSB | 0.8893 | 0.8626 |
| Croston | 0.8910 | 0.8639 |
| Moving average | 0.8950 | 0.8720 |
| Seasonal naive | 1.1563 | 1.1017 |
| Zero forecast | 1.3963 | 1.4132 |

A 3.8% CV improvement over a tuned TSB baseline. Modest, and reported as such —
the direct 28-day-lag design gives up short-horizon accuracy in exchange for a
leakage audit that is arithmetic rather than trust.

Quantiles calibrate well where the optimizer uses them: nominal 0.95 covers
0.958 empirically, nominal 0.99 covers 0.990.

### The data work

- **46,881,677-row panel** from 59.2M raw rows. The 20.8% dropped are SKU-store
  pairs the store did not carry yet — verified, not assumed: the price-NULL block
  is contiguous and leading for **100.00%** of all 30,490 series, and no row
  anywhere has positive sales with a NULL price.
- **91% of series are intermittent or lumpy** (Syntetos-Boylan). ETS and ARIMA
  are the wrong family for nearly the whole catalogue.
- **45% of revenue sits in the AZ box** — high value *and* highly erratic. This
  is the direct argument for forecasting the distribution rather than the mean.
- **3.99% of panel days are likely stockouts**, found with a per-series adaptive
  threshold rather than a fixed rule.
- **SNAP days lift FOODS +7.7%** but HOBBIES only +1.3% — an end-to-end
  validation of the calendar → store → state join that nothing was tuned to produce.

## Methods

[docs/02-methodology.md](docs/02-methodology.md) — the protocol, the model
ladder, the leakage rules, the policy derivation, and the LP formulation.

## Limitations

[docs/03-assumptions-and-limitations.md](docs/03-assumptions-and-limitations.md).
**Read it before the results table.** Ten numbered assumptions, each with its
direction of bias. The load-bearing ones:

- The baseline is reconstructed. M5 has no inventory data at all.
- Every cost parameter is assumed. The holding-to-stockout ratio (~1:26 over the
  exposure window) drives the entire answer.
- The counterfactual replays realised demand, which was itself shaped by the
  stockouts the real stores had. It therefore *understates* what a better policy
  would achieve.

## Things that went wrong, and were fixed

Written up rather than quietly patched — they are the most useful part of the
repo:

1. **The stockout rule flagged 18.3% of the panel** with a mean run of 40 days.
   That is dormancy, not stockout. Cause: a lifetime demand rate cannot see
   drift. Fixed with a trailing 56-day local rate → 3.99%, median run 8 days.
2. **The alpha sensitivity table was inert** — it recomputed thresholds over the
   already-filtered table, so every alpha above the build value returned the same
   answer. The real range is 2.37%–10.86%.
3. **The first cost comparison was rigged** — policies ran at different achieved
   service levels, so the "32% more expensive" result was measuring the service
   gap. Fixed by bisecting every policy onto the same fill rate.
4. **The first allocation overfit** — it chose each series' level by simulating on
   the realised test demand. A 39% "saving" that was hindsight. Levels now come
   from forecast-derived expected cost only.

## How to run locally

```bash
uv sync --extra dev --extra models --extra optimize --extra app
cp .env.example .env      # set SCI_DATA_DIR and SCI_M5_SOURCE

uv run sci ingest all              # M5 from local zip + Open-Meteo weather
uv run sci warehouse build         # ~4 min -> 163 MB DuckDB
uv run sci analyze                 # the five SQL analytics reports

uv run python scripts/run_model_ladder.py     # ~40 min
uv run python scripts/run_optimizer.py        # ~15 min
uv run python scripts/build_serving_bundle.py

uv run streamlit run app/dashboard.py
uv run uvicorn app.api:app --reload
```

Or containerised:

```bash
uv run python scripts/stage_deploy.py   # copies 23 MB of serving artifacts into the build context
docker compose up --build               # dashboard :8501, API :8000/docs
```

> **Not yet verified.** Docker is not installed on the machine this was built on,
> so the Dockerfile and compose file are written but have never been built or
> run. Treat them as untested until someone does.

Requires Python 3.11 or 3.12 and the M5 dataset from
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
(accept the competition rules first). Set `SCI_DATA_DIR` somewhere outside any
synced folder — the warehouse and matrices run to several GB.

Tests: `uv run pytest` (101 tests).
