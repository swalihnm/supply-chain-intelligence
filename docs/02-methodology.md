# Methodology

Written layer by layer as each is built. Weeks 2–4 sections are stubs until
those layers exist.

---

## Layer 0 — Warehouse

### Pipeline order

```
sql/ddl/         hand-written star schema (runs first, on an empty database)
   |
loads            INSERT ... SELECT straight off staging parquet
   |
sql/transform/   CTAS over loaded data: panel, zero_runs, stockout_flags, series_class
   |
sql/analytics/   pure SELECT reports, run on demand
```

Transforms are a separate stage from DDL because they are `CREATE TABLE AS
SELECT` over data that does not exist until the loads have run. Putting them in
the DDL would silently build empty tables.

### Grain

| Table | Grain | Rows |
|---|---|---:|
| `core.fact_sales` | item × store × day | 59,181,090 |
| `core.panel` | item × store × day, **available only** | 46,881,677 |
| `core.fact_price` | item × store × week | 6,841,121 |
| `core.zero_runs` | one row per consecutive-zero run | 6,613,508 |
| `core.stockout_flags` | zero runs exceeding their threshold | 254,122 |
| `core.series_class` | one row per series | 30,490 |

### Availability filtering

`core.panel` inner-joins sales to price. That join is the availability filter:
a missing weekly price means the store did not carry the SKU. Evidence for
treating it that way is in [04-results.md](04-results.md#the-availability-filter)
— it is verified against the data, not assumed.

---

## Stockout detection

The most consequential data decision in the project. Observed sales censor true
demand: a stockout day shows zero sales but had positive demand. Train on that
and forecasts fall, safety stock falls, and the optimizer under-orders.

### Why not a fixed threshold

"Zero for 7+ days = stockout" fails in both directions on this data. There are
6.6M zero runs inside availability windows with a median length of 2 days, but
demand rates vary enormously: 9.0% of series sell on most days, 7.7% sell less
than once in ten days. Seven zeros is an emergency for the first group and a
Tuesday for the second.

### The rule

Estimate p = P(sale on a given day). Under a working independence assumption, a
run of k zeros has probability (1−p)^k. Flag when that drops below alpha:

```
(1 − p)^k < alpha    <=>    k > ln(alpha) / ln(1 − p)
```

The threshold is *derived per series* rather than imposed. At alpha = 0.005 a
series selling 90% of days trips at k=3; one selling 5% of days needs k=97.

### p is local, not lifetime

The first implementation used each series' lifetime sale rate. It flagged 18.3%
of the panel across 97.3% of series with a mean run of 40.3 days — which is
dormancy and seasonality, not stockout. The constant-p assumption was the cause:
a series that sold daily in 2012 and faded by 2015 carries a high lifetime p, so
every genuine 2015 quiet stretch looked impossible.

p is now estimated from the **56 days immediately preceding each run**, which
asks the question actually worth asking: this series was selling at rate p right
up to the gap — how surprising is the gap? Series with under 28 days of prior
history fall back to the lifetime rate, recorded in `rate_basis` so the fallback
is never invisible.

Result: 3.99% of panel days flagged, median run 8 days.

### Stockout vs dormant

Runs over 28 days are labelled `dormant` rather than `stockout`. Both are
censored, but they need opposite treatment: a stockout should be **imputed**
(demand existed, we could not see it), while a dormant stretch should be
**excluded** (the product was effectively not on sale). Collapsing them would
impute phantom demand into dead SKUs and inflate every inventory number
downstream.

### Known weaknesses

- **Independence is false.** Demand is autocorrelated and day-of-week seasonal,
  so genuine quiet stretches are likelier than the geometric model implies. The
  rule stays somewhat over-eager. That is the safer direction: excluding a real
  zero costs a little information, while training on a censored zero biases
  every cost number.
- **Partial stockouts are invisible.** A shelf that emptied at noon shows
  truncated, not zero, sales. Not recoverable from this data.
- **alpha, the 56-day window, and the 28-day cutoff are choices.** Sensitivity
  to alpha spans 2.37%–10.86% and is reported in
  [04-results.md](04-results.md#sensitivity-to-alpha).

---

## Segmentation

Three orthogonal classifications, each answering a different question:

| Scheme | Question | Cutoffs |
|---|---|---|
| ABC | How much revenue is at stake? | cumulative revenue ≤80% / ≤95% / rest |
| XYZ | How predictable is it? | CV of daily units <0.5 / ≤1.0 / above |
| Syntetos-Boylan | Which model family does it need? | ADI 1.32, CV² of demand *size* 0.49 |

The SB CV² is computed on demand **size** (units on days with a sale), not on
the raw series. Mixing those up misclassifies most of the catalogue.

---

## Analytical SQL

Five reports in `sql/analytics/`, all excluding censored days:

| Script | What it establishes |
|---|---|
| `abc_xyz_segmentation.sql` | The 9-box, with a policy hint per box |
| `demand_seasonality_profile.sql` | Day-of-week, month, SNAP and event lift |
| `price_elasticity_naive.sql` | Sanity baseline only — heavily caveated |
| `stockout_proxy_detection.sql` | Censoring report + alpha sensitivity |
| `promo_lift_analysis.sql` | Promo lift vs matched controls |

### Matched controls in the promo analysis

Each promo day is compared against non-promo days of the **same series, same
weekday, same quarter**, holding item, store, weekly seasonality and coarse
season fixed. Two refinements were needed to make the estimate trustworthy:

1. **Temporary promotions are separated from permanent markdowns.** A
   trailing-only price reference cannot tell them apart — after a list-price
   cut, the next eight weeks all look promotional. A week counts as a promotion
   only if the price recovers to 97% of reference within eight weeks. That test
   looks forward, which is fine in a descriptive report run once over history
   but would be leakage in the feature layer, where promo flags must be built
   from trailing information only.

2. **Lift is exposure-weighted.** The estimator is total observed promo units
   over units expected had those same days run at their cell's base rate. An
   unweighted `avg(promo)/avg(control)` across cells is a ratio of means and
   Simpson's paradox can reverse it outright. A median per-cell lift is reported
   alongside as a robustness check; the two now agree closely.

Both arms require at least 5 observations per cell. An earlier asymmetric
version accepted single-observation promo cells — almost all of them zero, since
91% of series are intermittent — which dominated the estimate.

---

## Layer 2 — Modelling

_Week 2. Backtest protocol will be defined before any model is fit._

## Layer 3 — Optimization

_Week 3._

## Layer 4 — Serving

_Week 4._
