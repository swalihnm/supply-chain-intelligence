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

### The protocol came first

`evaluation/protocol.py` contains no modelling code and imports nothing that
fits anything. It was written before any model existed, and every other module
imports the folds from it rather than defining its own.

- Rolling-origin CV, 5 folds, 28-day horizon each
- Training for a fold ends the day before its validation window opens — no gap, no overlap
- The final 28 days (d1914–1941) are a **locked test set**, scored once
- Validation windows are disjoint, so no observation is scored twice

`tests/test_protocol.py` asserts all of it. The point is that the rules cannot
be quietly adjusted after seeing a result, which is the easiest way to fool
yourself in a forecasting project.

### Metrics, and why three

| Metric | Question it answers |
|---|---|
| WRMSSE | The M5 official metric. Scale-free and dollar-weighted, so being wrong about a high-revenue SKU costs more. |
| MASE | Scale-free absolute error. Reported alongside because squared error on intermittent data is dominated by a handful of spike days. |
| Pinball | The one that matters downstream. The optimizer consumes quantiles, so a great point forecast with badly calibrated tails is useless to it. |

All are NaN-aware: a day outside a series' availability window is skipped, not
scored as zero error.

### The ladder

1. **Baselines** — seasonal naive, moving average, Croston, TSB, and a zero
   forecast. Tuned properly and kept to the end, because a gain over a bad
   baseline is not a gain. The zero forecast earns its place: on absolute-error
   terms it is genuinely competitive, which is a fact about the metric worth
   knowing.
2. **Global LightGBM** — one Tweedie model across all 30,490 series.

Tweedie rather than squared error because the target is a zero-inflated
non-negative count: 68% of observed days are zero, and a compound
Poisson-gamma puts a point mass there.

### Why one global model instead of 30,490 local ones

- Data per series is thin and mostly zero. A local model for an intermittent
  SKU-store fits a handful of non-zero points and overfits them.
- Day-of-week, SNAP, holiday and price effects are properties of the category
  and store, not of one SKU. Estimating them 30,490 times separately gets a
  noisier answer every time.
- New series work immediately. With a median launch at d=160 and half the
  catalogue starting after d=100, this is not a corner case here.
- One artifact to train, validate, ship and monitor.

The cost is that a genuinely idiosyncratic series gets pulled toward the crowd.
The error decomposition by Syntetos-Boylan class makes that visible.

### Leakage: a direct multi-step design

Every lag and rolling statistic is offset by at least 28 days, so a feature for
target day *t* reads only day *t−28* or earlier. That costs some short-horizon
accuracy — the model never sees yesterday — but it removes recursion entirely
and makes the leakage audit **arithmetic rather than trust**.
`tests/test_features.py` checks the lag values against the source matrix
directly.

### The weather decision

Feeding actual future weather into a 28-day forecast is leakage, and the kind
that quietly inflates a headline. Nobody has a usable 28-day forecast of daily
rainfall.

So the model gets **climatology** — the historical average for that state and
day-of-year, computed only from days strictly before the forecast origin. That
is genuinely knowable in advance; it is just seasonality. One actual-weather
feature survives: the anomaly at *t−28*, which is observed by forecast time.

This is weaker than real weather data would be. That is the honest price of not
cheating, and a test asserts no other weather feature can appear.

### Quantiles

Two routes, both produced and compared:

1. **Direct quantile regression** — one LightGBM per level. Independently
   fitted levels can cross; they are sorted per row afterwards and the crossing
   rate is reported rather than hidden.
2. **Split conformal** — distribution-free calibration of the point model's
   residuals, computed within series class so an intermittent series is not
   calibrated using a smooth series' errors.

The interesting question is not which has the better pinball loss but whether
either is **calibrated**. A 0.95 quantile that only covers 88% of outcomes makes
the optimizer buy a service level it will not achieve in the store.

### Hierarchical reconciliation

**Why it matters to the business, not just statistically.** Forecasts get
consumed at different altitudes: a store orders at SKU level, a category buyer
plans at department level, finance plans at state and total. Made
independently, those numbers do not add up — and then three people hold three
different plans for the same business, and any disagreement becomes an argument
about whose spreadsheet is right rather than about what to do. Reconciliation
produces **one** set of numbers, coherent by construction.

The M5 structure has 12 levels and **42,840 nodes over 30,490 bottom series**:

| Level | Nodes | | Level | Nodes |
|---|---:|---|---|---:|
| total | 1 | | store × category | 30 |
| state | 3 | | store × department | 70 |
| store | 10 | | item | 3,049 |
| category | 3 | | item × state | 9,147 |
| department | 7 | | item × store | 30,490 |
| state × category | 9 | | | |
| state × department | 21 | | | |

This is a **grouped** structure, not a strict tree: `item` aggregates across
stores while `store` aggregates across items, and neither nests in the other.
That rules out the simple recursive tree algorithms and forces a linear solve.

#### Base forecasts are made independently, on purpose

- **Bottom level** — the global LightGBM from the model ladder
- **Levels 1–11** — AutoETS per node

The split is not arbitrary. ETS on an intermittent SKU-store series fits noise,
which the baseline table demonstrates. But summed over a store or a category,
demand is smooth, weekly-seasonal and has no zeros — ETS's home ground. Each
method is used where its assumptions hold.

#### The solve, and why it fits in memory

With summing matrix **S** and base forecasts ŷ, the reconciled bottom is

```
b̃ = (S' W⁻¹ S)⁻¹ S' W⁻¹ ŷ
```

`S' W⁻¹ S` is 30,490 × 30,490 **and dense** — every pair of bottom series
co-occurs at the `total` node, so no sparsity survives. Materialising it needs
about 7.4 GB, on a machine with 7.3 GB of RAM.

So it is never materialised. Conjugate gradients needs only the matrix's action
on a vector:

```
(S' W⁻¹ S) v  =  S' ( W⁻¹ ( S v ) )
```

— two sparse mat-vecs against a matrix with 365,880 non-zeros (0.028% dense).
The system is symmetric positive definite, and it is Jacobi-preconditioned with
the exact diagonal, which *is* cheap to compute. A problem that does not fit in
memory becomes a few seconds of sparse arithmetic.

`tests/test_reconcile.py` verifies the matrix-free solver against the dense
closed form on a small hierarchy where the textbook formula can be computed
directly — that is what makes the fast path trustworthy.

Weight choices: `ols` (W = I), `wls_struct` (W = row sums of S), `wls_var`
(W = in-sample MSE). Full MinT with a shrinkage covariance is **not**
implemented: it needs the 42,840² error covariance, about 14 GB dense, and the
literature finds `wls_var` is usually close to it in practice.

---

## Layer 3 — Optimization

### The policy

Periodic review every T=7 days. At a review, if inventory **position** (on hand
plus on order) has fallen to the reorder point *s*, order up to *S*.

```
s = quantile_q( demand over L )       protects until an order placed now arrives
S = quantile_q( demand over T + L )   protects until the order AFTER that arrives
```

S covers T+L, not L. That is the classic mistake in periodic-review systems:
after ordering you wait L days for delivery **and** another T days before you
can order again, so sizing S on L alone under-stocks by about a review period of
demand every cycle.

### Why not z·σ·√L

The textbook safety-stock formula assumes normal, i.i.d. demand and a constant
lead time. All three are false here: 72.6% of series are intermittent and
zero-inflated, demand is autocorrelated and day-of-week seasonal, and lead time
is a distribution. Applied to an intermittent series the formula produces
negative reorder points at low service levels and badly wrong ones in the tail.

Instead, lead-time demand is simulated by **scaled block bootstrap**: draw a
lead time, draw a random start in the series' own clean history, take that many
*contiguous* days and sum, scale to the forecast level, repeat.

Contiguous blocks are the point. Sampling individual days independently destroys
the autocorrelation and understates the variance of the sum — exactly the error
√L makes. Censored days inside a block are imputed with the series' clean mean,
never zero-filled, or the censoring bias removed in week 1 would walk straight
back in at the last step.

The normal approximation is still implemented — as the comparison arm, so the
results can *quantify* what assuming normality costs rather than assert it.

### The allocation, and the overfitting it had to avoid

Choosing a service level per series is the real decision. Two corrections were
needed to make it honest:

**1. Comparisons must be at matched service.** A service-level *parameter* and
an achieved *fill rate* are different quantities, and the normal approximation
and the empirical bootstrap convert one into the other very differently on
intermittent demand. A first version ran the baseline at whatever parameter
reproduced observed behaviour and the others at a fixed 0.95, then compared
cost — and reported the optimized policy as 32% more expensive while it was
serving 2.8 points less demand. Every policy is now bisected onto the same
achieved fill rate.

**2. Levels must be chosen on expected cost, not realised cost.** The first
allocation simulated every candidate level on the realised test demand and took
the per-series minimum. That is selection on the outcome: each series gets the
level that happened to work best on the very days the result is reported over.
It produced a 39% "saving" that was mostly hindsight.
`optimize/expected_cost.py` now derives cost from the forecast's lead-time
demand distribution alone; realised demand is touched exactly once, to evaluate
the chosen allocation.

### Constrained allocation and shadow prices

A multiple-choice knapsack: each series takes one level from K candidates,
subject to per-store capacity and a total budget.

As an integer program that is 30,490 × 8 = 243,920 binaries, which CBC will not
close on a laptop. The **LP relaxation is near-exact here**, and not as a
concession: a basic feasible solution has at most as many basic variables as
rows, and the assignment constraints force one unit of mass per series, so only
the *side* constraints can split a series. With 10 store rows plus a budget row,
at most 11 of 30,490 series can be fractional. Measured: 10.

Those are rounded toward the **smaller footprint**, not the cheaper level, so a
binding capacity row cannot end up over its limit — rounding toward cost pushed
utilisation to 100.2% in testing, which is an infeasible answer.

The relaxation also yields what the integer program would not: **duals**. The
dual on a capacity row is what one more unit of shelf space at that store is
worth in avoided cost per period — which turns "we are short on space" into
"space at this store is worth $X and at that one is worth nothing".

---

## Layer 4 — Serving

FastAPI + Streamlit over a compact serving bundle. The full panel matrices are
~500 MB in memory, which is the wrong thing to put in a container that only
answers questions about the forecast horizon; `scripts/build_serving_bundle.py`
reduces it to ~30 MB of numpy arrays.

That is small enough that `POST /simulate` runs a **real** simulation over all
30,490 series on each call rather than interpolating a precomputed table.
