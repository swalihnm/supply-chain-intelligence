# Supply Chain Intelligence — the whole project, explained

A single narrative walkthrough: what was built, why each decision was made, what
the numbers came out as, and what is wrong with them.

Every figure here was read back out of the artifacts in `$SCI_DATA_DIR/artifacts/`
rather than from memory. Nothing is rounded in a flattering direction.

**Contents**

1. [The question](#1-the-question)
2. [The data](#2-the-data)
3. [The warehouse](#3-the-warehouse)
4. [The discovery that shaped everything: availability](#4-the-discovery-that-shaped-everything-availability)
5. [The hardest data problem: stockout censoring](#5-the-hardest-data-problem-stockout-censoring)
6. [What the catalogue actually looks like](#6-what-the-catalogue-actually-looks-like)
7. [Forecasting: the protocol came first](#7-forecasting-the-protocol-came-first)
8. [Forecasting: the model ladder](#8-forecasting-the-model-ladder)
9. [Forecasting: quantiles and calibration](#9-forecasting-quantiles-and-calibration)
10. [Hierarchical reconciliation](#10-hierarchical-reconciliation)
11. [From forecast to decision](#11-from-forecast-to-decision)
12. [The headline result](#12-the-headline-result)
13. [The service-level frontier](#13-the-service-level-frontier)
14. [Constraints and shadow prices](#14-constraints-and-shadow-prices)
15. [Serving it](#15-serving-it)
16. [What is wrong with all of this](#16-what-is-wrong-with-all-of-this)
17. [Six things that went wrong during the build](#17-six-things-that-went-wrong-during-the-build)
18. [How to run it](#18-how-to-run-it)

---

## 1. The question

A retailer holds inventory at thousands of store-SKU combinations. Hold too
much and capital sits on shelves and eventually gets marked down. Hold too
little and sales walk out of the door.

> **Given a service level target, how much inventory should we hold at each
> store-SKU, and what does that cost versus what we currently do?**

The deliverable is deliberately *not* a forecast. A forecast is an input. The
deliverable is a **replenishment decision with a defended cost number attached**
— and, just as importantly, a written account of every assumption that number
rests on.

That framing drives the whole architecture. The forecast layer exists to feed
the optimizer, which is why it produces distributions rather than point
estimates, and why it is judged on pinball loss and calibration rather than on
accuracy alone.

---

## 2. The data

Four independent sources, deliberately not one flat CSV.

| Source | What it gives | Grain | Status |
|---|---|---|---|
| **M5 / Walmart** (Kaggle) | Units sold, prices, calendar, SNAP + event flags | item × store × day | 59,181,090 rows |
| **Open-Meteo** | Daily temperature, precipitation, snowfall, wind | state × day | 6,210 rows |
| **FRED** | CPI, gas price, unemployment | series × native freq | client written, dormant (no key) |
| **Google Trends** | Category search interest | category × state × week | deliberately skipped |

### The spine

M5 covers **30,490 SKU-store series × 1,941 days** — 3,049 items across 10
stores in California (4), Texas (3) and Wisconsin (3), from 2011-01-29 to
2016-05-22.

It arrives in **wide** form: one row per series, 1,941 columns named `d_1` …
`d_1941`. Unpivoting that to `(item_id, store_id, day, units)` produces 59.2M
rows, which is too large to hold in memory on this machine while also doing
anything useful. It runs through the Polars streaming engine and sinks straight
to parquet — about 3 seconds.

### Weather, and an honest approximation

M5 discloses each store's **state** but not its city. Weather is pulled for one
representative metro per state (Los Angeles, Houston, Madison) and joined on
`state_id`.

This is an approximation and it is labelled as one. A CA_1 store in San Diego
and a CA_3 store in Sacramento get identical weather, so genuine within-state
signal is averaged away. That makes the weather features *weaker* than they
would be with true store locations — it does not make them misleading. The
alternative, inventing store cities, would look more precise while being
fabricated.

### Provenance

Every ingest step records the SHA-256 of its input and the row count of its
output in a manifest. Re-running a step whose input has not changed is a no-op.
That is what makes the pipeline cheap to iterate on and gives every table an
auditable trail back to source bytes.

---

## 3. The warehouse

A hand-written DuckDB star schema. No ORM generated it.

```
core.dim_date      1,969 rows    d -> real date, SNAP flags, events
core.dim_item      3,049 rows    item -> department -> category
core.dim_store        10 rows    store -> state
core.fact_sales   59,181,090     item × store × day   (observed SALES, not demand)
core.fact_price    6,841,121     item × store × week
core.fact_weather      6,210     state × day
```

Then a transform layer builds what everything downstream actually reads:

```
core.panel          46,881,677   the analysis panel (see §4)
core.zero_runs       6,613,508   every consecutive-zero run, with its baseline
core.stockout_flags    254,122   the subset judged to be censoring (see §5)
core.series_class       30,490   ABC, XYZ and Syntetos-Boylan per series
```

The whole thing rebuilds from the raw zip in about four minutes and occupies
roughly **1.9 GB** on disk — of which about 1.1 GB is live data and the rest is
free blocks. DuckDB does not return pages to the OS after a `CREATE OR REPLACE`,
so the file grows across repeated rebuilds without the data growing.

> An earlier draft of this document said 163 MB. That figure was measured at the
> end of week 1, before the transform layer added `core.panel` (46.9M rows) and
> `core.zero_runs` (6.6M rows) — 53M rows that were never reflected back into the
> number. Corrected here from `PRAGMA database_size`: 7,527 total blocks, 4,101
> used, 3,426 free, at 256 KB per block.

### One engineering decision worth defending

**`core.fact_sales` has no PRIMARY KEY.** DuckDB materialises an ART index for
a PK, and a three-column index over 59.2M rows costs real memory on a laptop
while buying nothing — every access path to that table is a full or filtered
scan. The uniqueness constraint on `(item_id, store_id, d)` is still enforced,
by `tests/test_warehouse.py` rather than by the insert path. The constraint is
checked; it just isn't paid for on every write.

---

## 4. The discovery that shaped everything: availability

**20.78% of the raw rows have no `sell_price`.**

The lazy reading is "missing data, fill it in". The correct reading turned out
to be something else entirely, and three checks against the data — not
assumptions — establish it:

1. The price-NULL block is **contiguous and leading for 100.00%** of all 30,490
   series. Not most. All of them.
2. **Zero** rows exist anywhere with `units > 0` and a NULL price.
3. The median gap between a series' first priced day and its first sale is
   **2 days** (p95 = 6).

So a missing price does not mean a lost price. It means **the store did not
carry that SKU yet**. Those rows are structural zeros — the product did not
exist there.

This matters enormously. The median first-priced day is **d=160**, and **15,752
of 30,490 series (51.7%) do not start until after d=100**. Training on those
zeros would teach every model that roughly a fifth of demand is zero when the
product simply wasn't on the shelf, dragging forecasts — and therefore safety
stock, and therefore the whole cost answer — downward.

`core.panel` is the inner join of sales to price. That join *is* the
availability filter, and it is the single most important line in the SQL.

---

## 5. The hardest data problem: stockout censoring

`fact_sales.units` records what was **sold**, not what was **demanded**. On a
day a store ran out, observed sales are zero while true demand was positive.

Train on that and the model learns to forecast *supply failures*. Forecasts
fall, safety stock falls, the optimizer under-orders, and the system converges
on being reliably out of stock. This is the most consequential data-quality
issue in the project.

### Why a fixed threshold fails

The obvious rule — "zero for 7+ consecutive days = stockout" — is wrong in both
directions, and the data shows exactly why. Inside availability windows there
are **6.6M zero runs with a median length of 2 days**, but demand rates vary
enormously: 9.0% of series sell on most days, while 7.7% sell less than once in
ten days.

For a fast mover, seven straight zeros is a screaming anomaly. For a slow mover,
seven straight zeros is a Tuesday.

### The rule used instead

Estimate the daily probability of a sale, *p*. Under a working assumption of
day-to-day independence, a run of *k* consecutive zeros has probability
(1−p)^k. Flag the run when that drops below a significance level α:

```
(1 − p)^k < α     ⟺     k > ln(α) / ln(1 − p)
```

The threshold is **derived per series rather than imposed**. At α = 0.005 a
series selling 90% of days trips at k=3; one selling 5% of days needs k=97.
Same statistical statement, adapted to each series' own baseline.

### The correction that mattered

The first implementation used each series' **lifetime** sale rate. It flagged
**18.3% of the panel across 97.3% of series, with a mean run of 40.3 days**.

A 40-day stockout is not a stockout. Replenishment cycles are days to weeks; a
gap that long is a dead SKU or a seasonal product out of season. The rule was
measuring dormancy and calling it censoring.

The cause was the constant-*p* assumption. A series that sold daily in 2012 and
faded by 2015 carries a high lifetime *p*, so every genuine 2015 quiet stretch
looked impossible. **Demand rate drifts; a lifetime average cannot see that.**

*p* is now estimated from the **56 days immediately preceding each run**, which
asks the question actually worth asking: this series was selling at rate *p*
right up to the gap — how surprising is the gap?

### The result

| Censoring type | Runs | Days | % of panel | Series affected | Median run |
|---|---:|---:|---:|---:|---:|
| **stockout** (impute) | 188,806 | 1,868,039 | **3.99%** | 27,215 | 8 days |
| **dormant** (exclude) | 65,316 | 6,689,647 | 14.27% | 23,785 | 74 days |

3.99% is a plausible grocery out-of-stock rate, and the *direction* of variation
is right too: FOODS stocks out more than HOBBIES (4.45% vs 3.33%), A-items more
than C-items (4.52% vs 2.82%). Fast movers run out. Nothing was tuned to make
that come out.

Runs over 28 days are labelled **dormant** rather than stockout. Both are
censored, but they need opposite treatment: a stockout should be *imputed*
(demand existed, we could not see it), a dormant stretch *excluded* (the product
was effectively not on sale). Collapsing them would impute phantom demand into
dead SKUs and inflate every inventory number downstream.

### And the honest caveat

α = 0.005 is a choice, not a measurement. Across the plausible range:

| α | Stockout days | % of panel |
|---:|---:|---:|
| 0.001 | 1,108,851 | 2.37% |
| **0.005** (used) | 1,868,039 | **3.99%** |
| 0.010 | 2,425,736 | 5.17% |
| 0.050 | 5,091,514 | 10.86% |

**A 4.6× swing.** Every downstream cost number inherits that uncertainty.

---

## 6. What the catalogue actually looks like

### Syntetos-Boylan classification

Computed on demand *size* (units on days with a sale), not the raw series —
mixing those up silently misclassifies most of a catalogue.

| Class | Series | % series | % revenue |
|---|---:|---:|---:|
| intermittent | 22,143 | 72.6% | 51.6% |
| lumpy | 5,612 | 18.4% | 16.6% |
| smooth | 1,880 | 6.2% | 24.2% |
| erratic | 855 | 2.8% | 7.7% |

**91% of series are intermittent or lumpy.** Fitting ETS or ARIMA to those is a
category error — they need Croston or TSB. This single table is why the model
ladder routes by series class instead of running one method everywhere.

### ABC × XYZ

| Box | Series | % series | % revenue | % intermittent/lumpy |
|---|---:|---:|---:|---:|
| AX | 83 | 0.3% | 3.1% | 0% |
| AY | 2,731 | 9.0% | 31.7% | 22% |
| **AZ** | **8,407** | **27.6%** | **45.1%** | **96%** |
| BY | 269 | 0.9% | 0.5% | 48% |
| BZ | 9,157 | 30.0% | 14.5% | 99.6% |
| CY | 48 | 0.2% | 0.0% | 48% |
| CZ | 9,795 | 32.1% | 5.0% | 100% |

**The AZ cell is the entire problem in one box.** 45% of revenue sits in series
that are simultaneously high-value and highly erratic, and 96% of them are
intermittent. A point forecast on an AZ series is nearly meaningless. That is
the direct, data-driven argument for probabilistic forecasting — not a
preference, a necessity.

Revenue is also less concentrated than the usual Pareto folklore: it takes
**36.8% of series to reach 80% of revenue**, and the top 20% carry 63.7%.

### Seasonality — and a free validation

Day-of-week lift against each category's own mean:

| Category | Sat | Sun | Mon | Tue | Wed | Thu | Fri |
|---|---:|---:|---:|---:|---:|---:|---:|
| FOODS | +18.7% | +19.9% | −3.4% | −10.6% | −11.9% | −11.5% | −1.3% |
| HOBBIES | +22.5% | +11.2% | −6.8% | −10.8% | −9.9% | −10.0% | +3.7% |
| HOUSEHOLD | +25.6% | +21.5% | −5.6% | −13.8% | −14.8% | −13.9% | +0.9% |

SNAP benefit days — the cleanest check that the joins are correct:

| Category | Lift on SNAP days |
|---|---:|
| **FOODS** | **+7.7%** |
| HOUSEHOLD | +1.9% |
| HOBBIES | +1.3% |

SNAP is *food* assistance. FOODS should respond and the other two should barely
move. They do. Nothing in the pipeline was tuned to produce this, which makes it
a genuine end-to-end validation of the calendar → store → state join.

### Price — with heavy caveats

Naive log-log elasticity: FOODS −0.432, HOUSEHOLD −0.362, HOBBIES −0.262. Correct
sign everywhere, weak fit everywhere (R² 0.09–0.16) — exactly what an
uncontrolled fit on retail data should produce. **This is not causal.** Prices
are set in *response* to expected demand, no controls are applied, and dropping
zero-sale days selects out precisely the low-demand days a high price would cause.

Promotion lift, against matched controls (same series, same weekday, same
quarter), separating temporary promotions from permanent markdowns:

| Category | Discount | Lift |
|---|---|---:|
| FOODS | 20–35% | **+103%** |
| FOODS | 5–10% | **−27%** |
| FOODS | 10–20% | **−30%** |
| HOUSEHOLD | 10–20% | −56% |

Deep discounts lift demand a lot. **Shallow discounts associate with *lower*
demand** — and that survives both a volume-weighted and an unweighted per-cell
estimator, so it is not an aggregation artifact.

The likely explanation is endogeneity, not a real negative price response:
retailers apply modest markdowns *reactively*, to products already slowing down.
Matched controls hold item, store, weekday and quarter fixed, but they cannot
hold fixed the retailer's private reason for discounting. **Do not read an
uplift factor off that table into the optimizer.**

---

## 7. Forecasting: the protocol came first

`evaluation/protocol.py` contains no modelling code and imports nothing that
fits anything. It was written **before any model existed**, and every other
module imports the folds from it rather than defining its own.

- Rolling-origin CV, **5 folds, 28-day horizon** each
- Training for a fold ends the day *before* its validation window opens — no gap, no overlap
- The final 28 days (d1914–1941) are a **locked test set**, scored exactly once
- Validation windows are disjoint, so no observation is scored twice

`tests/test_protocol.py` asserts all of it. The point is that the rules cannot be
quietly adjusted after seeing a result, which is the easiest way to fool yourself
in a forecasting project.

### Three metrics, because they answer different questions

| Metric | Question |
|---|---|
| **WRMSSE** | The M5 official metric. Scale-free and dollar-weighted, so being wrong about a high-revenue SKU costs more. |
| **MASE** | Scale-free absolute error. Reported alongside because squared error on intermittent data is dominated by a handful of spike days. |
| **Pinball** | The one that matters downstream. The optimizer consumes quantiles, so a great point forecast with bad tails is useless to it. |

All are NaN-aware: a day outside a series' availability window is skipped, never
scored as zero error.

### Leakage, proved by arithmetic

Every lag and rolling statistic is offset by **at least 28 days**, so a feature
for target day *t* reads only day *t−28* or earlier. That is a *direct*
multi-step design rather than a recursive one.

It costs accuracy at short lead times — the model never sees yesterday — and buys
something worth more: the leakage audit becomes **index arithmetic rather than
trust**. `tests/test_features.py` checks lag values against the source matrix
directly.

### The weather trap

Feeding actual future weather into a 28-day forecast is leakage, and the kind
that quietly inflates a headline number. **Nobody has a usable 28-day forecast of
daily rainfall.**

So the model gets **climatology** instead: the historical average for that state
and day-of-year, computed only from days strictly before the forecast origin.
That is genuinely knowable in advance — it is just seasonality. One actual-weather
feature survives: the anomaly at *t−28*, which is observed by forecast time.

This is weaker than real weather data would be. That is the honest price of not
cheating, and a test asserts no other weather feature can appear.

---

## 8. Forecasting: the model ladder

Rolling-origin CV, 5 folds. The test fold was scored once.

| Model | CV WRMSSE | CV sd | CV MASE | Test WRMSSE |
|---|---:|---:|---:|---:|
| **Global LightGBM (Tweedie)** | **0.8559** | 0.017 | 1.215 | **0.8446** |
| TSB | 0.8893 | 0.015 | 1.217 | 0.8626 |
| Croston | 0.8910 | 0.016 | 1.196 | 0.8639 |
| Moving average | 0.8950 | 0.013 | 1.073 | 0.8720 |
| Seasonal Croston | 0.9162 | 0.022 | 1.213 | 0.8824 |
| Seasonal naive | 1.1563 | 0.035 | 1.208 | 1.1017 |
| Naive | 1.2359 | 0.046 | 1.750 | 1.1766 |
| Zero forecast | 1.3963 | 0.040 | **0.986** | 1.4132 |

**A 3.8% CV improvement over a tuned TSB baseline.** Modest, and reported as
such. A recursive design using lag-1 would score better and be far harder to
defend.

### Why one global model instead of 30,490 local ones

- **Data per series is thin and mostly zero** — 68% of observed days. A local
  model for an intermittent SKU-store fits a handful of non-zero points and
  overfits them.
- **Effects are shared.** Day-of-week, SNAP, holiday and price-response patterns
  are properties of the category and store, not of one SKU. Estimating them
  30,490 separate times gets a noisier answer every time.
- **New series work immediately** — which matters when the median launch is d=160
  and half the catalogue starts after d=100.
- **It is operable.** One artifact to train, validate, ship and monitor.

The cost is that a genuinely idiosyncratic series gets pulled toward the crowd.
The error decomposition below is what makes that visible rather than assumed.

Tweedie rather than squared error, because the target is a zero-inflated
non-negative count and a compound Poisson-gamma puts a point mass at zero.

### Three things in that table worth pausing on

**The zero forecast has the best MASE of any model (0.986) and the worst WRMSSE
by a distance.** That is not a bug. With 68% of days at zero, "always predict
zero" minimises absolute error while missing every spike that squared error
punishes. It is a clean demonstration that **metric choice is a modelling
decision**, and it is why the optimizer is driven by pinball loss rather than
either of these.

**Seasonal naive is far worse than a plain moving average** (1.156 vs 0.895).
Repeating last week's exact pattern on a sparse series propagates noise; on
intermittent data smoothing is worth more than seasonality.

**Seasonal Croston — a day-of-week reshaping of the Croston rate — made things
worse** (0.916 vs 0.891). A negative result, kept in the table rather than
dropped.

### Error by demand class

Mean RMSSE, CV folds:

| Model | intermittent | lumpy | smooth | erratic |
|---|---:|---:|---:|---:|
| moving average | **0.734** | **0.714** | 0.779 | 0.738 |
| global LightGBM | 0.759 | 0.750 | **0.756** | 0.738 |
| croston | 0.758 | 0.747 | 0.787 | 0.761 |
| TSB | 0.762 | 0.746 | 0.785 | 0.754 |
| zero | 0.844 | 0.857 | 1.387 | 1.120 |

A nuance worth knowing: on *unweighted* RMSSE the moving average edges out the
global model on intermittent and lumpy series. On **dollar-weighted WRMSSE the
global model wins every class** — it is better where the money is, which is what
the business cares about, and it is also a reminder that a single headline number
can hide a real reversal.

---

## 9. Forecasting: quantiles and calibration

The optimizer does not order the expected demand. It orders enough to meet a
service level — a statement about the **tail** of lead-time demand. Safety stock
*is* a quantile. So the forecast layer must produce a distribution.

Two routes were built and compared:

1. **Direct quantile regression** — one LightGBM per level, with the quantile
   objective. Independently fitted levels can cross (2.7% of cells did), so they
   are sorted per row afterwards and the crossing rate is reported rather than
   hidden.
2. **Split conformal** — distribution-free calibration of the point model's
   residuals, computed within series class so an intermittent series isn't
   calibrated using a smooth series' errors.

### The result, on the locked test fold

| Nominal | Empirical (direct) | Gap | Empirical (conformal) |
|---:|---:|---:|---:|
| 0.50 | 0.638 | +0.138 | 0.630 |
| **0.90** | **0.9007** | **+0.0007** | 0.8887 |
| **0.95** | **0.9505** | **+0.0005** | 0.9434 |
| **0.99** | **0.9885** | −0.0015 | 0.9882 |

**Gaps under 0.002 exactly where the optimizer reads the distribution.** This
matters more than the WRMSSE: a 0.95 quantile that only covered 0.88 would make
the optimizer buy a service level it never achieves in the store.

Direct quantile regression beat conformal at 0.90 and 0.95, so the direct models
are used.

**The 0.50 quantile is badly calibrated (0.638 vs 0.50), and that is a property
of the data, not a defect.** Coverage is P(actual ≤ predicted), and with 68% of
days at zero the median of an intermittent series *is* zero — so "actual ≤ 0" is
true on every zero day. The statistic saturates against the point mass. The upper
quantiles, the only ones the policy uses, are unaffected.

---

## 10. Hierarchical reconciliation

### The problem, made concrete

Forecasts are consumed at different altitudes. A store orders at SKU level; a
category buyer plans at department level; finance plans at state and total. Made
independently, those numbers do not add up.

Here, the **total-level forecast says 1,147,983 units while the bottom level sums
to 1,241,178** — **8.1% apart**, with a largest single-node disagreement of 7,934
units. Two teams, the same 28 days, two different plans, and any dispute becomes
an argument about whose spreadsheet is right.

### The structure

The full published M5 hierarchy: **42,840 nodes over 30,490 bottom series**
across 12 levels (total, state, store, category, department, state×category,
state×department, store×category, store×department, item, item×state,
item×store).

It is a **grouped** structure, not a strict tree: `item` aggregates across stores
while `store` aggregates across items, and neither nests in the other. That rules
out the simple recursive tree algorithms and forces a linear solve.

### Why it fits in memory at all

With summing matrix **S** and base forecasts ŷ, the reconciled bottom is
`b̃ = (S'W⁻¹S)⁻¹ S'W⁻¹ŷ`.

`S'W⁻¹S` is 30,490 × 30,490 **and dense** — every pair of bottom series co-occurs
at the `total` node, so no sparsity survives. Materialising it needs about
**7.4 GB, on a machine with 7.3 GB of RAM total**.

So it is never materialised. Conjugate gradients needs only the matrix's action
on a vector:

```
(S'W⁻¹S) v  =  S'( W⁻¹( S v ) )
```

— two sparse mat-vecs against a matrix with 365,880 non-zeros (0.028% dense),
Jacobi-preconditioned with the exact diagonal. A problem that does not fit in
memory becomes a few seconds of sparse arithmetic.

`tests/test_reconcile.py` verifies the matrix-free solver against the **dense
closed form** on a small hierarchy where the textbook formula can be computed
directly. That is what makes the fast path trustworthy.

### Base forecasts, routed by series character

- **Bottom level** — the global LightGBM
- **Levels 1–9** (154 nodes) — AutoETS. Summed over a whole store or category,
  demand is smooth and weekly-seasonal: ETS's home ground.
- **item and item×state** (12,196 nodes) — TSB. These are *still intermittent*;
  one slow SKU summed over ten stores still has zero days.

### Coherence: solved

| Method | Max adding-up violation |
|---|---:|
| base (independent) | 7,933.7 units |
| bottom_up | **0.0** |
| ols | **0.0** |
| wls_struct | **0.0** |
| wls_var | **0.0** |

### Accuracy: the honest result

RMSSE by level:

| Level | Nodes | base | bottom_up | ols | wls_struct | wls_var |
|---|---:|---:|---:|---:|---:|---:|
| total | 1 | 0.669 | **0.298** | 0.626 | 0.618 | 0.613 |
| state | 3 | 0.700 | **0.377** | 0.673 | 0.658 | 0.629 |
| store | 10 | 0.698 | **0.458** | 0.671 | 0.673 | 0.649 |
| category | 3 | 0.630 | **0.368** | 0.586 | 0.594 | 0.629 |
| department | 7 | 0.893 | **0.480** | 0.796 | 0.850 | 0.820 |
| state × category | 9 | 0.664 | **0.466** | 0.635 | 0.634 | 0.650 |
| state × department | 21 | 0.880 | **0.583** | 0.811 | 0.831 | 0.801 |
| store × category | 30 | 0.677 | **0.556** | 0.687 | 0.670 | 0.675 |
| store × department | 70 | 0.827 | **0.655** | 0.824 | 0.799 | 0.776 |
| item | 3,049 | 0.908 | 0.864 | 0.910 | 0.870 | **0.858** |
| item × state | 9,147 | 0.849 | 0.837 | 0.853 | 0.832 | **0.825** |
| item × store | 30,490 | 0.824 | 0.824 | 0.830 | 0.820 | **0.818** |

Two findings, and the second is the one worth telling:

**1. MinT improves the bottom level for free.** `wls_var` scores 0.8176 against
the base model's 0.8241 — 0.8% better with no new model, purely from pooling
information across levels. It also wins at `item` (−5.5%) and `item×state`
(−2.8%).

**2. Bottom-up beats every MinT variant on the aggregate levels, by up to 55%.**
That is not the textbook expectation, and it is a fact about *these base
forecasts* rather than about reconciliation. The bottom model is tuned,
backtested over five folds and given 33 engineered features; the aggregate
forecasts are an out-of-the-box AutoETS. MinT blends in proportion to assumed
error variance, so blending a good forecast with a poor one lands between them.
Simply summing the good one wins.

**What that implies is specific:** to make MinT beat bottom-up on aggregates, the
aggregate forecasts have to be competitive — which means training the global
model at each level rather than reaching for ETS. That is real work, and it is
the honest recommendation rather than a claim that reconciliation "improved
accuracy".

---

## 11. From forecast to decision

### The policy

Periodic review every **T = 7 days**. At a review, if the inventory **position**
(on hand *plus on order*) has fallen to the reorder point *s*, order up to *S*.

```
s = quantile_SL( demand over L )        protects until an order placed now arrives
S = quantile_SL( demand over T + L )    protects until the order AFTER that arrives
```

**S covers T+L, not L.** That is the classic mistake in periodic-review systems:
after ordering you wait L days for delivery *and* another T days before you can
order again, so sizing S on L alone under-stocks by about a review period of
demand every single cycle.

Comparing against inventory *position* rather than on-hand matters too —
comparing against on-hand alone re-orders every review while stock is in
transit, producing duplicate orders and wildly overstating both holding cost and
order count.

### Why not z·σ·√L

The textbook safety-stock formula assumes normal, i.i.d. demand and constant lead
time. All three are false here:

- 72.6% of series are intermittent — zero-inflated and right-skewed, nothing like
  a normal
- demand is autocorrelated and day-of-week seasonal, so √L understates the
  variance of a *run* of days
- lead time is a distribution, not a constant

Applied to an intermittent series the formula produces negative reorder points at
low service levels and badly wrong ones in the tail.

### Scaled block bootstrap instead

For each series: draw a lead time, draw a random start point in that series' own
clean history, take that many **contiguous** days, sum them, scale by
(forecast level / historical level), repeat.

**Contiguous blocks are the whole point.** Sampling individual days independently
destroys the autocorrelation and day-of-week structure and understates the
variance of the sum — which is exactly the error √L makes. Blocks carry that
structure through for free, with no distributional assumption anywhere.

Censored days inside a block are **imputed with the series' clean mean, never
zero-filled** — otherwise the censoring bias removed in §5 walks straight back in
at the last step.

The normal approximation is still implemented, as the **comparison arm**, so the
cost of assuming normality can be *quantified* rather than asserted.

### The cost model

```
Total cost = holding + stockout + ordering
  holding  = h × average_on_hand_units × unit_cost
  stockout = p × expected_unmet_units  × unit_margin
  ordering = K × number_of_orders
```

| Parameter | Default | Basis |
|---|---|---|
| gross margin | 0.25 | US grocery retail runs 22–28% |
| holding rate | 0.25/year | Standard carrying cost: capital, warehousing, shrinkage, obsolescence |
| stockout penalty | 1.5 × lost margin | Lost margin plus something for the customer who goes elsewhere |
| order cost | $0.50/line | Marginal admin cost of one more line on a truck already going |

**None of these are in the data.** M5 gives a selling price and nothing else
economic. The ratio between them determines the entire answer — see §16.

---

## 12. The headline result

All policies replay the **same realised demand** over the locked test window
(d1914–1941), after a 28-day warm-up so no policy benefits from its own choice of
starting inventory.

| Policy | Fill rate | Cycle service | Avg units on hand | Unmet units | Total cost | vs baseline | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Reconstructed baseline | 94.05% | 98.42% | 18.46 | 73,263 | $127,930 | — | — |
| Better forecast only | 94.03% | 98.49% | 19.16 | 73,588 | $129,254 | +1.0% | −1.3% to +3.5% |
| Empirical safety stock, uniform level | 94.02% | 98.02% | 18.74 | 73,659 | $133,976 | +4.7% | +1.4% to +8.3% |
| **Per-series cost-optimal allocation** | **95.06%** | 98.35% | 20.17 | **60,805** | **$121,500** | **−5.0%** | **−8.5% to −1.9%** |

The recommended policy is **cheaper and serves more demand** — dominance, so
there is no trade-off to argue about. It cut unmet units from 73,263 to 60,805
(**−17%**) while spending 5% less.

Cost breakdown for the recommendation: $28,882 holding + $71,817 stockout +
$20,802 ordering.

### The decomposition is the actual finding

Three arms were run precisely so the gain could be **attributed**, and the answer
was not the expected one:

- **A better forecast bought nothing.** Swapping the moving average for the global
  model, holding safety-stock method fixed, moved cost +1.0% with a confidence
  interval straddling zero. The 3.8% WRMSSE improvement did not convert into
  money.
- **Replacing normal safety stock with an empirical bootstrap, at a uniform
  service level, was actively worse** (+4.7%, significant). Matching the
  *aggregate unit* fill rate does not match *margin-weighted* stockout cost — the
  empirical method hit the same fill rate while concentrating its failures on
  more valuable items.
- **The entire gain came from the allocation** — deciding which series deserve
  the stock, at 7 distinct service levels with a median of 0.99.

The honest one-line summary is not "ML forecasting saves money". It is **"the
value was in the decision layer, not the forecast"** — which is the opposite of
where most of the effort in a project like this usually goes.

### Two methodological corrections behind that number

**Comparisons must be at matched service.** A service-level *parameter* and an
achieved *fill rate* are different quantities, and the normal approximation and
the empirical bootstrap convert one into the other very differently on
intermittent demand. A first version ran the baseline at whatever parameter
reproduced observed behaviour and the others at a fixed 0.95, then compared cost
— and reported the optimized policy as **32% more expensive** while it was serving
2.8 points less demand. Every policy is now bisected onto the same achieved fill
rate, to within 0.0005.

**Levels must be chosen on expected cost, not realised cost.** The first
allocation simulated every candidate level on the realised test demand and took
the per-series minimum. That is selection on the outcome: each series gets the
level that happened to work best on the very days the result is reported over. It
produced a **39% "saving" that was mostly hindsight**. Levels now come from the
forecast's lead-time demand distribution alone; realised demand is touched exactly
once, to evaluate the chosen allocation.

### Sensitivity — the sign never flips

Across the full 27-point factorial of gross margin × holding rate × stockout
penalty:

| | delta |
|---|---:|
| Best case (margin 0.30, holding 0.15, penalty 3.0) | −8.76% |
| Median | −5.16% |
| Worst case (margin 0.20, holding 0.40, penalty 1.0) | −0.77% |

**All 27 cells show a saving.** The magnitude is uncertain; the direction is not.
That is a stronger claim than any single number.

---

## 13. The service-level frontier

![Service level versus cost](frontier.png)

| Target | Achieved fill | Cycle service | Avg on hand | Avg safety stock | Total cost |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 82.95% | 93.49% | 10.88 | 3.63 | $279,622 |
| 0.85 | 85.02% | 94.40% | 11.81 | 4.56 | $251,938 |
| 0.90 | 87.69% | 95.47% | 13.18 | 5.90 | $215,667 |
| 0.95 | 91.05% | 96.86% | 15.56 | 8.15 | $171,469 |
| 0.975 | 93.32% | 97.74% | 17.85 | 10.28 | $141,869 |
| 0.99 | 95.16% | 98.44% | 20.43 | 12.93 | $119,852 |
| 0.995 | 95.99% | 98.76% | 21.86 | 14.50 | $110,471 |
| 0.999 | 97.01% | 99.11% | 23.88 | 16.72 | $99,618 |
| *reconstructed baseline* | *94.05%* | *98.42%* | *18.46* | *10.79* | *$127,930* |

**The realised frontier is monotone — there is no interior optimum.** Cost keeps
falling as service rises, all the way to 99.9%.

That is not a bug, it is the cost model's arithmetic. Over a ~12-day exposure
window a held unit costs about **$0.025** while a lost unit costs about **$1.50**
— a ratio of roughly **26:1**. The newsvendor critical ratio p/(p+h) therefore
sits near 0.98, and stocking more keeps paying.

**This is precisely why the recommendation is a per-series allocation rather than
a uniform target.** If one number were the answer it would just be "as high as
capacity allows", which is not a decision worth building a project around.

The *expected*-cost curve, computed from the forecast distribution rather than the
realised path, does turn — minimising at 0.99 ($96,881 expected) and rising at
0.995 and 0.999. The gap between the two is itself informative: realised demand in
this particular 28-day window rewarded stock more than the forecast distribution
predicted it would.

### "What if service moved from 95% to 98%?"

Non-linear, and in this cost regime it moves in the *helpful* direction. Going
from a 0.95 target to a 0.99 target raises average inventory **31%** (15.56 →
20.43 units) but **cuts total cost 30%**, because avoided stockouts are worth far
more than the extra carrying cost.

Halve the stockout penalty and that relationship reverses — which is exactly what
the sensitivity grid is for.

---

## 14. Constraints and shadow prices

Real stores cannot set every series to the same level: shelf space is finite and
so is working capital. Something has to give, and the question is *which series*.

### The formulation

A **multiple-choice knapsack**: each series takes one service level from K
candidates, subject to per-store capacity and a total budget.

```
minimise    Σ  cost[i,k] · y[i,k]
subject to  Σ_k y[i,k] = 1                      for every series
            Σ  units[i,k] · y[i,k] ≤ capacity[store]
            Σ  capital[i,k] · y[i,k] ≤ budget
```

As an integer program that is 30,490 × 8 = **243,920 binaries**, which CBC will
not close on a laptop.

### Why the LP relaxation is near-exact here

Not a concession — a structural result. A basic feasible solution of an LP has at
most as many basic variables as it has rows. The assignment constraints force one
unit of mass per series, so **only the side constraints can split a series**. With
10 store rows plus a budget row, **at most 11 of 30,490 series can be fractional**.

Measured: exactly **11**. A satisfying confirmation that the formulation behaves
as the theory predicts.

Those are rounded toward the **smaller footprint**, not the cheaper level, so a
binding capacity row cannot end up over its limit — rounding toward cost pushed
utilisation to 100.2% in testing, which is an infeasible answer.

### The duals

Capacity set to 85% of what the unconstrained optimum wants at each store, plus a
budget cap at 90% of its capital, so the constraints bind and the duals are
informative.

| Constraint | Utilisation | Shadow price |
|---|---:|---:|
| capacity **CA_3** | 100.0% | **0.0514** |
| capacity CA_1 | 100.0% | 0.0480 |
| capacity TX_2 | 100.0% | 0.0449 |
| capacity CA_2 | 100.0% | 0.0436 |
| capacity TX_3 | 100.0% | 0.0426 |
| capacity WI_1 | 100.0% | 0.0424 |
| capacity CA_4 | 100.0% | 0.0422 |
| capacity WI_3 | 99.9% | 0.0417 |
| capacity WI_2 | 100.0% | 0.0413 |
| capacity **TX_1** | 100.0% | **0.0392** |
| **budget** | 100.0% | **0.0032** |

Constrained cost $99,310 vs unconstrained $96,759 — **the constraints cost $2,551
per 28 days**, about 2.6%.

### What you actually tell the ops director

An extra unit of shelf space is worth about **5.1¢ per 28 days at CA_3** and
**3.9¢ at TX_1** — roughly 30% more valuable at the former. If space can be moved
or added anywhere, CA_3 is where it pays back.

The budget dual is an **order of magnitude smaller** (0.32¢ per dollar). So:
**space binds, capital does not.** Asking for more working capital would buy
almost nothing; asking for more shelf would.

That is the sentence that turns an optimization model into a decision.

---

## 15. Serving it

**FastAPI** over a compact serving bundle:

```
GET  /health                             artifact + table status
GET  /series                             browse/filter the catalogue
GET  /forecast/{store_id}/{item_id}      point + quantiles + actuals
GET  /policy/{store_id}/{item_id}        s, S, safety stock, chosen level
POST /simulate                           live re-simulation
GET  /results                            every headline table
```

**Streamlit dashboard**, four views: Executive (frontier, cost delta, shadow
prices), Forecast explorer, Policy explorer (with live re-simulation), Model
health (backtests, calibration, reconciliation).

The full panel matrices are ~500 MB in memory, which is the wrong thing to put in
a container that only answers questions about the forecast horizon.
`build_serving_bundle.py` reduces it to ~20 MB of numpy arrays — small enough that
**`/simulate` runs a real simulation over all 30,490 series on each call** (~42 s)
rather than interpolating a precomputed table.

Containerised and verified: image 1.4 GB, API ~209 MB resident, dashboard ~76 MB,
both healthy, `/simulate` returning numbers identical to the local run.

Charts use a colourblind-safe palette validated against both light and dark
surfaces (worst adjacent CVD ΔE 9.1 / 8.4), and every chart ships a table view
beside it.

---

## 16. What is wrong with all of this

The most important section. Ten numbered assumptions live in
[03-assumptions-and-limitations.md](03-assumptions-and-limitations.md); these are
the load-bearing ones.

### The baseline is reconstructed, not observed

M5 contains **no inventory data** — no stock levels, no purchase orders, no
receipts. So there is no observed replenishment policy to compare against.

The reconstruction is an (s,S) policy of the same structure driven by what a
conventional system does (28-day moving average, textbook normal safety stock),
with its service-level parameter **bisected until its simulated fill rate matches
the rate implied by the stockout proxy**. Calibrating to observed behaviour is
what stops it being a strawman.

**Bias: unknown in direction, possibly large.** If real stores ran a better
policy, the saving is overstated. If they ran worse — manual ordering, fixed par
levels — it is understated. Nothing in M5 can settle it.

### Every cost parameter is assumed

The **holding-to-stockout ratio of roughly 1:26** over the exposure window drives
the entire answer, and it comes from literature figures rather than the
retailer's books. That single ratio is why the frontier has no interior minimum.

This is why the result is reported across a 27-cell factorial rather than at a
point. The defensible claim is never "this saves 5.0%" — it is **"this saves 5.0%
under these stated assumptions, and here is how it moves when they change"**.

### The counterfactual is a simulation, not an experiment

Every policy is evaluated by replaying **the same realised demand**. That embeds a
real circularity: observed sales were themselves shaped by the stockouts the real
stores had, and a policy that prevented those stockouts would have sold more than
the data records.

The stockout proxy mitigates this for training but cannot manufacture demand that
was never observed. **The simulation therefore systematically understates what a
better policy would achieve.** A real deployment needs a phased rollout with
control stores.

### Structural limits

- **No supplier constraints** — no minimums, truck capacity, or order-frequency
  limits beyond the review period.
- **No multi-echelon transshipment.** Stores are independent. Real networks move
  stock laterally, which materially reduces total safety stock — a genuine
  multi-echelon system would beat everything reported here.
- **No substitution.** A stockout is modelled as a fully lost sale. In grocery,
  customers frequently buy the adjacent brand.
- **No shelf life or spoilage**, despite FOODS being 47% of the catalogue.
- **Lead time (3–7 days) and case packs are invented.** M5 has no supplier data.

### Residual data issues

- **α is chosen, not measured** — the censoring rate swings 2.4%–10.9% across its
  plausible range.
- **Partial stockouts are invisible.** A shelf that emptied at noon shows
  truncated, not zero, sales.
- Weather is state-level, not store-level.
- Weekly prices are held flat within the week, smearing promotion start/end
  effects to week boundaries.

---

## 17. Six things that went wrong during the build

These are written up rather than quietly patched, because the corrections are
more instructive than the final numbers.

**1. The stockout rule flagged 18.3% of the panel** with a mean run of 40 days.
That is dormancy, not stockout. Cause: a lifetime demand rate cannot see drift.
Fixed with a trailing 56-day local rate → 3.99%, median run 8 days.

**2. The α sensitivity table was inert.** It recomputed thresholds over the
*already-filtered* table, so every α looser than the build-time value returned an
identical answer and the parameter looked far more robust than it is. Fixed by
keeping the unfiltered runs; the real range is 2.37%–10.86%.

**3. The first cost comparison was rigged.** Policies ran at different achieved
service levels, so a "32% more expensive" result was measuring the service gap,
not the method. Fixed by bisecting every policy onto the same fill rate.

**4. The first allocation overfit.** It chose each series' level by simulating on
the realised test demand — selection on the outcome. A 39% "saving" that was
hindsight. Levels now come from forecast-derived expected cost only.

**5. AutoETS was run on all 12,350 aggregate nodes.** It took over 20 minutes and
drove the machine into swap — for the `item` and `item×state` levels, which are
*still intermittent* and where a smoothing model has least to offer. Now routed:
AutoETS on the 154 genuinely smooth nodes, TSB on the 12,196 sparse ones. Runs in
seconds.

**6. Two chart bugs, found only by rendering the chart and looking at it.** The
frontier PNG came out with a `nan%` label and the recommendation marker on the
wrong point — Polars treats null and NaN as different, and the baseline row
carries NaN, so `is_not_null()` alone let it through and it was drawn as a ninth
point *on* the frontier line. **The dashboard had the identical bug, shipped and
unnoticed.** Separately, the chart had marked a point on the uniform-target curve
as "recommended", which it is not — the recommendation allocates per series and
sits off that curve entirely.

A false claim in a docstring was also corrected along the way: `cycle service
always exceeds fill rate`. It does not — which is higher depends on whether
failures are rare-and-large or frequent-and-small. A test now pins both orderings.

---

## 18. How to run it

```bash
uv sync --extra dev --extra models --extra optimize --extra app
cp .env.example .env      # set SCI_DATA_DIR and SCI_M5_SOURCE

uv run sci ingest all              # M5 from local zip + Open-Meteo weather
uv run sci warehouse build         # ~4 min -> ~1.9 GB DuckDB
uv run sci analyze                 # the five SQL analytics reports

uv run python scripts/run_model_ladder.py     # ~40 min
uv run python scripts/run_optimizer.py        # ~15 min
uv run python scripts/run_reconciliation.py   # ~2 min
uv run python scripts/build_serving_bundle.py
uv run python scripts/make_figures.py

uv run streamlit run app/dashboard.py
uv run uvicorn app.api:app --reload
```

Containerised:

```bash
uv run python scripts/stage_deploy.py
docker compose up --build          # dashboard :8501, API :8000/docs
```

**Tests: 123, all passing.** `SCI_DATA_DIR` should point somewhere outside any
synced folder — the warehouse and matrices run to several GB.

---

## In one paragraph

Most portfolio projects of this shape report "my model beat the baseline by X%".
This one measured **where the money actually came from** and found the forecast
was not it: a better forecast alone moved cost +1.0% with a confidence interval
straddling zero, and the entire 5.0% gain came from deciding which series deserve
the stock. Along the way the stockout detector had to be rebuilt after it
mistook dormancy for censoring, the cost comparison had to be redone after it
turned out to be measuring a service gap rather than a method, and the allocation
had to be rebuilt after it was caught selecting on the outcome it was scored
against. All of that is in the repo, in writing. The number at the top is smaller
because of it, and considerably harder to knock down.
