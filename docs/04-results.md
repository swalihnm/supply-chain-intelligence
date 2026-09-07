# Results

Every number here is produced by a script in this repo and reproducible with
`sci warehouse build` followed by `sci analyze <name>`.

**Week 1 (data + analytics) is complete. Weeks 2–4 (models, optimizer, serving)
are not started, so there is no cost or service-level result yet.**

---

## 1. The panel

| | |
|---|---|
| Raw sales rows | 59,181,090 (30,490 series × 1,941 days) |
| Rows after availability filter | 46,881,677 (79.2%) |
| Series | 30,490 (3,049 items × 10 stores) |
| Stores | 10 across CA (4), TX (3), WI (3) |
| Date range | 2011-01-29 → 2016-05-22 |
| Warehouse size | 163 MB (DuckDB, zstd parquet staging) |
| Full rebuild time | ~4 minutes from raw zip |

### The availability filter

20.8% of raw rows have no `sell_price`. This is not missing data — it is a
clean signal that the store did not carry that SKU yet. Three checks establish
that, all run against the real data rather than assumed:

- the price-NULL block is **contiguous and leading for 100.00%** of all 30,490 series
- **zero** rows exist where `units > 0` but `sell_price IS NULL`
- median gap between first priced day and first sale is **2 days** (p95 = 6)

Median first-priced day is d=160, and 15,752 of 30,490 series (51.7%) do not
start until after d=100. Training on those structural zeros would teach every
model that a fifth of demand is zero when the product simply did not exist.

---

## 2. Demand structure

### Syntetos-Boylan classification

| Class | Series | % series | % revenue |
|---|---:|---:|---:|
| intermittent | 22,143 | 72.6% | 51.6% |
| lumpy | 5,612 | 18.4% | 16.6% |
| smooth | 1,880 | 6.2% | 24.2% |
| erratic | 855 | 2.8% | 7.7% |

**91% of series are intermittent or lumpy.** Fitting ETS or ARIMA to those is a
category error — they need Croston/TSB. This single table is why the model
ladder routes by series class rather than running one method everywhere.

### ABC × XYZ 9-box

| Box | Series | % series | % revenue | % intermittent/lumpy |
|---|---:|---:|---:|---:|
| AX | 83 | 0.3% | 3.1% | 0% |
| AY | 2,731 | 9.0% | 31.7% | 22% |
| **AZ** | **8,407** | **27.6%** | **45.1%** | **96%** |
| BY | 269 | 0.9% | 0.5% | 48% |
| BZ | 9,157 | 30.0% | 14.5% | 99.6% |
| CY | 48 | 0.2% | 0.0% | 48% |
| CZ | 9,795 | 32.1% | 5.0% | 100% |

**AZ is the whole problem in one cell:** 45% of revenue sits in series that are
simultaneously high-value and highly erratic, and 96% of them are intermittent
or lumpy. That is the direct argument for probabilistic forecasting — a point
forecast on an AZ series is nearly meaningless, and the optimizer needs the
distribution to set safety stock at all.

Revenue is also less concentrated than the usual Pareto assumption: it takes
36.8% of series to reach 80% of revenue, and the top 20% of series carry 63.7%.

---

## 3. Stockout proxy

Detection uses a per-series adaptive threshold rather than a fixed rule — see
[02-methodology.md](02-methodology.md#stockout-detection).

| Censoring type | Runs | Days | % of panel | Series affected | Median run |
|---|---:|---:|---:|---:|---:|
| stockout (impute) | 188,806 | 1,868,039 | **3.99%** | 27,215 | 8 days |
| dormant (exclude) | 65,316 | 6,689,647 | 14.27% | 23,785 | 74 days |

3.99% is a plausible grocery out-of-stock rate. The direction of variation is
right too: FOODS stocks out more than HOBBIES (4.45% vs 3.33%), and A-items
more than C-items (4.52% vs 2.82%) — fast movers run out.

### Sensitivity to alpha

The 3.99% headline depends on a significance level that was chosen, not
measured. The honest range:

| alpha | Stockout days | % of panel |
|---:|---:|---:|
| 0.001 | 1,108,851 | 2.37% |
| **0.005** (used) | 1,868,039 | **3.99%** |
| 0.010 | 2,425,736 | 5.17% |
| 0.050 | 5,091,514 | 10.86% |

A 4.6× swing across the plausible range. Any downstream cost number inherits
this uncertainty and the counterfactual in week 3 must be re-run across it.

---

## 4. Seasonality

Computed on the panel with censored days removed, as lift against each
category's own mean.

**Day of week** — weekends dominate every category:

| Category | Sat | Sun | Mon | Tue | Wed | Thu | Fri |
|---|---:|---:|---:|---:|---:|---:|---:|
| FOODS | +18.7% | +19.9% | −3.4% | −10.6% | −11.9% | −11.5% | −1.3% |
| HOBBIES | +22.5% | +11.2% | −6.8% | −10.8% | −9.9% | −10.0% | +3.7% |
| HOUSEHOLD | +25.6% | +21.5% | −5.6% | −13.8% | −14.8% | −13.9% | +0.9% |

**SNAP benefit days** — the cleanest validation that the joins are correct:

| Category | Lift on SNAP days |
|---|---:|
| FOODS | **+7.7%** |
| HOUSEHOLD | +1.9% |
| HOBBIES | +1.3% |

SNAP is food assistance, so FOODS should respond and the other two should
barely move. They do. Nothing in the pipeline was tuned to produce this, which
makes it a genuine end-to-end check on the calendar → store → state join.

---

## 5. Price

### Naive log-log elasticity

| Category | Elasticity | R² | n |
|---|---:|---:|---:|
| FOODS | −0.432 | 0.089 | 10,662,222 |
| HOUSEHOLD | −0.362 | 0.119 | 5,771,078 |
| HOBBIES | −0.262 | 0.163 | 2,505,971 |

Correct sign everywhere, weak fit everywhere — exactly what an uncontrolled
log-log fit on retail data should produce. **This is not a causal estimate** and
the file says so at length: prices are set in response to expected demand, no
controls are applied, and dropping zero-sale days selects out the low-demand
days a high price would cause.

### Promotion lift vs matched controls

Promo days compared against non-promo days of the same series, same weekday,
same quarter. Temporary promotions (price recovers) are separated from permanent
markdowns (it does not).

| Category | Type | Discount | Lift | Median cell lift |
|---|---|---|---:|---:|
| FOODS | promo | 20–35% | **+103%** | 1.80 |
| FOODS | markdown | 20–35% | +66% | 1.20 |
| FOODS | promo | 5–10% | **−27%** | 0.83 |
| FOODS | promo | 10–20% | **−30%** | 0.67 |
| HOUSEHOLD | promo | 10–20% | −56% | 0.63 |

Deep discounts lift demand substantially. **Shallow discounts associate with
lower demand**, and the effect survives both a volume-weighted and an unweighted
per-cell estimator, so it is not an aggregation artifact.

The most likely explanation is endogeneity rather than a real negative price
response: retailers apply modest markdowns *reactively*, to products already
slowing down. This is a good illustration of why the modelling layer needs price
as a controlled feature rather than a lift factor lifted from this table.

---

## Still to come

| Week | Deliverable |
|---|---|
| 2 | Model ladder, backtest harness, calibrated quantiles |
| 3 | (s,S) policy, cost model, service-level frontier, counterfactual with CI |
| 4 | Dashboard, deploy, README headline number |
