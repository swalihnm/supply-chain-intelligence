# Results

Every number here is produced by a script in this repo and reproducible with
`sci warehouse build` followed by `sci analyze <name>`.

**All four layers are built, plus hierarchical reconciliation. The headline is in §7; reconciliation is in §10.**

---

## 1. The panel

| | |
|---|---|
| Raw sales rows | 59,181,090 (30,490 series × 1,941 days) |
| Rows after availability filter | 46,881,677 (79.2%) |
| Series | 30,490 (3,049 items × 10 stores) |
| Stores | 10 across CA (4), TX (3), WI (3) |
| Date range | 2011-01-29 → 2016-05-22 |
| Warehouse size | ~1.9 GB on disk; ~1.1 GB live data, the rest free blocks |
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

## 6. Forecast accuracy

Rolling-origin CV, 5 folds, 28-day horizon. The test fold (d1914–1941) was
scored once.

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
such. The direct 28-day-lag design gives up short-horizon accuracy in exchange
for a leakage audit that is arithmetic rather than trust; a recursive design
using lag-1 would score better and be far harder to defend.

### Three things in this table worth stating

**The zero forecast has the best MASE of every model** (0.986) while having the
worst WRMSSE by a distance. That is not a bug — with 68% of days at zero,
"always predict zero" minimises absolute error while missing every spike that
squared error punishes. It is a clean demonstration that the metric choice
*is* a modelling decision, and it is why the optimizer is driven by pinball
loss rather than either of these.

**Seasonal naive is much worse than a plain moving average** (1.156 vs 0.895).
Repeating last week's exact pattern on a sparse series propagates noise; on
intermittent data the smoothing is worth more than the seasonality.

**Seasonal Croston — a day-of-week reshaping of the Croston rate — made things
worse** (0.916 vs 0.891). A negative result, kept in the table rather than
dropped.

### Error by demand class (mean RMSSE, CV folds)

| Model | intermittent | lumpy | smooth | erratic |
|---|---:|---:|---:|---:|
| moving average | 0.734 | 0.714 | 0.779 | 0.738 |
| TSB | 0.762 | 0.746 | 0.785 | 0.754 |
| zero | 0.844 | 0.857 | 1.387 | 1.120 |
| seasonal naive | 0.964 | 0.912 | 0.987 | 0.929 |

The zero forecast is competitive on intermittent and lumpy series and collapses
on smooth ones — exactly where you would expect.

### Quantile calibration

Where the optimizer actually reads the distribution, calibration is close to
exact — and it is *better* on the locked test fold than in CV, which is the
direction you want:

| Nominal | cv4 | cv5 | **test (locked)** | test gap |
|---:|---:|---:|---:|---:|
| 0.50 | 0.665 | 0.660 | 0.638 | +0.138 |
| 0.90 | 0.914 | 0.913 | **0.9007** | **+0.0007** |
| 0.95 | 0.958 | 0.958 | **0.9505** | **+0.0005** |
| 0.99 | 0.990 | 0.990 | **0.9885** | −0.0015 |

Split conformal was tried as an alternative and came out slightly *worse* at
0.90 (0.8887) and 0.95 (0.9434), so the direct quantile models are used.

**The 0.5 quantile is badly calibrated (0.665 vs 0.50), and that is a property
of the data, not a defect.** Coverage is defined as P(actual ≤ predicted), and
with 68% of days at zero the median of an intermittent series *is* zero — so
"actual ≤ 0" is true on every zero day. The statistic saturates against the
point mass. The upper quantiles, which are the only ones the policy uses, are
unaffected.

Quantile crossing occurred on 2.7% of cells and was repaired by sorting.
Conformal calibration was slightly worse than direct quantile regression at 0.90
and 0.95, so the direct models are used.

---

## 7. The decision — counterfactual on the test window

All policies replay the same realised demand over d1914–1941, after a 28-day
warm-up so none benefits from its starting inventory.

| Policy | Fill rate | Cycle service | Avg units on hand | Total cost | vs baseline | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Reconstructed baseline | 94.05% | 98.42% | 18.46 | $127,930 | — | — |
| Better forecast only | 94.03% | 98.49% | 19.16 | $129,254 | +1.0% | −1.3% to +3.5% |
| Empirical safety stock, uniform level | 94.02% | 98.02% | 18.74 | $133,976 | +4.7% | +1.4% to +8.3% |
| **Per-series cost-optimal allocation** | **95.06%** | 98.35% | 20.17 | **$121,500** | **−5.0%** | **−8.5% to −1.9%** |

The recommended policy is **cheaper and serves more demand** — dominance, so
there is no trade-off to argue about. It cut unmet units from 73,263 to 60,805
(−17%) while spending 5% less.

### The decomposition is the finding

Three arms were run precisely so the gain could be attributed, and the answer
was not the expected one:

- **A better forecast bought nothing.** Swapping the moving average for the
  global model, holding the safety-stock method fixed, moved cost by +1.0% with
  a confidence interval straddling zero. The 3.8% WRMSSE improvement did not
  convert into money.
- **Replacing normal safety stock with an empirical bootstrap, at a uniform
  service level, was actively worse** (+4.7%, significant). Matching the
  *aggregate unit* fill rate does not match *margin-weighted* stockout cost —
  the empirical method hit the same fill rate while concentrating its failures
  on more valuable items.
- **The entire gain came from the allocation** — deciding which series deserve
  the stock, at 7 distinct service levels with a median of 0.99.

The honest one-line summary is not "ML forecasting saves money". It is **"the
value was in the decision layer, not the forecast"** — which is the opposite of
where most of the effort in a project like this usually goes.

### Sensitivity — the sign never flips

Across the full 27-point factorial of gross margin × holding rate × stockout
penalty:

| | delta |
|---|---:|
| Best case (margin 0.30, holding 0.15, penalty 3.0) | −8.76% |
| Median | −5.16% |
| Worst case (margin 0.20, holding 0.40, penalty 1.0) | −0.77% |

**Every one of the 27 cells shows a saving.** The magnitude is uncertain; the
direction is not. That is a stronger claim than any single number.

---

## 8. Service level frontier

| Target | Achieved fill | Avg on hand | Avg safety stock | Total cost |
|---:|---:|---:|---:|---:|
| 0.80 | 82.95% | 10.88 | 3.63 | $279,622 |
| 0.85 | 85.02% | 11.81 | 4.56 | $251,938 |
| 0.90 | 87.69% | 13.18 | 5.90 | $215,667 |
| 0.95 | 91.05% | 15.56 | 8.15 | $171,469 |
| 0.975 | 93.32% | 17.85 | 10.28 | $141,869 |
| 0.99 | 95.16% | 20.43 | 12.93 | $119,852 |
| 0.995 | 95.99% | 21.86 | 14.50 | $110,471 |
| 0.999 | 97.01% | 23.88 | 16.72 | $99,618 |
| *baseline* | *94.05%* | *18.46* | *10.79* | *$127,930* |

**The realised frontier is monotone — there is no interior optimum.** Cost keeps
falling as service rises, all the way to 99.9%.

That is not a bug, it is the cost model's arithmetic. Over a ~12-day exposure
window a held unit costs about $0.025 while a lost unit costs about $1.50 — a
ratio of roughly 26:1 — so the newsvendor critical ratio p/(p+h) sits near 0.98
and stocking more keeps paying.

**This is why the recommendation is a per-series allocation rather than a
uniform target.** If one number were the answer it would just be "as high as
capacity allows", which is not a decision worth building a project around.

The *expected*-cost curve, computed from the forecast distribution rather than
the realised path, does turn — minimising at 0.99 ($96,881 expected) and rising
at 0.995 and 0.999. The gap between the two is itself informative: realised
demand in this particular 28-day window rewarded stock more than the forecast
distribution predicted it would.

### Answering "what if service moved from 95 to 98?"

Non-linear, and in this cost regime it moves in the *helpful* direction: going
from a 0.95 target to a 0.99 target raises average inventory 31% (15.56 → 20.43
units) but cuts total cost 30%, because avoided stockouts are worth far more
than the extra carrying cost. Under a cost model with a cheaper stockout penalty
that relationship reverses — which is exactly what the sensitivity grid is for.

---

## 9. Constrained allocation and shadow prices

Capacity set to 85% of what the unconstrained optimum wants at each store, plus a
budget cap at 90% of its capital, so the constraints bind and the duals are
informative.

| Constraint | Utilisation | Shadow price |
|---|---:|---:|
| capacity CA_3 | 100.0% | **0.0514** |
| capacity CA_1 | 100.0% | 0.0480 |
| capacity CA_2 | 100.0% | 0.0436 |
| capacity CA_4 | 100.0% | 0.0422 |
| capacity WI_1 | 100.0% | 0.0424 |
| capacity WI_3 | 99.9% | 0.0417 |
| capacity WI_2 | 100.0% | 0.0413 |
| capacity TX_3 | 100.0% | 0.0426 |
| capacity TX_1 | 100.0% | 0.0392 |
| budget | 100.0% | 0.0032 |

Constrained cost $99,310 vs unconstrained $96,759 — **the constraints cost
$2,551 per 28 days**, about 2.6%.

**What to tell the ops director:** an extra unit of shelf space is worth about
5.1¢ per 28 days at CA_3 and 3.9¢ at TX_1 — roughly 30% more valuable at the
former. If space can be moved or added anywhere, CA_3 is where it pays. The
budget dual is an order of magnitude smaller (0.32¢ per dollar), so **capital is
not the binding problem here; shelf space is.**

The LP relaxation produced **11 fractional assignments out of 30,490** — exactly
the theoretical bound of one per side constraint (10 stores + 1 budget), which
is a satisfying confirmation that the relaxation is doing what the formulation
predicts.

---

## 10. Hierarchical reconciliation

42,840 nodes over 30,490 bottom series, across the 12 published M5 levels. Base
forecasts made independently: global LightGBM at the bottom, AutoETS on the 154
genuinely smooth nodes, TSB on the 12,196 `item`/`item_state` nodes that are
still intermittent.

### The problem, made concrete

The total-level forecast says **1,147,983 units**. The bottom-level forecasts
sum to **1,241,178**. Two teams looking at the same 28 days, **8.1% apart** —
and the largest single-node disagreement is 7,934 units.

| Method | Max adding-up violation |
|---|---:|
| base (independent) | 7,933.7 units |
| bottom_up | **0.0** |
| ols | **0.0** |
| wls_struct | **0.0** |
| wls_var | **0.0** |

Every reconciliation method makes the numbers add up exactly. That is the
deliverable; the accuracy question below is separate.

### RMSSE by level

| Level | Nodes | base | bottom_up | ols | wls_struct | **wls_var** |
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

### Two findings, and the second is the honest one

**1. MinT improves the bottom level for free.** `wls_var` scores 0.8176 at
item × store against the base model's 0.8241 — 0.8% better, with no new model,
purely from pooling information across levels. It also wins at `item` (−5.5%)
and `item × state` (−2.8%).

**2. Bottom-up beats every MinT variant on the aggregate levels, by a lot** —
−55% at total, −46% at department. That is not what the textbook expects, and
it is a fact about *these* base forecasts rather than about reconciliation.

The reason is straightforward once stated: **the bottom-level model is far
stronger than the aggregate-level models.** The global LightGBM was tuned,
backtested over five folds and given 33 engineered features; the aggregate
forecasts are an out-of-the-box AutoETS and a TSB rate. MinT blends base
forecasts in proportion to their assumed error variance, so blending a good
forecast with a poor one lands between them. Simply summing the good one wins.

**What that implies for next steps** is specific: to make MinT beat bottom-up on
aggregates, the aggregate forecasts have to be competitive — which means
training the global ML model at each level rather than reaching for ETS. That is
a real piece of work, and it is the honest recommendation rather than a claim
that reconciliation "improved accuracy".

### What to actually ship

`wls_var`. It is coherent, it is the best method at the three levels holding
30,490 / 9,147 / 3,049 nodes (which is where the replenishment decisions are
made), and it costs one sparse linear solve. If aggregate-level accuracy were
the priority instead, bottom-up is coherent too and better there — the choice is
between two coherent options, which is a much better place to be than choosing
between incoherent ones.

---

## 11. What is not here

- No Favorita cross-validation (spec's first cut item)
- No Google Trends (second cut item; heavily rate-limited)
- No FRED macro covariates — the client is written and tested but no API key was
  configured, so `core.fact_macro` is empty
- No full MinT with a shrinkage covariance — it needs the 42,840² error
  covariance (~14 GB dense). `wls_var` is the practical variant and the
  literature finds it close.
- Aggregate-level base forecasts are out-of-the-box AutoETS/TSB rather than
  tuned models, which is why bottom-up beats MinT on those levels (see §10).
- No live deployment URL yet — the container builds and runs locally.
