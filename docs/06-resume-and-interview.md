# Resume bullets and interview prep

Every number below is measured and reproducible from this repo. Nothing is
rounded up, and nothing is claimed that the code does not produce.

---

## Resume bullets

Six bullets, each tied to a figure that is reproducible from this repo. Nothing
is rounded in a flattering direction.

**Supply Chain Intelligence** — demand forecasting & inventory optimization

- Built an end-to-end demand forecasting and inventory optimization system over
  **30,490 retail SKU-store series**, integrating transactional, weather and
  calendar data into a DuckDB star schema — 59.2M raw rows reduced to a
  46.9M-row point-in-time-correct analysis panel.

- Engineered a **per-series adaptive stockout detector** that identifies censored
  demand on 3.99% of panel days, replacing a fixed-threshold rule that
  misclassified 18.3% of the panel as censoring and would have biased every
  downstream safety-stock calculation downward.

- Developed a probabilistic forecasting stack (Croston, TSB, global LightGBM with
  Tweedie and quantile objectives) improving **WRMSSE 3.8%** over a tuned
  intermittent-demand baseline, with quantiles calibrated to **within 0.002 of
  nominal** at the 90th, 95th and 99th percentiles on a locked hold-out set.

- Implemented **MinT hierarchical reconciliation across 42,840 nodes** via a
  matrix-free preconditioned conjugate-gradient solve, eliminating an 8.1%
  adding-up disagreement between SKU- and company-level forecasts on hardware too
  small to materialize the 7.4 GB system matrix.

- Translated forecast distributions into a constrained (s,S) replenishment policy
  that **cut simulated inventory cost 5.0% while raising fill rate 1.0 point**
  (95% bootstrap CI: −8.5% to −1.9%), allocating service levels per series
  through an LP relaxation of a **243,920-variable multiple-choice knapsack**,
  with shadow-price analysis identifying shelf space rather than capital as the
  binding constraint.

- Deployed as a **containerized FastAPI service and Streamlit dashboard** with
  live scenario simulation across all 30,490 series, backed by **123 tests**
  covering leakage audits, backtest-protocol integrity and numerical correctness.

### If space is tight

Keep bullets 1, 3, 5 and 6 — data engineering, modelling, business impact,
shipping. Bullets 2 and 4 reward a technical reader but cost two lines each.

### What NOT to claim

- Do **not** say "improved forecast accuracy, reducing inventory cost". The
  measured decomposition says the opposite: a better forecast alone moved cost
  +1.0% with a CI straddling zero. The gain came from the allocation layer.
- Do **not** call the baseline "actual" or "historical". It is reconstructed.
- Do **not** quote the 39% figure that appears in early git history. It was an
  overfit artifact and is documented as such.

---

## The questions, and the honest answers

### Why a global model instead of per-series models?

Four reasons, in order of weight here. Data per series is thin and mostly zero —
68% of observed days are zero — so a local model for an intermittent SKU-store
fits a handful of non-zero points and overfits them. Day-of-week, SNAP and price
effects are properties of the category and store, not of one SKU, so estimating
them 30,490 times separately gets a noisier answer every time. New series work
immediately, which matters when the median launch is d=160 and half the
catalogue starts after d=100. And it is one artifact to train, validate and
monitor rather than 30,490.

The cost is that a genuinely idiosyncratic series gets pulled toward the crowd.
The error decomposition by Syntetos-Boylan class is what makes that visible
rather than assumed.

### Why do you need quantiles rather than point forecasts?

Because the optimizer does not order the expected demand — it orders enough to
hit a service level, which is a statement about the tail of lead-time demand.
Safety stock *is* a quantile. A point forecast contains none of the information
the decision needs.

It matters more here than usual: 45% of revenue sits in the AZ box, high value
*and* high variability, 96% of it intermittent. For those series the mean is a
poor summary and the spread is the entire question.

### How did you handle stockout censoring, and what does it bias?

Observed sales are not demand. On a day a store ran out, sales are zero while
demand was positive; train on that and the model learns to forecast supply
failures, so forecasts fall, safety stock falls, and the optimizer
systematically under-orders.

Detection uses a per-series adaptive threshold, not a fixed rule. Estimate the
daily sale probability *p* from the trailing 56 days; under independence a run
of *k* zeros has probability (1−p)^k, so flag when that drops below α = 0.005 —
which means k > ln(α)/ln(1−p). A series selling 90% of days trips at 3 zeros;
one selling 5% of days needs 97.

**The first version used each series' lifetime rate and flagged 18.3% of the
panel with a mean run of 40 days.** A 40-day stockout is not a stockout — that
was dormancy. The lifetime average cannot see drift. The local rate gives 4.0%,
median run 8 days, which is a plausible grocery figure, and FOODS stocks out
more than HOBBIES and A-items more than C-items, as they should.

Residual bias: α is chosen, not measured. Across 0.001–0.05 the flagged share
moves 2.4%–10.9%, and every downstream cost number inherits that.

### Why does hierarchical reconciliation matter to the business?

Because forecasts are consumed at different altitudes and independent ones do
not add up. Here the total-level forecast says 1,147,983 units while the bottom
level sums to 1,241,178 — **8.1% apart**. Two teams, same 28 days, two plans,
and the disagreement becomes an argument about whose spreadsheet is right.

Reconciliation makes them add up by construction. That is worth more
operationally than the accuracy change.

**And the accuracy result was not the textbook one.** Bottom-up beat every MinT
variant on the aggregate levels, by up to 55%. That is a fact about these base
forecasts, not about reconciliation: the bottom model is tuned and backtested
with 33 features while the aggregate models are out-of-the-box AutoETS, and MinT
blends toward the weaker forecast. `wls_var` still improved the bottom level
0.8% for free. The fix, if aggregate accuracy mattered, is to train the global
model at each level.

### What is your safety stock formula and why not assume normality?

`safety_stock = quantile_SL(lead-time demand) − E[lead-time demand]`, where the
distribution is simulated by **scaled block bootstrap** — draw a lead time, draw
a random start in the series' own clean history, take that many *contiguous*
days, sum, scale to the forecast level.

Contiguous blocks are the point: sampling days independently destroys the
autocorrelation and day-of-week structure and understates the variance of the
sum, which is exactly the error z·σ·√L makes.

Normality fails on three counts here. 72.6% of series are intermittent, so
demand is zero-inflated and right-skewed, nothing like a normal. Demand is
autocorrelated and weekly-seasonal, so √L understates run variance. And lead
time is a distribution, not a constant. Applied to an intermittent series the
normal formula gives negative reorder points at low service levels.

The normal approximation is still implemented — as the comparison arm, so the
cost of assuming normality can be *quantified* rather than asserted.

### What is the shadow price of the warehouse capacity constraint?

Shelf space is worth **5.1¢ per unit per 28 days at CA_3** and **3.9¢ at TX_1** —
about 30% more valuable at the former. The budget constraint's dual is an order
of magnitude smaller at 0.32¢ per dollar.

What I would tell the ops director: **space binds, capital does not.** If space
can be added or reallocated anywhere, CA_3 pays back most; asking for more
capital instead would buy almost nothing. The constraints together cost $2,551
per 28 days against the unconstrained optimum, about 2.6%.

### If service moved from 95% to 98%, what happens to cost?

In this cost regime it moves in the *helpful* direction, which surprises people.
Going from a 95% to a 99% target raises average inventory 31% (15.6 → 20.4
units) but **cuts** total cost 30%, because avoided stockouts are worth far more
than the extra carrying cost.

The reason is the ratio: over a ~12-day exposure window a held unit costs about
$0.025 while a lost unit costs about $1.50 — roughly 26:1. The newsvendor
critical ratio p/(p+h) therefore sits near 0.98, and the measured frontier is
monotone with no interior optimum.

That is entirely a property of the assumed parameters. Halve the stockout
penalty and the relationship reverses — which is what the 27-cell sensitivity
grid is for. The saving's sign holds across all 27 cells; only the magnitude
moves (−0.8% to −8.8%).

### How do you know there is no leakage?

By arithmetic rather than by trust. Every lag and rolling statistic is offset at
least 28 days, so a feature for target day *t* reads only day *t−28* or earlier.
That is a direct multi-step design — it costs short-horizon accuracy, and buys
an audit that is a matter of index arithmetic. `tests/test_features.py` checks
the lag values against the source matrix directly.

The trap most people miss is weather. Nobody has a usable 28-day forecast of
daily rainfall, so feeding actual future weather in would inflate the headline.
The model gets **climatology** — the historical average for that state and
day-of-year, computed only from days strictly before the forecast origin — plus
one observed t−28 anomaly. A test asserts no other weather feature can appear.

### What would you do next?

1. Train the global model at each hierarchy level, so MinT has competitive base
   forecasts to blend and can beat bottom-up on aggregates.
2. Replace the assumed cost parameters with real ones. The holding-to-stockout
   ratio drives the entire answer and is currently a literature figure.
3. A phased rollout with control stores. The counterfactual replays realised
   demand, which was itself shaped by the stockouts the real stores had — so it
   *understates* what a better policy achieves. Only an experiment settles that.

---

## The thing worth leading with

Most portfolio projects of this shape report "my model beat the baseline by X%".
This one measured where the money actually came from and found the forecast was
not it — a better forecast alone moved cost +1.0% with a confidence interval
straddling zero, and the entire gain came from deciding *which series deserve
the stock*.

That is a more useful thing to have learned, and being able to say "the result
contradicted what I expected, here is the decomposition that shows it" is worth
more in an interview than any accuracy number.
