# Assumptions and limitations

This document is written as the project is built, not retrofitted at the end.
An assumption recorded on the day it was made is honest; one reconstructed
later is a guess.

Each entry states **what** was assumed, **why**, and **which direction it
biases the result**.

---

## A1 — Store cities are approximated by one metro per state

**Status:** live, affects all weather features.

M5 discloses each store's state (`CA`, `TX`, `WI`) but not its city. Weather is
pulled for one representative metro per state:

| State | Metro used | Coordinates |
|---|---|---|
| CA | Los Angeles | 34.0522, -118.2437 |
| TX | Houston | 29.7604, -95.3698 |
| WI | Madison | 43.0731, -89.4012 |

**Why this and not something more precise:** the alternative is inventing store
cities, which would look more precise while being fabricated. A stated
approximation beats an unstated invention.

**Bias:** weather features are attenuated. A CA_1 store in San Diego and a CA_3
store in Sacramento get identical weather, so genuine within-state weather
signal is averaged away. This makes the weather covariates *weaker* than they
would be with true store locations — it does not make them misleading. Expect
the measured weather effect to be a lower bound on the real one.

---

## A2 — Sales censor true demand

**Status:** mitigated in week 1. See
[02-methodology.md](02-methodology.md#stockout-detection) for the rule and
[04-results.md](04-results.md#stockout-proxy) for the measured rates.

3.99% of panel days are flagged as likely stockouts (median run 8 days) and a
further 14.27% as dormant. The threshold is derived per series from its own
local demand rate rather than imposed as a fixed number of days.

**Residual, and it is not small:** the detection rests on a significance level
alpha that was chosen, not measured. Across the plausible range 0.001–0.05 the
flagged share moves from 2.37% to 10.86% — a 4.6× swing. Every downstream cost
number inherits that uncertainty, and the week-3 counterfactual must be re-run
across the range rather than reported at a single alpha.

**Also residual:** partial stockouts, where the shelf emptied mid-day and sales
were truncated rather than zeroed, are undetectable in this data and survive as
understated positives.

<details>
<summary>Original entry, before the mitigation was built</summary>

**Status:** known, mitigation planned for week 1 (`stockout_proxy_detection.sql`).

`fact_sales.units` is what was **sold**, not what was **demanded**. On a day
when a store ran out, observed sales are zero or truncated while true demand was
positive. Training directly on this teaches the model to predict stockouts
rather than demand.

**Mitigation:** flag likely stockout days (a zero on a series that is normally
non-zero) and either exclude or impute them before training.

**Bias:** downward on demand, therefore downward on safety stock, therefore the
optimizer will *under*-order if this is not handled. This is the single most
consequential data-quality issue in the project.

**Residual:** the proxy cannot distinguish a genuine zero-demand day from a
stockout with certainty. Some censoring survives.

</details>

---

## A2b — SKU availability is inferred from price presence

**Status:** live, and it is the filter defining `core.panel`.

A missing weekly `sell_price` is treated as "this store did not carry this SKU
yet" rather than as missing data. 20.8% of raw rows are dropped on this basis.

**Why this is safe:** three checks against the data, not assumptions. The
price-NULL block is contiguous and leading for 100.00% of all 30,490 series;
zero rows have `units > 0` with a NULL price; and the median gap from first
priced day to first sale is 2 days.

**Bias:** if the interpretation were wrong, the pipeline would be discarding
real zero-demand days and biasing forecasts *upward*. The checks above are what
rule that out, and `tests/test_analytics.py` pins the filter so it cannot
silently disappear.

---

## A3 — Weekly prices held flat within the week

**Status:** deferred to the feature layer.

`sell_prices` is weekly. Daily models need a daily price. The price for a
Walmart fiscal week is broadcast to all seven days of that week.

**Why:** it is what the source supports. Intra-week price variation is not
observable.

**Bias:** promotion start and end effects are smeared to week boundaries. A
midweek price drop appears to have started on the Saturday.

---

## A4 — Macro series forward-filled, never interpolated

**Status:** enforced by construction — macro is stored at native frequency and
resampled only in the feature layer.

Monthly CPI and weekly gas prices are forward-filled to daily. They are never
interpolated, because interpolation between month *t* and month *t+1* uses the
*t+1* value on days before it was published. That is leakage.

**Bias:** none toward leakage, which is the point. It does mean the daily macro
series is a step function, so any model reading it sees discrete jumps rather
than smooth trends.

---

## A5 — The observed baseline is reconstructed, not observed

**Status:** live. This is the assumption the headline number rests on entirely.

M5 contains no inventory data — no stock levels, no purchase orders, no
receipts. So there is no observed replenishment policy to compare against.

**The reconstruction:** an (s, S) policy of the same structure, driven by what a
conventional replenishment system actually does — a 28-day moving-average
forecast and textbook normal-approximation safety stock — with its
service-level parameter bisected until its *simulated* fill rate matches the
fill rate implied by the stockout proxy on real data.

**Why calibrate rather than pick a number:** a retailer running a naive system
is not aiming at 95% and hitting it. They achieve whatever the data says they
achieved. Tuning the baseline to fail as often as the real stores failed is
what stops it being a strawman that the new method beats trivially.

**Bias:** unknown in direction, and it could be large. If real stores ran a
*better* policy than a moving average with normal safety stock, the baseline is
too weak and the reported saving is overstated. If they ran worse — manual
ordering, fixed par levels — it is understated. Nothing in M5 can settle this.

**Second-order assumption inside it:** converting a stockout *day rate* (3.99%)
into a unit *fill rate* requires assuming how much demand was lost on a stockout
day. We assume the series' own average daily demand. Real stockouts are more
likely on high-demand days, so this probably understates lost units and
therefore overstates the observed fill rate.

---

## A6 — Every cost parameter is assumed

**Status:** live, with full-factorial sensitivity reported.

| Parameter | Default | Basis |
|---|---|---|
| gross margin | 0.25 | US grocery retail typically 22–28% |
| holding rate | 0.25/year | Standard retail carrying cost: capital, warehousing, shrinkage, obsolescence |
| stockout penalty | 1.5 × lost margin | Lost margin plus something for the customer who goes elsewhere |
| order cost | $0.50/line | Marginal admin cost of one more line on a truck already going to the store |

None of these are in the data. M5 gives a selling price and nothing else
economic.

**This matters more than it looks.** The ratio between holding and stockout cost
determines the entire answer. Over a ~12-day exposure window a held unit costs
about $0.025 while a lost unit costs about $1.50 — a factor of ~26. That single
ratio is why the measured service-level frontier has no interior minimum.

Sensitivity is reported across a full factorial rather than one-at-a-time,
because holding cost and stockout penalty push the optimum in opposite
directions and can partially cancel.

**The defensible claim is never "this saves X". It is "this saves X under these
stated assumptions, and here is how X moves when they change."**

---

## A7 — Lead time and pack sizes are invented

**Status:** live.

Lead time is assumed uniform on 3–7 days, typical for grocery DC-to-store
replenishment. M5 has no supplier data at all.

Case packs are assigned by price band (cheap items in 12s, expensive in 1s) on
the reasoning that cheap fast-movers ship in bigger cases. This is a fabrication
with a plausible shape. Its only real effect is to coarsen order quantities,
which raises average inventory slightly and reduces the order count.

**Bias:** a longer or more variable lead time would raise safety stock across
the board. The comparison between methods is largely insulated, since every
policy faces the same assumed distribution.

---

## A8 — Structural limits of the model

- **No supplier-side constraints.** No supplier minimums, no truck capacity, no
  order-frequency limits beyond the review period.
- **No multi-echelon transshipment.** Stores are modelled independently. Real
  networks move stock laterally, which reduces total safety stock materially —
  so a genuine multi-echelon system would beat everything reported here.
- **No substitution.** A stockout is modelled as a fully lost sale. In grocery,
  customers frequently buy the adjacent brand, so the true cost of a stockout is
  lower than modelled for substitutable goods and higher for destination items.
- **No shelf-life or spoilage**, despite FOODS being 47% of the catalogue.

---

## A9 — The counterfactual is a simulation, not an experiment

**Status:** live, and it is the deepest limitation in the project.

Every policy is evaluated by replaying **the same realised demand** through a
simulated inventory system. That embeds a real circularity: observed sales were
themselves shaped by the stockouts the real stores had, and a policy that
prevented those stockouts would have sold more than the data records.

The stockout proxy mitigates this for training but cannot manufacture the demand
that was never observed. So the simulation systematically understates what a
better policy would have achieved.

A real deployment would need a phased rollout with a control group of stores —
not a replay.

---

## A10 — Promotions are inferred from price, and the lift estimate is not causal

**Status:** live, affects `promo_lift_analysis.sql` only. Nothing here feeds a
model yet.

M5 has no promotion flag, so promotions are inferred as weeks priced below a
trailing 8-week reference maximum.

**The measured result is odd and worth stating plainly:** deep discounts lift
demand a lot (FOODS 20–35% off: +103%), but *shallow* discounts associate with
**lower** demand (FOODS 5–20% off: −27% to −30%). The effect survives both a
volume-weighted and an unweighted per-cell estimator, so it is not an
aggregation artifact.

**Most likely cause: endogeneity, not a real negative price response.**
Retailers apply modest markdowns reactively to products already slowing down.
Matched controls hold item, store, weekday and quarter fixed, but they cannot
hold fixed the retailer's private reason for discounting.

**Consequence for later work:** do not read a promotional uplift factor off this
table into the optimizer. Price belongs in the model as a controlled feature.

**One method note:** the promotion-vs-markdown split uses a forward-looking
price-recovery test. That is legitimate in a descriptive report run once over
history, and would be leakage in the feature layer. Promo features that reach
`features/` must be built from trailing information only.

---

## Engineering decisions worth defending

**`fact_sales` carries no PRIMARY KEY.** DuckDB materialises an ART index for a
PK. A three-column index over 59.2M rows costs real memory on a laptop and buys
nothing here, because every access path to this table is a full or filtered
scan. The uniqueness constraint on `(item_id, store_id, d)` is still enforced —
by `tests/test_warehouse.py`, not by the insert path. The constraint is
checked; it just isn't paid for on every write.

**Bulk data lives outside the repo folder.** `SCI_DATA_DIR` defaults to
`C:/sci-data` rather than `./data`, because the repo sits inside a OneDrive
synced folder and a multi-gigabyte DuckDB file that gets rewritten on every
build would be re-uploaded each time.
