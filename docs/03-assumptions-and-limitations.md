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

## To be filled in as the project progresses

- **A5** — reconstruction method for the observed baseline replenishment policy _(week 3)_
- **A6** — holding cost `h`, stockout penalty `p`, ordering cost `K` _(week 3, with sensitivity analysis)_
- **A7** — lead time distribution _(week 3)_
- **A8** — no supplier-side constraints or multi-echelon transshipment modelled
- **A9** — the counterfactual is a simulation, not an experiment

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
