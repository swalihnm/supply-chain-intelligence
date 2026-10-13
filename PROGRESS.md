# Build progress

Single source of truth for where this build stands. **Update this file at the
end of every working session**, before the context runs out.

Spec: `PROJECT-01-supply-chain-forecasting.md` (the original brief).

---

## Current state

**Feature-complete. All four weeks plus hierarchical reconciliation, built and
run end to end. 123 tests pass. The container is BUILT AND VERIFIED as of
2026-09-12. The only thing left is hosting it somewhere public.**

| Week | Deliverable | Status |
|---|---|---|
| 1 | Ingestion + warehouse + SQL analytics | ✅ complete |
| 2 | Feature store + model ladder + backtest harness | ✅ complete |
| 3 | Optimizer + counterfactual + shadow prices | ✅ complete |
| 4 | API + dashboard + docs | ✅ complete (container built + verified) |
| + | Hierarchical reconciliation (42,840 nodes) | ✅ complete |

### Headline

Per-series cost-optimal allocation vs reconstructed baseline, on the locked test
window: **−5.0% cost at +1.0 point fill rate**, CI −8.5% to −1.9%. Every one of
27 cost-assumption cells shows a saving.

Global LightGBM: **0.8446 WRMSSE** on the locked test set vs 0.8626 for the best
baseline (TSB).

Reconciliation: base forecasts disagree by 8.1% at the total level; all four
methods drive the adding-up violation to exactly zero. `wls_var` improves the
bottom level 0.8% for free, but bottom-up wins on aggregate levels because the
bottom model is much stronger than the aggregate ones.

### What is genuinely left

1. **Host the container somewhere public** — the only remaining deliverable,
   and it is blocked on ONE interactive step: `fly auth login` opens a browser.

   Everything else is prepared and verified:
   - Docker Desktop 29.7.2 installed; image built and running locally
   - `fly.toml` written, with the `PORT` override verified working in-container
     (`docker run -e PORT=8080` -> Streamlit binds 8080, health returns ok)
   - 1 GB not 512 MB: the dashboard idles at ~76 MB but "Run simulation"
     simulates all 30,490 series and would be OOM-killed on the smaller machine
   - `min_machines_running = 0` to stay inside the free allowance

   ```bash
   winget install -e --id Fly.Flyctl
   fly auth login                          # <- the user's step, opens a browser
   uv run python scripts/stage_deploy.py
   fly launch --no-deploy --copy-config    # first time only
   fly deploy
   ```

   Then replace the "not yet hosted" line at the top of the README with the URL.

   **Why Fly and not Render/Railway:** `deploy/data/artifacts/` is gitignored, so
   a repo-connected build ships an image with no artifacts — the app starts,
   finds no serving bundle, and shows its "artifacts not built" error rather than
   crashing, which makes it easy to misdiagnose. Fly pushes the local image, so
   the gitignore is irrelevant. Full detail in `docs/05-deployment.md`.

2. Optional: FRED macro covariates (client written, dormant, needs a free key);
   Favorita cross-validation; Google Trends. All three are explicit cut items in
   the spec.

3. Optional: full MinT with a shrinkage covariance — see decision 16 for why
   `wls_var` is used instead.

### Reproduce everything

```bash
uv sync --extra dev --extra models --extra optimize --extra app
uv run sci ingest all
uv run sci warehouse build
uv run python scripts/run_model_ladder.py      # ~40 min
uv run python scripts/run_optimizer.py         # ~15 min
uv run python scripts/run_reconciliation.py    # ~2 min
uv run python scripts/build_serving_bundle.py
uv run python scripts/make_figures.py          # docs/frontier.png
uv run pytest                                  # 123 tests
```

---

## Environment (verified working)

- Repo: `c:\Users\swali\OneDrive\Desktop\project 1` (git, branch `main`)
- Bulk data: `C:\sci-data` — **deliberately outside OneDrive**, set by `SCI_DATA_DIR` in `.env`
- Python 3.12 via `uv`. `uv sync --extra dev` installs; extras: `models`, `optimize`, `app`
- M5 zip already unpacked to `C:\sci-data\raw\m5` (the original zip in the repo folder can be deleted)
- `FRED_API_KEY` is **not set** — macro ingest skips cleanly, nothing depends on it
- Google Trends deliberately skipped (first in the spec's cut order)

### Rebuild from scratch

```bash
uv sync --extra dev
uv run sci ingest all        # M5 from local zip + Open-Meteo weather
uv run sci warehouse build   # ~4 min -> ~1.9 GB DuckDB
uv run pytest                # 123 tests
```

---

## What exists

### Warehouse tables (`C:\sci-data\warehouse\sci.duckdb`)

| Table | Rows | Built by |
|---|---:|---|
| `core.fact_sales` | 59,181,090 | load from staging parquet |
| `core.fact_price` | 6,841,121 | load |
| `core.fact_weather` | 6,210 | load |
| `core.dim_date` / `dim_item` / `dim_store` | 1,969 / 3,049 / 10 | load |
| `core.panel` | 46,881,677 | `sql/transform/010_panel.sql` |
| `core.zero_runs` | 6,613,508 | `sql/transform/020_stockout_proxy.sql` |
| `core.stockout_flags` | 254,122 | same file, filtered subset |
| `core.series_class` | 30,490 | `sql/transform/030_series_class.sql` |

`core.fact_macro` exists but is empty (no FRED key).

### Code

- `sci/` — config, logging, checksum manifest, warehouse runner, Typer CLI
- `features/` — `panel_matrix.py` (dense arrays), `build_features.py` (33 features)
- `models/` — `baselines.py`, `global_ml.py`, `quantile.py`, `statistical.py`,
  `hierarchy.py`, `reconcile.py`
- `optimize/` — `cost_model.py`, `lead_time_demand.py`, `policy.py`,
  `expected_cost.py`, `constrained_solver.py`
- `evaluation/` — `protocol.py` (frozen), `metrics.py`, `backtest.py`,
  `simulate_counterfactual.py`
- `app/` — `api.py`, `dashboard.py`, `viz.py`, `store.py`
- `scripts/` — `run_model_ladder.py`, `run_optimizer.py`,
  `run_reconciliation.py`, `build_serving_bundle.py`, `stage_deploy.py`,
  `make_figures.py`
- `docs/` — 01 sources, 02 methodology, 03 assumptions, 04 results,
  05 deployment, 06 resume + interview prep, `frontier.png`
- `fly.toml` — Fly.io config; `PORT` override verified working in-container
- `ingest/` — `m5_loader.py`, `weather_client.py`, `fred_client.py` (dormant)
- `sql/ddl/` → `sql/transform/` → `sql/analytics/` (pipeline order matters)
- `tests/` — 123 tests, all passing

---

## Week 2–3 decisions — do not silently revisit

7. **Backtest protocol is frozen in `evaluation/protocol.py`** and imported
   everywhere. 5 rolling-origin folds, 28-day horizon, locked test set d1914–1941
   scored once. `tests/test_protocol.py` asserts no overlap and no test leakage.

8. **Direct multi-step design: every lag ≥ 28 days.** No recursion, so the
   leakage audit is arithmetic. Costs some short-horizon accuracy; worth it.

9. **Weather is CLIMATOLOGY only** (state × day-of-year, from days before each
   fold origin) plus a t−28 anomaly. Actual future weather would be leakage —
   nobody has a 28-day rainfall forecast. A test asserts no other weather
   feature can appear.

10. **Policies are compared at MATCHED achieved fill rate**, found by bisection.
    A service-level *parameter* and a fill rate are different quantities and the
    normal and empirical methods convert between them differently. An earlier
    grid-based version left a 0.56-point fill gap and reported a 9% cost
    difference that was purely the service gap.

11. **Service levels are chosen on EXPECTED cost** (`optimize/expected_cost.py`),
    derived from the forecast's lead-time demand distribution — never from
    realised demand. Choosing on realised cost produced a 39% "saving" that was
    hindsight. Realised demand is touched once, to evaluate the chosen allocation.

12. **The realised-cost frontier is monotone** to 99.9% — no interior optimum,
    because a lost unit costs ~26× holding one over the exposure window. The
    *expected*-cost curve does turn, at 0.99. This is why the recommendation is a
    per-series allocation rather than a uniform target.

13. **LP relaxation, not MIP**, for the constrained allocation. At most
    (#side-constraints) series can be fractional — measured 11 of 30,490, against
    a bound of 11 (10 store rows + 1 budget row).
    Fractional assignments round toward the SMALLER footprint so capacity rows
    stay feasible.

14. **Hierarchical reconciliation is solved MATRIX-FREE.** `S'W^-1S` is
    30,490x30,490 and dense (every pair co-occurs at `total`), which needs ~7.4 GB
    on a 7.3 GB machine. Preconditioned CG only needs `S'(W^-1(Sv))` — two sparse
    mat-vecs over 365,880 non-zeros. `tests/test_reconcile.py` verifies it against
    the dense closed form on a small hierarchy.

15. **Statistical models (AutoETS) run on AGGREGATE levels, not the bottom.**
    ETS on an intermittent SKU-store series fits noise; aggregates are smooth and
    weekly-seasonal, which is ETS's home ground. Use each method where its
    assumptions hold. Fitted via statsforecast's parallel batch API — a serial
    per-node loop had not cleared 2,000 of 12,350 fits after two minutes.

16. **Full MinT (shrinkage covariance) is NOT implemented.** It needs the
    42,840^2 error covariance, ~14 GB dense. `wls_var` is the practical variant
    and the literature finds it close to full MinT.

---

## Week 1 decisions — do not silently revisit

1. **Availability filter.** `core.panel` inner-joins sales to price. A missing
   weekly `sell_price` means the store did not carry that SKU. Verified: the
   price-NULL block is contiguous and leading for 100.00% of all 30,490 series,
   and no row has `units > 0` with a NULL price. Drops 20.8% of raw rows.

2. **Stockout `p` is local (trailing 56 days), not lifetime.** The lifetime
   version flagged 18.3% of the panel with a mean run of 40 days — dormancy, not
   stockout. Current rule: 3.99%, median run 8 days.

3. **Runs over 28 days are `dormant`, not `stockout`.** Stockouts get imputed;
   dormant stretches get excluded. Collapsing them would impute phantom demand
   into dead SKUs.

4. **`core.zero_runs` is kept unfiltered** so the alpha sensitivity report can
   loosen alpha as well as tighten it.

5. **`fact_sales` has no PRIMARY KEY.** A 3-column ART index over 59M rows costs
   memory for no query benefit. Uniqueness is enforced in `tests/`.

6. **Promo lift is exposure-weighted**, not a ratio of unweighted means, and
   both arms require 5+ observations per cell.

---

## Key numbers

**`docs/04-results.md` is authoritative** — it carries every figure with its
context and caveats, and is regenerated from the artifacts. This is the
quick-reference subset for a session handoff.

### Data and analytics (week 1)

- 59,181,090 raw rows -> 46,881,677-row panel (20.8% dropped as not-yet-stocked)
- Availability filter verified: price-NULL block contiguous and leading for
  **100.00%** of all 30,490 series; zero rows with units > 0 and NULL price
- 91% of series are intermittent or lumpy (Syntetos-Boylan)
- 45.1% of revenue in the AZ box (high value, high variability), 96% intermittent
- Revenue is NOT 80/20: it takes 36.8% of series to reach 80% of revenue
- Stockout proxy: **3.99%** of panel days; dormant a further 14.27%
- Alpha sensitivity: 2.37%-10.86% across 0.001-0.05 (a 4.6x swing)
- SNAP lifts FOODS **+7.7%**, HOUSEHOLD +1.9%, HOBBIES +1.3%
- Weekend lift +18.7% to +25.6% by category
- Naive elasticity: FOODS -0.43, HOUSEHOLD -0.36, HOBBIES -0.26 (weak R², not causal)
- Promo: deep discounts +103% (FOODS 20-35%); shallow 5-20% associate with
  LOWER demand (-27% to -30%) - most likely endogeneity, not a price response

### Forecasting (week 2)

| Model | CV WRMSSE | Test WRMSSE |
|---|---:|---:|
| **global LightGBM (Tweedie)** | **0.8559** | **0.8446** |
| TSB | 0.8893 | 0.8626 |
| Croston | 0.8910 | 0.8639 |
| moving average | 0.8950 | 0.8720 |
| seasonal naive | 1.1563 | 1.1017 |
| zero forecast | 1.3963 | 1.4132 |

- 3.8% CV improvement over a tuned TSB baseline
- The zero forecast has the BEST MASE (0.986) and worst WRMSSE - metric choice
  is a modelling decision
- Calibration on the locked test fold: q0.90 -> 0.9007, q0.95 -> 0.9505,
  q0.99 -> 0.9885 (gaps under 0.002). q0.50 -> 0.638, which is the zero point
  mass, not a defect
- Conformal was worse than direct quantile regression at 0.90 and 0.95
- Quantile crossing repaired on 2.7% of cells

### Decision (week 3)

| Policy | Fill rate | Cost (28d) | vs baseline | 95% CI |
|---|---:|---:|---:|---|
| reconstructed baseline | 94.05% | $127,930 | - | - |
| better forecast only | 94.03% | $129,254 | +1.0% | -1.3% to +3.5% |
| empirical SS, uniform level | 94.02% | $133,976 | +4.7% | +1.4% to +8.3% |
| **per-series allocation** | **95.06%** | **$121,500** | **-5.0%** | **-8.5% to -1.9%** |

- Unmet units 73,263 -> 60,805 (**-17%**) while spending 5% less
- **The gain came from the ALLOCATION, not the forecast.** A better forecast
  alone moved cost +1.0% with a CI straddling zero
- Sensitivity: all 27 cost-assumption cells show a saving; -0.77% to -8.76%,
  median -5.16%
- Frontier is MONOTONE to 99.9% - no interior optimum, because a lost unit costs
  ~26x holding one over the ~12-day exposure window
- Shadow prices: shelf space worth 5.1c/28d at CA_3, 3.9c at TX_1; budget dual
  0.32c - **space binds, capital does not**. Constraints cost $2,551 (2.6%)
- LP relaxation left 11 fractional assignments of 30,490 - exactly the
  theoretical bound of one per side constraint

### Reconciliation

- 42,840 nodes over 30,490 bottom series, 12 levels
- Base forecasts disagree **8.1%** at the total level (1,147,983 vs 1,241,178);
  max single-node violation 7,934 units -> **0.0** for all four methods
- `wls_var` improves the bottom level 0.8% for free (0.8176 vs 0.8241)
- **bottom_up beats every MinT variant on aggregate levels** (up to 55% at
  total) - a fact about these base forecasts, not about reconciliation

### Engineering

- Warehouse ~1.9 GB on disk (~1.1 GB live, rest free blocks); rebuilds in ~4 min
- Image 1.4 GB; API ~209 MB resident, dashboard ~76 MB; /simulate ~42 s
- 123 tests, verified passing with 0.43 GB free RAM

## Open items / known risks

- **Shallow-discount negative promo lift** (−27% to −30% at 5–20% off) is
  unexplained. Survives three robustness checks. Logged as A10; most likely
  endogeneity. Do not read an uplift factor off that table into the optimizer.
- **Compute budget.** 30,490 series × 5 folds makes per-series statistical
  models (AutoETS/AutoARIMA) expensive. Plan: run those on a stratified sample,
  run global ML on everything, and document the split honestly.
- FRED macro covariates absent. Optional; nothing blocks on them.

---

## Session log

### Session 1 — 2026-09-08
Scaffolded the repo, ingested M5 + weather, built the star schema, the panel
transform, the stockout proxy and all five analytics reports. Two self-corrections
during the session (lifetime→local stockout rate; broken alpha sensitivity) are
written up in `docs/02-methodology.md` rather than quietly patched.

### Session 2 — 2026-09-09

Built weeks 2, 3 and 4. Ran the full model ladder and optimizer end to end.

Four self-corrections, all written into the docs rather than quietly patched:

1. **Comparison was rigged.** Policies ran at different achieved service levels,
   so a "32% more expensive" result was measuring the service gap, not the
   method. Fixed by bisecting every policy onto the same fill rate.
2. **Allocation overfit.** Service levels were chosen by simulating each
   candidate on realised test demand and taking the per-series minimum -
   selection on the outcome. A 39% "saving" that was hindsight. Now chosen on
   forecast-derived expected cost (`optimize/expected_cost.py`).
3. **False claim in a docstring.** `cost_model.service_level` asserted cycle
   service always exceeds fill rate. It does not - which is higher depends on
   whether failures are rare-and-large or frequent-and-small. A test now pins
   both orderings.
4. **Dashboard silently blank.** Renaming the recommended policy left the
   executive view filtering for a name that no longer existed; it rendered
   nothing rather than erroring. Caught by `tests/test_dashboard.py`, which runs
   every view through Streamlit's AppTest.

The headline finding was not the expected one: a better forecast bought nothing
(CI spans zero), an empirical safety-stock distribution at a uniform service
level was *worse* than the textbook normal approximation, and the entire gain
came from the per-series allocation. That is reported as found.

### Session 3 — 2026-09-09

Built hierarchical reconciliation and the statistical rung. Two more
self-corrections, both written into the docs:

5. **AutoETS on all 12,350 aggregate nodes** took over 20 minutes and drove the
   machine into swap — for `item` and `item x state`, which are still
   intermittent and where smoothing has least to offer. Now routed: AutoETS on
   the 154 smooth nodes, TSB on the 12,196 sparse ones. Seconds.
6. **Bottom-up beat every MinT variant on aggregate levels.** Not the textbook
   result. It is a fact about these base forecasts — the bottom model is tuned
   and backtested, the aggregate models are out of the box — and it is reported
   that way, with the concrete next step rather than a spun conclusion.

**Docker was attempted and is genuinely blocked**, not skipped:
- Docker Desktop's installer needs an elevated shell; the agent cannot click
  the UAC prompt. (An earlier version of this line said the user is not an
  admin - wrong; see the correction under "What is genuinely left".)
- WSL2 (Ubuntu 24.04) is installed and virtualization is on, but `sudo` needs a
  password that cannot be supplied non-interactively.
- Rootless Docker needs `uidmap` and `iptables`; neither is present and both
  need sudo to install.

Prerequisites that ARE in place: WSL2, cgroup2, subuid/subgid configured for the
user, 954 GB free. The one-time command the user must run is in
"What is genuinely left" above.

### Session 4 — 2026-09-12

Docker Desktop 29.7.2 installed by the user via an elevated `winget install`.
The image was then **built and verified for the first time**, and it worked on
the first build — no Dockerfile changes needed.

Verified in-container:
- both services healthy (`api` :8000, `dashboard` :8501)
- `/health` loads all 19 artifact tables
- `/forecast` and `/policy` return correct per-series values
- `/simulate` runs the full 30,490-series simulation in ~42 s and returns
  numbers IDENTICAL to the local run (fill 0.9103, cost 171,736) — which is the
  real proof the serving bundle survived the copy intact
- the dashboard container loads the store and builds Plotly figures (checked by
  exec'ing a script, since Streamlit only runs its script on websocket connect,
  so an HTTP 200 alone proves nothing)
- no Traceback/ERROR lines in either container's logs

Image 1.4 GB; api ~209 MB resident, dashboard ~76 MB.

Also corrected a factual error in this file: it had claimed the user is not a
Windows administrator. `IsInRole(Administrator)` reports False for any
non-elevated shell regardless of group membership; the account is in fact in the
Administrators group. The real constraint was always narrower — an agent cannot
click a UAC prompt.

### Session 5 — 2026-09-14

Closed the last gaps found by auditing the repo against the spec rather than
against memory.

**Bug found by rendering a chart.** `docs/frontier.png` came out with a "nan%"
label and the recommended-point ring on the wrong marker. Cause: Polars treats
null and NaN as different, and the baseline row carries NaN in
`target_service_level` — so `is_not_null()` alone let it through and it was
drawn as a ninth point ON the frontier line. **The dashboard had the identical
bug** (`app/dashboard.py`), shipped and unnoticed, because nothing had ever
looked at that chart closely. Both now filter `is_not_null() & is_not_nan()`.

That chart also originally marked a point on the uniform-target curve as
"recommended". It is not: the recommendation allocates a service level per
series and sits off the curve entirely. It is now a separate star, which makes
the dominance result (cheaper AND higher service) visible at a glance.

Added:
- `scripts/make_figures.py` + `docs/frontier.png`, embedded in the README. The
  spec calls the frontier the best chart in the project for an interview, and a
  README that only described it had not delivered it.
- `fly.toml`, with the `PORT` override verified in a real container run.
- `docs/05-deployment.md` — local (verified) and Fly, plus the gitignore gotcha.
- `docs/06-resume-and-interview.md` — resume bullets with measured numbers, a
  "what NOT to claim" list, and cold answers to all seven spec questions.

Audited: no TODO/FIXME/placeholder left anywhere; all relative markdown links in
README, PROGRESS and docs resolve (checked programmatically, 0 broken).

Not done: `docs/architecture.png`. The README's ASCII diagram renders everywhere
including plain text and stays correct when the pipeline changes; a PNG would be
a second thing to keep in sync for no gain in clarity. Noted as a deliberate
choice rather than an omission.

### Session 6 — 2026-10-13

Repo published to https://github.com/swalihnm/supply-chain-intelligence (public).
History rewritten to remove AI-attribution trailers and re-author every commit as
`swalihnm <swalihnm@users.noreply.github.com>`; MIT LICENSE added with a note
that no M5 data is redistributed.

**A flaky-test class of bug, found by running the suite under memory pressure.**
Two tests - the fact_sales grain check (count DISTINCT over 59.2M rows) and
promo_lift_analysis (window functions over the 46.9M-row panel) - passed on an
idle machine and failed with OutOfMemoryException once Docker Desktop and two
containers were holding a couple of gigabytes.

DuckDB defaults to ~80% of system RAM and then FAILS rather than spilling when
that estimate is wrong. `sci.warehouse.connect()` now sets an explicit
`memory_limit` (2 GB, `SCI_DUCKDB_MEMORY_LIMIT`) and a `temp_directory` under
`SCI_DATA_DIR/tmp`, so heavy queries spill to disk. Both tests now route through
that function instead of a bare `duckdb.connect()`. Verified: 123/123 pass with
0.43 GB free RAM.

**Corrected a stale figure.** Five places claimed the warehouse is 163 MB. It is
~1.9 GB (1.1 GB live, the rest free blocks). The 163 MB was measured at the end
of week 1, before the transform layer added core.panel and core.zero_runs - 53M
rows that were never reflected back into the number.

---

<!-- METRICS:START - generated by scripts/make_metrics_appendix.py -->

## Complete metrics appendix

**Generated by `scripts/make_metrics_appendix.py` directly from the artifact
CSVs and the warehouse — not transcribed.** Re-run it after any pipeline
change. Everything between the markers is overwritten; edit the generator,
not this block.

`docs/04-results.md` remains the place for interpretation and caveats; this is
the raw measured record.

### Warehouse contents

| table_name | rows |
|---|---|
| fact_sales | 59,181,090 |
| panel | 46,881,677 |
| fact_price | 6,841,121 |
| zero_runs | 6,613,508 |
| stockout_flags | 254,122 |
| series_class | 30,490 |
| fact_weather | 6,210 |
| dim_item | 3,049 |
| dim_date | 1,969 |
| dim_store | 10 |
| fact_macro | 0 |

File on disk **1.97 GB** (7,527 blocks x 256 KB): **1.08 GB live**, 0.90 GB free blocks. DuckDB does not return pages to the OS after `CREATE OR REPLACE`, so the file grows across rebuilds without the data growing.


### ABC x XYZ segmentation

| box | series | pct_series | revenue | pct_revenue | avg_cv | avg_adi | pct_intermittent_or_lumpy | policy_hint |
|---|---|---|---|---|---|---|---|---|
| AX | 83 | 0.2700 | 5,996,408 | 3.13 | 0.4200 | 1.01 | 0.0000 | tight safety stock, high service |
| AY | 2,731 | 8.96 | 60,787,832 | 31.73 | 0.8400 | 1.22 | 22.4 | core optimizer focus |
| AZ | 8,407 | 27.57 | 86,474,724 | 45.14 | 1.41 | 2.18 | 96 | widest quantiles, most stock |
| BY | 269 | 0.8800 | 931,279 | 0.4900 | 0.9100 | 1.31 | 48 | standard policy |
| BZ | 9,157 | 30.03 | 27,807,859 | 14.52 | 1.85 | 3.51 | 99.6 | standard policy |
| CY | 48 | 0.1600 | 63,908 | 0.0300 | 0.9200 | 1.33 | 47.9 | standard policy |
| CZ | 9,795 | 32.13 | 9,515,535 | 4.97 | 2.73 | 7.61 | 100 | simple rule, do not over-engineer |


### Stockout proxy

**Headline** — runs | censored days | series | avg len | median | max

| label | metric_a | metric_b | pct_of_panel | metric_c | metric_d | metric_e | metric_f |
|---|---|---|---|---|---|---|---|
| dormant | 65,316 | 6,689,647 | 14.269 | 23,785 | 102.4 | 74 | 1,770 |
| stockout | 188,806 | 1,868,039 | 3.985 | 27,215 | 9.9 | 8 | 28 |

**By ABC / demand class** — series | stockout days | % days | dormant days | % dormant

| label | metric_a | metric_b | pct_of_panel | metric_c | metric_d |
|---|---|---|---|---|---|
| A / erratic | 788 | 60,503 | 4.397 | 42,297 | 3.1 |
| A / intermittent | 6,475 | 508,291 | 4.529 | 1,327,467 | 11.8 |
| A / lumpy | 2,210 | 214,797 | 5.408 | 568,408 | 14.3 |
| A / smooth | 1,748 | 94,398 | 3.314 | 64,247 | 2.3 |
| B / erratic | 61 | 3,202 | 4.652 | 1,542 | 2.2 |
| B / intermittent | 7,304 | 455,730 | 4 | 1,663,126 | 14.6 |
| B / lumpy | 1,949 | 169,147 | 5.26 | 512,971 | 16 |
| B / smooth | 112 | 3,717 | 3.729 | 311 | 0.3000 |
| C / erratic | 6 | 241 | 5.838 | 0.0000 | 0.0000 |
| C / intermittent | 8,364 | 276,347 | 2.565 | 2,131,898 | 19.8 |
| C / lumpy | 1,453 | 81,313 | 4.288 | 377,380 | 19.9 |
| C / smooth | 20 | 353 | 3.994 | 0.0000 | 0.0000 |

**Alpha sensitivity** — runs | days | % of panel

| label | metric_a | metric_b | pct_of_panel |
|---|---|---|---|
| alpha=0.001 | 99,322 | 1,108,851 | 2.365 |
| alpha=0.005 | 188,806 | 1,868,039 | 3.985 |
| alpha=0.010 | 267,174 | 2,425,736 | 5.174 |
| alpha=0.050 | 715,249 | 5,091,514 | 10.86 |


### Seasonality profile


**Day of week**

| cat_id | segment | n_obs | mean_units | lift | pct_vs_baseline |
|---|---|---|---|---|---|
| FOODS | Saturday | 2,513,442 | 3.1164 | 1.187 | 18.7 |
| FOODS | Sunday | 2,511,769 | 3.1498 | 1.199 | 19.9 |
| FOODS | Monday | 2,491,613 | 2.5381 | 0.9660 | -3.4 |
| FOODS | Tuesday | 2,487,772 | 2.3489 | 0.8940 | -10.6 |
| FOODS | Wednesday | 2,486,850 | 2.3143 | 0.8810 | -11.9 |
| FOODS | Thursday | 2,489,014 | 2.3254 | 0.8850 | -11.5 |
| FOODS | Friday | 2,495,345 | 2.5918 | 0.9870 | -1.3 |
| HOBBIES | Saturday | 1,065,520 | 1.0299 | 1.225 | 22.5 |
| HOBBIES | Sunday | 1,064,761 | 0.9352 | 1.112 | 11.2 |
| HOBBIES | Monday | 1,057,447 | 0.7839 | 0.9320 | -6.8 |
| HOBBIES | Tuesday | 1,057,047 | 0.7503 | 0.8920 | -10.8 |
| HOBBIES | Wednesday | 1,057,110 | 0.7580 | 0.9010 | -9.9 |
| HOBBIES | Thursday | 1,057,650 | 0.7572 | 0.9000 | -10 |
| HOBBIES | Friday | 1,058,948 | 0.8720 | 1.037 | 3.7 |
| HOUSEHOLD | Saturday | 1,930,631 | 1.38 | 1.256 | 25.6 |
| HOUSEHOLD | Sunday | 1,929,070 | 1.3349 | 1.215 | 21.5 |
| HOUSEHOLD | Monday | 1,914,782 | 1.0376 | 0.9440 | -5.6 |
| HOUSEHOLD | Tuesday | 1,912,557 | 0.9475 | 0.8620 | -13.8 |
| HOUSEHOLD | Wednesday | 1,912,020 | 0.9359 | 0.8520 | -14.8 |
| HOUSEHOLD | Thursday | 1,913,387 | 0.9461 | 0.8610 | -13.9 |
| HOUSEHOLD | Friday | 1,917,256 | 1.109 | 1.009 | 0.9000 |

**SNAP days**

| cat_id | segment | n_obs | mean_units | lift | pct_vs_baseline |
|---|---|---|---|---|---|
| FOODS | non_snap_day | 11,695,357 | 2.4902 | 0.9230 | -7.7 |
| FOODS | snap_day | 5,780,448 | 2.9061 | 1.077 | 7.7 |
| HOBBIES | non_snap_day | 4,976,488 | 0.8338 | 0.9870 | -1.3 |
| HOBBIES | snap_day | 2,441,995 | 0.8564 | 1.013 | 1.3 |
| HOUSEHOLD | non_snap_day | 9,011,019 | 1.0856 | 0.9810 | -1.9 |
| HOUSEHOLD | snap_day | 4,418,684 | 1.1274 | 1.019 | 1.9 |

**Month**

| cat_id | segment | n_obs | mean_units | lift | pct_vs_baseline |
|---|---|---|---|---|---|
| FOODS | 01 | 1,476,862 | 2.5146 | 0.9550 | -4.5 |
| FOODS | 02 | 1,514,059 | 2.6497 | 1.007 | 0.7000 |
| FOODS | 03 | 1,666,971 | 2.6117 | 0.9920 | -0.8000 |
| FOODS | 04 | 1,632,137 | 2.5854 | 0.9820 | -1.8 |
| FOODS | 05 | 1,614,178 | 2.5614 | 0.9730 | -2.7 |
| FOODS | 06 | 1,304,279 | 2.7805 | 1.057 | 5.7 |
| FOODS | 07 | 1,347,673 | 2.7943 | 1.062 | 6.2 |
| FOODS | 08 | 1,357,129 | 2.8215 | 1.072 | 7.2 |
| FOODS | 09 | 1,360,554 | 2.6877 | 1.021 | 2.1 |
| FOODS | 10 | 1,416,676 | 2.5909 | 0.9840 | -1.6 |
| FOODS | 11 | 1,377,283 | 2.4763 | 0.9410 | -5.9 |
| FOODS | 12 | 1,408,004 | 2.5069 | 0.9530 | -4.7 |
| HOBBIES | 01 | 597,118 | 0.8303 | 0.9880 | -1.2 |
| HOBBIES | 02 | 629,610 | 0.8881 | 1.057 | 5.7 |
| HOBBIES | 03 | 701,472 | 0.8538 | 1.016 | 1.6 |
| HOBBIES | 04 | 696,086 | 0.8628 | 1.026 | 2.6 |
| HOBBIES | 05 | 694,893 | 0.8363 | 0.9950 | -0.5000 |
| HOBBIES | 06 | 567,922 | 0.8746 | 1.04 | 4 |
| HOBBIES | 07 | 593,154 | 0.8328 | 0.9910 | -0.9000 |
| HOBBIES | 08 | 599,144 | 0.8062 | 0.9590 | -4.1 |
| HOBBIES | 09 | 580,788 | 0.7817 | 0.9300 | -7 |
| HOBBIES | 10 | 602,349 | 0.8428 | 1.003 | 0.3000 |
| HOBBIES | 11 | 573,978 | 0.8274 | 0.9840 | -1.6 |
| HOBBIES | 12 | 581,969 | 0.8502 | 1.011 | 1.1 |
| HOUSEHOLD | 01 | 1,099,444 | 1.0333 | 0.9390 | -6.1 |
| HOUSEHOLD | 02 | 1,154,561 | 1.1421 | 1.038 | 3.8 |
| HOUSEHOLD | 03 | 1,295,213 | 1.1076 | 1.007 | 0.7000 |
| HOUSEHOLD | 04 | 1,278,357 | 1.0775 | 0.9790 | -2.1 |
| HOUSEHOLD | 05 | 1,252,008 | 1.0522 | 0.9560 | -4.4 |
| HOUSEHOLD | 06 | 1,008,784 | 1.1171 | 1.015 | 1.5 |
| HOUSEHOLD | 07 | 1,056,402 | 1.121 | 1.019 | 1.9 |
| HOUSEHOLD | 08 | 1,069,064 | 1.1782 | 1.071 | 7.1 |
| HOUSEHOLD | 09 | 1,036,549 | 1.1684 | 1.062 | 6.2 |
| HOUSEHOLD | 10 | 1,070,993 | 1.1228 | 1.02 | 2 |
| HOUSEHOLD | 11 | 1,038,575 | 1.067 | 0.9700 | -3 |
| HOUSEHOLD | 12 | 1,069,753 | 1.0166 | 0.9240 | -7.6 |

**Named events**

| cat_id | segment | n_obs | mean_units | lift | pct_vs_baseline |
|---|---|---|---|---|---|
| FOODS | LaborDay | 44,543 | 3.3419 | 1.31 | 31 |
| FOODS | SuperBowl | 53,621 | 3.2982 | 1.293 | 29.3 |
| FOODS | OrthodoxEaster | 44,987 | 3.1458 | 1.233 | 23.3 |
| FOODS | Father's day | 33,225 | 3.0503 | 1.196 | 19.6 |
| FOODS | Easter | 48,572 | 2.8837 | 1.13 | 13 |
| FOODS | IndependenceDay | 43,442 | 2.8186 | 1.105 | 10.5 |
| FOODS | Ramadan starts | 43,555 | 2.766 | 1.084 | 8.4 |
| FOODS | Mother's day | 55,746 | 2.742 | 1.075 | 7.5 |
| FOODS | Eid al-Fitr | 44,074 | 2.74 | 1.074 | 7.4 |
| FOODS | Purim End | 54,098 | 2.722 | 1.067 | 6.7 |
| FOODS | NBAFinalsEnd | 43,552 | 2.7084 | 1.062 | 6.2 |
| FOODS | ColumbusDay | 46,001 | 2.7006 | 1.059 | 5.9 |
| HOBBIES | Pesach End | 23,322 | 0.9639 | 1.243 | 24.3 |
| HOBBIES | OrthodoxEaster | 19,213 | 0.9595 | 1.238 | 23.8 |
| HOBBIES | Father's day | 14,490 | 0.9379 | 1.21 | 21 |
| HOBBIES | SuperBowl | 22,055 | 0.9109 | 1.175 | 17.5 |
| HOBBIES | Purim End | 22,669 | 0.8943 | 1.154 | 15.4 |
| HOBBIES | no_event | 6,817,852 | 0.8468 | 1.092 | 9.2 |
| HOBBIES | Ramadan starts | 19,193 | 0.8372 | 1.08 | 8 |
| HOBBIES | StPatricksDay | 22,656 | 0.8371 | 1.08 | 8 |
| HOBBIES | ValentinesDay | 22,269 | 0.8327 | 1.074 | 7.4 |
| HOBBIES | VeteransDay | 19,159 | 0.8293 | 1.07 | 7 |
| HOBBIES | Chanukah End | 18,776 | 0.8239 | 1.063 | 6.3 |
| HOBBIES | LentStart | 22,261 | 0.8203 | 1.058 | 5.8 |
| HOUSEHOLD | LaborDay | 34,349 | 1.3695 | 1.33 | 33 |
| HOUSEHOLD | OrthodoxEaster | 35,423 | 1.3039 | 1.266 | 26.6 |
| HOUSEHOLD | Father's day | 25,687 | 1.2189 | 1.184 | 18.4 |
| HOUSEHOLD | SuperBowl | 40,160 | 1.2102 | 1.175 | 17.5 |
| HOUSEHOLD | PresidentsDay | 41,088 | 1.1987 | 1.164 | 16.4 |
| HOUSEHOLD | Pesach End | 42,728 | 1.1924 | 1.158 | 15.8 |
| HOUSEHOLD | Eid al-Fitr | 34,578 | 1.167 | 1.133 | 13.3 |
| HOUSEHOLD | Purim End | 41,763 | 1.1545 | 1.121 | 12.1 |
| HOUSEHOLD | Easter | 37,836 | 1.116 | 1.084 | 8.4 |
| HOUSEHOLD | MemorialDay | 33,494 | 1.1144 | 1.082 | 8.2 |
| HOUSEHOLD | no_event | 12,337,706 | 1.1055 | 1.074 | 7.4 |
| HOUSEHOLD | EidAlAdha | 34,652 | 1.1014 | 1.07 | 7 |


### Price elasticity (naive, NOT causal)

| cat_id | dept_id | n_obs | elasticity | r2 | correlation | reading |
|---|---|---|---|---|---|---|
| FOODS | ALL | 10,662,222 | -0.4320 | 0.0889 | -0.2981 | inelastic (expected for grocery staples) |
| FOODS | FOODS_1 | 1,551,842 | -0.3693 | 0.0784 | -0.2800 | inelastic (expected for grocery staples) |
| FOODS | FOODS_2 | 2,489,171 | -0.3242 | 0.0561 | -0.2369 | inelastic (expected for grocery staples) |
| FOODS | FOODS_3 | 6,621,209 | -0.4574 | 0.0889 | -0.2982 | inelastic (expected for grocery staples) |
| HOBBIES | ALL | 2,505,971 | -0.2623 | 0.1631 | -0.4039 | inelastic (expected for grocery staples) |
| HOBBIES | HOBBIES_1 | 2,169,282 | -0.3051 | 0.2119 | -0.4603 | inelastic (expected for grocery staples) |
| HOBBIES | HOBBIES_2 | 336,689 | -0.1279 | 0.0487 | -0.2206 | inelastic (expected for grocery staples) |
| HOUSEHOLD | ALL | 5,771,078 | -0.3624 | 0.1187 | -0.3445 | inelastic (expected for grocery staples) |
| HOUSEHOLD | HOUSEHOLD_1 | 3,833,374 | -0.3755 | 0.1215 | -0.3485 | inelastic (expected for grocery staples) |
| HOUSEHOLD | HOUSEHOLD_2 | 1,937,704 | -0.1464 | 0.0311 | -0.1763 | inelastic (expected for grocery staples) |


### Promotion lift vs matched controls

| cat_id | price_state | discount_band | matched_cells | promo_days | control_days | lift | pct_uplift | median_cell_lift | avg_discount |
|---|---|---|---|---|---|---|---|---|---|
| FOODS | markdown | a. 5-10% | 8,092 | 50,385 | 51,256 | 1.073 | 7.3 | 1.02 | 0.0750 |
| FOODS | markdown | b. 10-20% | 8,456 | 53,335 | 53,176 | 1.119 | 11.9 | 1.055 | 0.1410 |
| FOODS | markdown | c. 20-35% | 1,325 | 8,464 | 8,302 | 1.656 | 65.6 | 1.198 | 0.2440 |
| FOODS | promo | a. 5-10% | 505 | 2,808 | 3,065 | 0.7270 | -27.3 | 0.8330 | 0.0780 |
| FOODS | promo | b. 10-20% | 1,084 | 5,985 | 7,163 | 0.7030 | -29.7 | 0.6670 | 0.1450 |
| FOODS | promo | c. 20-35% | 1,084 | 6,376 | 6,527 | 2.033 | 103.3 | 1.8 | 0.2730 |
| HOBBIES | markdown | a. 5-10% | 2,151 | 13,340 | 13,398 | 1.024 | 2.4 | 0.9380 | 0.0750 |
| HOBBIES | markdown | b. 10-20% | 1,135 | 7,187 | 7,242 | 0.9320 | -6.8 | 0.7780 | 0.1320 |
| HOBBIES | markdown | c. 20-35% | 190 | 1,238 | 1,217 | 0.7970 | -20.3 | 0.6370 | 0.2630 |
| HOUSEHOLD | markdown | a. 5-10% | 2,668 | 16,953 | 16,141 | 0.9800 | -2 | 0.9350 | 0.0680 |
| HOUSEHOLD | markdown | b. 10-20% | 1,094 | 7,060 | 6,494 | 0.8490 | -15.1 | 0.8000 | 0.1270 |
| HOUSEHOLD | markdown | c. 20-35% | 174 | 1,272 | 939 | 1.051 | 5.1 | 1.035 | 0.2710 |
| HOUSEHOLD | promo | a. 5-10% | 333 | 1,930 | 2,056 | 1.017 | 1.7 | 1.067 | 0.0690 |
| HOUSEHOLD | promo | b. 10-20% | 223 | 1,191 | 1,638 | 0.4410 | -55.9 | 0.6250 | 0.1530 |


### Backtest summary (CV folds, ranked)

| rank | model | wrmsse | wrmsse_sd | mase | folds |
|---|---|---|---|---|---|
| 1 | global_lgbm | 0.8559 | 0.0168 | 1.215 | 5 |
| 2 | tsb | 0.8893 | 0.0151 | 1.2171 | 5 |
| 3 | croston | 0.8910 | 0.0155 | 1.1962 | 5 |
| 4 | moving_average | 0.8950 | 0.0126 | 1.0729 | 5 |
| 5 | seasonal_croston | 0.9162 | 0.0221 | 1.2134 | 5 |
| 6 | seasonal_naive | 1.1563 | 0.0353 | 1.2075 | 5 |
| 7 | naive | 1.2359 | 0.0458 | 1.7499 | 5 |
| 8 | zero | 1.3963 | 0.0400 | 0.9859 | 5 |


### Backtest, per fold

| model | fold | wrmsse | rmsse_mean | mase_mean | n_series_scored |
|---|---|---|---|---|---|
| croston | cv1 | 0.9116 | 0.7509 | 1.2312 | 30,463 |
| croston | cv2 | 0.8811 | 0.7403 | 1.1904 | 30,474 |
| croston | cv3 | 0.9029 | 0.7672 | 1.2033 | 30,480 |
| croston | cv4 | 0.8842 | 0.7652 | 1.1891 | 30,490 |
| croston | cv5 | 0.8750 | 0.7651 | 1.167 | 30,490 |
| croston | test | 0.8639 | 0.7719 | 1.1331 | 30,490 |
| global_lgbm | cv1 | 0.8566 | 0.7375 | 1.2056 | 30,463 |
| global_lgbm | cv2 | 0.8395 | 0.7356 | 1.1949 | 30,474 |
| global_lgbm | cv3 | 0.8798 | 0.7759 | 1.2598 | 30,480 |
| global_lgbm | cv4 | 0.8634 | 0.7700 | 1.2282 | 30,490 |
| global_lgbm | cv5 | 0.8404 | 0.7627 | 1.1867 | 30,490 |
| global_lgbm | test | 0.8446 | 0.7675 | 1.1433 | 30,490 |
| moving_average | cv1 | 0.9078 | 0.7017 | 1.03 | 30,463 |
| moving_average | cv2 | 0.8891 | 0.7094 | 1.0264 | 30,474 |
| moving_average | cv3 | 0.9060 | 0.7440 | 1.072 | 30,480 |
| moving_average | cv4 | 0.8946 | 0.7580 | 1.1198 | 30,490 |
| moving_average | cv5 | 0.8773 | 0.7522 | 1.1162 | 30,490 |
| moving_average | test | 0.8720 | 0.7748 | 1.1095 | 30,490 |
| naive | cv1 | 1.3014 | 1.1222 | 1.8805 | 30,463 |
| naive | cv2 | 1.1978 | 1.0832 | 1.7714 | 30,474 |
| naive | cv3 | 1.259 | 1.1203 | 1.8034 | 30,480 |
| naive | cv4 | 1.2312 | 1.1084 | 1.7268 | 30,490 |
| naive | cv5 | 1.19 | 1.0575 | 1.5674 | 30,490 |
| naive | test | 1.1766 | 1.064 | 1.5196 | 30,490 |
| seasonal_croston | cv1 | 0.9476 | 0.8088 | 1.2495 | 30,463 |
| seasonal_croston | cv2 | 0.9084 | 0.7923 | 1.208 | 30,474 |
| seasonal_croston | cv3 | 0.9279 | 0.8174 | 1.2199 | 30,480 |
| seasonal_croston | cv4 | 0.9073 | 0.8164 | 1.206 | 30,490 |
| seasonal_croston | cv5 | 0.8900 | 0.8177 | 1.1834 | 30,490 |
| seasonal_croston | test | 0.8824 | 0.8185 | 1.151 | 30,490 |
| seasonal_naive | cv1 | 1.2099 | 0.9409 | 1.1915 | 30,463 |
| seasonal_naive | cv2 | 1.1725 | 0.9215 | 1.1541 | 30,474 |
| seasonal_naive | cv3 | 1.1455 | 0.9513 | 1.1938 | 30,480 |
| seasonal_naive | cv4 | 1.1254 | 0.9768 | 1.2421 | 30,490 |
| seasonal_naive | cv5 | 1.1282 | 0.9819 | 1.2562 | 30,490 |
| seasonal_naive | test | 1.1017 | 1.0017 | 1.2434 | 30,490 |
| tsb | cv1 | 0.9083 | 0.7539 | 1.2735 | 30,463 |
| tsb | cv2 | 0.8813 | 0.7446 | 1.2195 | 30,474 |
| tsb | cv3 | 0.9025 | 0.7704 | 1.2222 | 30,480 |
| tsb | cv4 | 0.8810 | 0.7680 | 1.2066 | 30,490 |
| tsb | cv5 | 0.8735 | 0.7656 | 1.1635 | 30,490 |
| tsb | test | 0.8626 | 0.7721 | 1.13 | 30,490 |
| zero | cv1 | 1.348 | 0.8305 | 0.8992 | 30,463 |
| zero | cv2 | 1.3731 | 0.8485 | 0.9248 | 30,474 |
| zero | cv3 | 1.4551 | 0.9103 | 1.0185 | 30,480 |
| zero | cv4 | 1.403 | 0.9167 | 1.034 | 30,490 |
| zero | cv5 | 1.4021 | 0.9310 | 1.0531 | 30,490 |
| zero | test | 1.4132 | 0.9685 | 1.0758 | 30,490 |


### Error by Syntetos-Boylan class

| sb_class | model | rmsse_mean | wrmsse | n_series |
|---|---|---|---|---|
| erratic | moving_average | 0.7379 | 0.9089 | 855 |
| erratic | global_lgbm | 0.7380 | 0.8768 | 855 |
| erratic | tsb | 0.7535 | 0.9085 | 855 |
| erratic | croston | 0.7605 | 0.9102 | 855 |
| erratic | seasonal_croston | 0.7855 | 0.9264 | 855 |
| erratic | seasonal_naive | 0.9293 | 1.1377 | 855 |
| erratic | naive | 0.9696 | 1.1805 | 855 |
| erratic | zero | 1.1197 | 1.6232 | 855 |
| intermittent | moving_average | 0.7337 | 0.8765 | 22,121 |
| intermittent | croston | 0.7577 | 0.8721 | 22,121 |
| intermittent | global_lgbm | 0.7587 | 0.8494 | 22,121 |
| intermittent | tsb | 0.7623 | 0.8710 | 22,121 |
| intermittent | seasonal_croston | 0.8152 | 0.9069 | 22,121 |
| intermittent | zero | 0.8438 | 1.2291 | 22,121 |
| intermittent | seasonal_naive | 0.9636 | 1.1627 | 22,121 |
| intermittent | naive | 1.1263 | 1.2358 | 22,121 |
| lumpy | moving_average | 0.7143 | 1.0119 | 5,610 |
| lumpy | tsb | 0.7462 | 0.9986 | 5,610 |
| lumpy | croston | 0.7473 | 1.0018 | 5,610 |
| lumpy | global_lgbm | 0.7500 | 0.9577 | 5,610 |
| lumpy | seasonal_croston | 0.7996 | 1.0415 | 5,610 |
| lumpy | zero | 0.8569 | 1.4103 | 5,610 |
| lumpy | seasonal_naive | 0.9115 | 1.2888 | 5,610 |
| lumpy | naive | 1.0188 | 1.3354 | 5,610 |
| smooth | global_lgbm | 0.7564 | 0.7967 | 1,877 |
| smooth | moving_average | 0.7790 | 0.8558 | 1,877 |
| smooth | tsb | 0.7852 | 0.8536 | 1,877 |
| smooth | croston | 0.7873 | 0.8554 | 1,877 |
| smooth | seasonal_croston | 0.7996 | 0.8501 | 1,877 |
| smooth | seasonal_naive | 0.9867 | 1.055 | 1,877 |
| smooth | naive | 1.0649 | 1.1839 | 1,877 |
| smooth | zero | 1.3867 | 1.7316 | 1,877 |


### Error by ABC class

| abc_class | model | rmsse_mean | wrmsse | n_series |
|---|---|---|---|---|
| A | moving_average | 0.7485 | 0.9077 | 11,221 |
| A | global_lgbm | 0.7554 | 0.8629 | 11,221 |
| A | tsb | 0.7758 | 0.9031 | 11,221 |
| A | croston | 0.7800 | 0.9058 | 11,221 |
| A | seasonal_croston | 0.8159 | 0.9234 | 11,221 |
| A | seasonal_naive | 0.9616 | 1.1545 | 11,221 |
| A | zero | 1.0465 | 1.5177 | 11,221 |
| A | naive | 1.0535 | 1.2464 | 11,221 |
| B | moving_average | 0.7166 | 0.8462 | 9,426 |
| B | global_lgbm | 0.7379 | 0.8246 | 9,426 |
| B | croston | 0.7382 | 0.8392 | 9,426 |
| B | tsb | 0.7394 | 0.8392 | 9,426 |
| B | seasonal_croston | 0.7917 | 0.8841 | 9,426 |
| B | zero | 0.8305 | 1.0425 | 9,426 |
| B | seasonal_naive | 0.9410 | 1.1437 | 9,426 |
| B | naive | 1.0631 | 1.1864 | 9,426 |
| C | moving_average | 0.7312 | 0.8773 | 9,816 |
| C | croston | 0.7510 | 0.8592 | 9,816 |
| C | zero | 0.7603 | 0.9747 | 9,816 |
| C | tsb | 0.7633 | 0.8640 | 9,816 |
| C | global_lgbm | 0.7751 | 0.8563 | 9,816 |
| C | seasonal_croston | 0.8224 | 0.9165 | 9,816 |
| C | seasonal_naive | 0.9592 | 1.203 | 9,816 |
| C | naive | 1.1834 | 1.2418 | 9,816 |


### Quantile calibration

| fold | method | quantile | nominal_coverage | empirical_coverage | gap |
|---|---|---|---|---|---|
| cv4 | lgbm_quantile | 0.5000 | 0.5000 | 0.6651 | 0.1651 |
| cv4 | lgbm_quantile | 0.9000 | 0.9000 | 0.9138 | 0.0138 |
| cv4 | lgbm_quantile | 0.9500 | 0.9500 | 0.9582 | 0.0082 |
| cv4 | lgbm_quantile | 0.9900 | 0.9900 | 0.9902 | 0.0002 |
| cv5 | lgbm_quantile | 0.5000 | 0.5000 | 0.6598 | 0.1598 |
| cv5 | lgbm_quantile | 0.9000 | 0.9000 | 0.9130 | 0.0130 |
| cv5 | lgbm_quantile | 0.9500 | 0.9500 | 0.9577 | 0.0077 |
| cv5 | lgbm_quantile | 0.9900 | 0.9900 | 0.9903 | 0.0003 |
| test | lgbm_quantile | 0.5000 | 0.5000 | 0.6384 | 0.1384 |
| test | lgbm_quantile | 0.9000 | 0.9000 | 0.9007 | 0.0007 |
| test | lgbm_quantile | 0.9500 | 0.9500 | 0.9505 | 0.0005 |
| test | lgbm_quantile | 0.9900 | 0.9900 | 0.9885 | -0.0015 |
| test | conformal | 0.5000 | 0.5000 | 0.6296 | 0.1296 |
| test | conformal | 0.9000 | 0.9000 | 0.8887 | -0.0113 |
| test | conformal | 0.9500 | 0.9500 | 0.9434 | -0.0066 |
| test | conformal | 0.9900 | 0.9900 | 0.9882 | -0.0018 |


### Calibration by demand class

| fold | segment | quantile | empirical_coverage | gap | n_series |
|---|---|---|---|---|---|
| cv4 | erratic | 0.5000 | 0.5246 | 0.0246 | 855 |
| cv4 | erratic | 0.9000 | 0.9112 | 0.0112 | 855 |
| cv4 | erratic | 0.9500 | 0.9566 | 0.0066 | 855 |
| cv4 | erratic | 0.9900 | 0.9870 | -0.0030 | 855 |
| cv4 | intermittent | 0.5000 | 0.6955 | 0.1955 | 22,143 |
| cv4 | intermittent | 0.9000 | 0.9148 | 0.0148 | 22,143 |
| cv4 | intermittent | 0.9500 | 0.9591 | 0.0091 | 22,143 |
| cv4 | intermittent | 0.9900 | 0.9905 | 0.0005 | 22,143 |
| cv4 | lumpy | 0.5000 | 0.6214 | 0.1214 | 5,612 |
| cv4 | lumpy | 0.9000 | 0.9160 | 0.0160 | 5,612 |
| cv4 | lumpy | 0.9500 | 0.9582 | 0.0082 | 5,612 |
| cv4 | lumpy | 0.9900 | 0.9905 | 0.0005 | 5,612 |
| cv4 | smooth | 0.5000 | 0.5010 | 0.0010 | 1,880 |
| cv4 | smooth | 0.9000 | 0.8968 | -0.0032 | 1,880 |
| cv4 | smooth | 0.9500 | 0.9481 | -0.0019 | 1,880 |
| cv4 | smooth | 0.9900 | 0.9870 | -0.0030 | 1,880 |
| cv5 | erratic | 0.5000 | 0.5631 | 0.0631 | 855 |
| cv5 | erratic | 0.9000 | 0.9188 | 0.0188 | 855 |
| cv5 | erratic | 0.9500 | 0.9603 | 0.0103 | 855 |
| cv5 | erratic | 0.9900 | 0.9877 | -0.0023 | 855 |
| cv5 | intermittent | 0.5000 | 0.6876 | 0.1876 | 22,143 |
| cv5 | intermittent | 0.9000 | 0.9124 | 0.0124 | 22,143 |
| cv5 | intermittent | 0.9500 | 0.9578 | 0.0078 | 22,143 |
| cv5 | intermittent | 0.9900 | 0.9906 | 0.0006 | 22,143 |
| cv5 | lumpy | 0.5000 | 0.6102 | 0.1102 | 5,612 |
| cv5 | lumpy | 0.9000 | 0.9161 | 0.0161 | 5,612 |
| cv5 | lumpy | 0.9500 | 0.9582 | 0.0082 | 5,612 |
| cv5 | lumpy | 0.9900 | 0.9909 | 0.0009 | 5,612 |
| cv5 | smooth | 0.5000 | 0.5247 | 0.0247 | 1,880 |
| cv5 | smooth | 0.9000 | 0.9091 | 0.0091 | 1,880 |
| cv5 | smooth | 0.9500 | 0.9534 | 0.0034 | 1,880 |
| cv5 | smooth | 0.9900 | 0.9862 | -0.0038 | 1,880 |
| test | erratic | 0.5000 | 0.5158 | 0.0158 | 855 |
| test | erratic | 0.9000 | 0.8990 | -0.0010 | 855 |
| test | erratic | 0.9500 | 0.9454 | -0.0046 | 855 |
| test | erratic | 0.9900 | 0.9836 | -0.0064 | 855 |
| test | intermittent | 0.5000 | 0.6686 | 0.1686 | 22,143 |
| test | intermittent | 0.9000 | 0.8998 | -0.0002 | 22,143 |
| test | intermittent | 0.9500 | 0.9507 | 0.0007 | 22,143 |
| test | intermittent | 0.9900 | 0.9888 | -0.0012 | 22,143 |
| test | lumpy | 0.5000 | 0.5800 | 0.0800 | 5,612 |
| test | lumpy | 0.9000 | 0.9040 | 0.0040 | 5,612 |
| test | lumpy | 0.9500 | 0.9510 | 0.0010 | 5,612 |
| test | lumpy | 0.9900 | 0.9881 | -0.0019 | 5,612 |
| test | smooth | 0.5000 | 0.5131 | 0.0131 | 1,880 |
| test | smooth | 0.9000 | 0.9024 | 0.0024 | 1,880 |
| test | smooth | 0.9500 | 0.9493 | -0.0007 | 1,880 |
| test | smooth | 0.9900 | 0.9875 | -0.0025 | 1,880 |


### Pinball loss

| fold | method | quantile | pinball_loss |
|---|---|---|---|
| cv4 | lgbm_quantile | 0.5000 | 1.2974 |
| cv4 | lgbm_quantile | 0.9000 | 0.6807 |
| cv4 | lgbm_quantile | 0.9500 | 0.4263 |
| cv4 | lgbm_quantile | 0.9900 | 0.1388 |
| cv5 | lgbm_quantile | 0.5000 | 1.1746 |
| cv5 | lgbm_quantile | 0.9000 | 0.6335 |
| cv5 | lgbm_quantile | 0.9500 | 0.4015 |
| cv5 | lgbm_quantile | 0.9900 | 0.1265 |
| test | lgbm_quantile | 0.5000 | 1.1861 |
| test | lgbm_quantile | 0.9000 | 0.6322 |
| test | lgbm_quantile | 0.9500 | 0.3964 |
| test | lgbm_quantile | 0.9900 | 0.1244 |
| test | conformal | 0.5000 | 0.5036 |
| test | conformal | 0.9000 | 0.3630 |
| test | conformal | 0.9500 | 0.2549 |
| test | conformal | 0.9900 | 0.0999 |


### Feature importance (mean gain share across folds)

| feature | gain_share |
|---|---|
| roll_mean_112 | 0.7273 |
| roll_mean_56 | 0.1525 |
| dow_mean_8 | 0.0223 |
| roll_mean_28 | 0.0215 |
| roll_mean_7 | 0.0084 |
| lag_364 | 0.0082 |
| wday | 0.0079 |
| month | 0.0057 |
| nonzero_frac_112 | 0.0045 |
| store_code | 0.0045 |
| price_rel_28 | 0.0043 |
| price | 0.0041 |
| dept_code | 0.0036 |
| sb_code | 0.0036 |
| day_of_month | 0.0026 |
| days_since_launch | 0.0025 |
| nonzero_frac_56 | 0.0021 |
| lag_28 | 0.0019 |
| dow_ratio_8 | 0.0017 |
| temp_climatology | 0.0014 |
| snap | 0.0013 |
| nonzero_frac_28 | 0.0012 |
| lag_35 | 0.0009 |
| temp_anomaly_lag28 | 0.0008 |
| lag_56 | 0.0008 |
| nonzero_frac_7 | 0.0007 |
| price_change_7 | 0.0007 |
| precip_climatology | 0.0006 |
| has_event | 0.0006 |
| lag_49 | 0.0005 |
| cat_code | 0.0005 |
| lag_42 | 0.0005 |
| state_code | 0.0003 |


### Counterfactual on the locked test window (d1914-1941)

| policy | fill_rate | cycle_service | avg_on_hand_units | total_demand_units | total_unmet_units | orders_placed | holding_cost | stockout_cost | ordering_cost | total_cost |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.9405 | 0.9842 | 18.4582 | 1,231,764 | 73,263 | 39,826 | 27,027 | 80,990 | 19,913 | 127,930 |
| forecast_only | 0.9403 | 0.9849 | 19.1641 | 1,231,764 | 73,588 | 38,781 | 28,140 | 81,723 | 19,390 | 129,254 |
| uniform_empirical | 0.9402 | 0.9802 | 18.7391 | 1,231,764 | 73,659 | 42,001 | 27,050 | 85,926 | 21,000 | 133,976 |
| cost_optimal_allocation | 0.9506 | 0.9835 | 20.1742 | 1,231,764 | 60,805 | 41,603 | 28,882 | 71,817 | 20,802 | 121,500 |


### Cost delta, 95% bootstrap CI (resampling series)

| comparison | cost_a | cost_b | delta | delta_pct | delta_pct_ci_lo | delta_pct_ci_hi | n_boot |
|---|---|---|---|---|---|---|---|
| baseline -> forecast_only | 127,930 | 129,254 | 1,324 | 1.0349 | -1.2556 | 3.5067 | 500 |
| baseline -> uniform_empirical | 127,930 | 133,976 | 6,047 | 4.7266 | 1.3537 | 8.313 | 500 |
| baseline -> cost_optimal_allocation | 127,930 | 121,500 | -6,430 | -5.0259 | -8.4583 | -1.8933 | 500 |


### Service-level frontier

| policy | target_service_level | fill_rate | cycle_service | avg_on_hand_units | avg_safety_stock | orders_placed | total_cost |
|---|---|---|---|---|---|---|---|
| sl_0.8 | 0.8000 | 0.8295 | 0.9349 | 10.8777 | 3.632 | 43,219 | 279,622 |
| sl_0.85 | 0.8500 | 0.8502 | 0.9440 | 11.8141 | 4.5571 | 42,927 | 251,938 |
| sl_0.9 | 0.9000 | 0.8769 | 0.9547 | 13.1791 | 5.9048 | 42,744 | 215,667 |
| sl_0.95 | 0.9500 | 0.9105 | 0.9686 | 15.558 | 8.1513 | 42,470 | 171,469 |
| sl_0.975 | 0.9750 | 0.9332 | 0.9774 | 17.8548 | 10.2766 | 41,963 | 141,869 |
| sl_0.99 | 0.9900 | 0.9516 | 0.9844 | 20.4292 | 12.9273 | 42,198 | 119,852 |
| sl_0.995 | 0.9950 | 0.9599 | 0.9876 | 21.859 | 14.4987 | 42,635 | 110,471 |
| sl_0.999 | 0.9990 | 0.9701 | 0.9911 | 23.8829 | 16.7206 | 43,158 | 99,618 |
| reconstructed_baseline | — | 0.9405 | 0.9842 | 18.4582 | 10.7919 | 39,826 | 127,930 |


### Cost sensitivity (full factorial)

All 27 cells, sorted. Range **-0.77% to -8.76%**, median **-5.16%**. All negative: **True**.

| gross_margin | holding_rate | stockout_penalty | baseline_cost | optimized_cost | delta | delta_pct | baseline_fill | optimized_fill |
|---|---|---|---|---|---|---|---|---|
| 0.3000 | 0.1500 | 3 | 229,424 | 209,336 | -20,088 | -8.7558 | 0.9405 | 0.9506 |
| 0.2500 | 0.1500 | 3 | 198,109 | 181,764 | -16,344 | -8.2502 | 0.9405 | 0.9506 |
| 0.3000 | 0.2500 | 3 | 239,514 | 220,118 | -19,395 | -8.0978 | 0.9405 | 0.9506 |
| 0.2000 | 0.1500 | 3 | 166,794 | 154,193 | -12,601 | -7.5549 | 0.9405 | 0.9506 |
| 0.2500 | 0.2500 | 3 | 208,919 | 193,317 | -15,603 | -7.4682 | 0.9405 | 0.9506 |
| 0.3000 | 0.4000 | 3 | 254,649 | 236,292 | -18,357 | -7.2086 | 0.9405 | 0.9506 |
| 0.3000 | 0.1500 | 1.5 | 132,236 | 123,155 | -9,080 | -6.8668 | 0.9405 | 0.9506 |
| 0.2000 | 0.2500 | 3 | 178,325 | 166,516 | -11,810 | -6.6226 | 0.9405 | 0.9506 |
| 0.2500 | 0.4000 | 3 | 225,136 | 210,646 | -14,490 | -6.436 | 0.9405 | 0.9506 |
| 0.2500 | 0.1500 | 1.5 | 117,119 | 109,947 | -7,172 | -6.1233 | 0.9405 | 0.9506 |
| 0.3000 | 0.2500 | 1.5 | 142,326 | 133,938 | -8,388 | -5.8934 | 0.9405 | 0.9506 |
| 0.2000 | 0.4000 | 3 | 195,623 | 185,000 | -10,623 | -5.4302 | 0.9405 | 0.9506 |
| 0.3000 | 0.1500 | 1 | 99,840 | 94,429 | -5,411 | -5.4198 | 0.9405 | 0.9506 |
| 0.2000 | 0.1500 | 1.5 | 102,002 | 96,739 | -5,263 | -5.1595 | 0.9405 | 0.9506 |
| 0.2500 | 0.2500 | 1.5 | 127,930 | 121,500 | -6,430 | -5.0259 | 0.9405 | 0.9506 |
| 0.3000 | 0.4000 | 1.5 | 157,461 | 150,112 | -7,349 | -4.6673 | 0.9405 | 0.9506 |
| 0.2500 | 0.1500 | 1 | 90,122 | 86,008 | -4,114 | -4.5648 | 0.9405 | 0.9506 |
| 0.3000 | 0.2500 | 1 | 109,930 | 105,211 | -4,719 | -4.2924 | 0.9405 | 0.9506 |
| 0.2000 | 0.2500 | 1.5 | 113,533 | 109,062 | -4,471 | -3.9384 | 0.9405 | 0.9506 |
| 0.2500 | 0.4000 | 1.5 | 144,146 | 138,829 | -5,317 | -3.6884 | 0.9405 | 0.9506 |
| 0.2000 | 0.1500 | 1 | 80,405 | 77,588 | -2,817 | -3.5031 | 0.9405 | 0.9506 |
| 0.2500 | 0.2500 | 1 | 100,933 | 97,561 | -3,372 | -3.3408 | 0.9405 | 0.9506 |
| 0.3000 | 0.4000 | 1 | 125,065 | 121,385 | -3,680 | -2.9425 | 0.9405 | 0.9506 |
| 0.2000 | 0.4000 | 1.5 | 130,831 | 127,546 | -3,284 | -2.5104 | 0.9405 | 0.9506 |
| 0.2000 | 0.2500 | 1 | 91,936 | 89,911 | -2,025 | -2.2029 | 0.9405 | 0.9506 |
| 0.2500 | 0.4000 | 1 | 117,149 | 114,890 | -2,259 | -1.9284 | 0.9405 | 0.9506 |
| 0.2000 | 0.4000 | 1 | 109,233 | 108,395 | -838.2031 | -0.7674 | 0.9405 | 0.9506 |


### Shadow prices

| constraint | used | limit | utilisation | shadow_price | binding |
|---|---|---|---|---|---|
| capacity_CA_1 | 62,227 | 62,228 | 1.0000 | 0.0480 | yes |
| capacity_CA_2 | 59,153 | 59,168 | 0.9998 | 0.0436 | yes |
| capacity_CA_3 | 89,950 | 89,951 | 1.0000 | 0.0514 | yes |
| capacity_CA_4 | 38,947 | 38,953 | 0.9998 | 0.0422 | yes |
| capacity_TX_1 | 52,633 | 52,639 | 0.9999 | 0.0392 | yes |
| capacity_TX_2 | 58,656 | 58,678 | 0.9996 | 0.0449 | yes |
| capacity_TX_3 | 58,454 | 58,465 | 0.9998 | 0.0426 | yes |
| capacity_WI_1 | 51,071 | 51,079 | 0.9998 | 0.0424 | yes |
| capacity_WI_2 | 79,109 | 79,117 | 0.9999 | 0.0413 | yes |
| capacity_WI_3 | 56,049 | 56,080 | 0.9995 | 0.0417 | yes |
| budget | 1,561,766 | 1,561,935 | 0.9999 | 0.0032 | yes |


### Replenishment policy

Per-series policy over 30,490 series, summarised by class.

| abc_class | sb_class | series | mean_s | mean_S | mean_safety_stock | mean_chosen_SL |
|---|---|---|---|---|---|---|
| A | erratic | 788 | 77.7525 | 144.1383 | 49.584 | 0.9900 |
| A | intermittent | 6,475 | 18.5151 | 33.7225 | 11.3021 | 0.9857 |
| A | lumpy | 2,210 | 38.7036 | 70.0457 | 25.2573 | 0.9892 |
| A | smooth | 1,748 | 69.5212 | 131.7019 | 38.2029 | 0.9899 |
| B | erratic | 61 | 54.8689 | 98.5902 | 35.0331 | 0.9898 |
| B | intermittent | 7,304 | 10.2611 | 17.8809 | 6.7499 | 0.9810 |
| B | lumpy | 1,949 | 24.8584 | 42.7681 | 16.8682 | 0.9881 |
| B | smooth | 112 | 32.7232 | 60.1875 | 18.8839 | 0.9900 |
| C | erratic | 6 | 42.3333 | 73.8333 | 25.4139 | 0.9900 |
| C | intermittent | 8,364 | 7.0441 | 11.8749 | 5.0546 | 0.9752 |
| C | lumpy | 1,453 | 18.9966 | 31.1211 | 14.0163 | 0.9861 |
| C | smooth | 20 | 29.6 | 53.55 | 17.6629 | 0.9900 |

**Service levels actually chosen by the allocator**

| chosen_service_level | series |
|---|---|
| 0.8000 | 88 |
| 0.8500 | 86 |
| 0.9000 | 284 |
| 0.9500 | 1,926 |
| 0.9750 | 6,471 |
| 0.9900 | 21,600 |
| 0.9950 | 35 |


### Hierarchy levels

| level | n_nodes |
|---|---|
| total | 1 |
| state | 3 |
| store | 10 |
| category | 3 |
| department | 7 |
| state_category | 9 |
| state_department | 21 |
| store_category | 30 |
| store_department | 70 |
| item | 3,049 |
| item_state | 9,147 |
| item_store | 30,490 |


### Reconciliation coherence (adding-up violation)

| method | max_abs_violation | relative_violation |
|---|---|---|
| base | 7,934 | 1.2556 |
| bottom_up | 0.0000 | 0.0000 |
| ols | 0.0000 | 0.0000 |
| wls_struct | 0.0000 | 0.0000 |
| wls_var | 0.0000 | 0.0000 |


### Reconciliation accuracy — RMSSE by level

| level | n_nodes | base | bottom_up | ols | wls_struct | wls_var |
|---|---|---|---|---|---|---|
| total | 1 | 0.6692 | 0.2983 | 0.6263 | 0.6178 | 0.6129 |
| state | 3 | 0.7002 | 0.3775 | 0.6728 | 0.6584 | 0.6289 |
| store | 10 | 0.6979 | 0.4584 | 0.6709 | 0.6731 | 0.6487 |
| category | 3 | 0.6295 | 0.3679 | 0.5855 | 0.5944 | 0.6286 |
| department | 7 | 0.8930 | 0.4802 | 0.7959 | 0.8503 | 0.8197 |
| state_category | 9 | 0.6636 | 0.4656 | 0.6348 | 0.6338 | 0.6498 |
| state_department | 21 | 0.8804 | 0.5833 | 0.8111 | 0.8309 | 0.8009 |
| store_category | 30 | 0.6765 | 0.5555 | 0.6870 | 0.6700 | 0.6747 |
| store_department | 70 | 0.8266 | 0.6549 | 0.8238 | 0.7985 | 0.7759 |
| item | 3,049 | 0.9076 | 0.8644 | 0.9097 | 0.8702 | 0.8575 |
| item_state | 9,147 | 0.8492 | 0.8367 | 0.8526 | 0.8324 | 0.8252 |
| item_store | 30,490 | 0.8241 | 0.8241 | 0.8297 | 0.8195 | 0.8176 |


<!-- METRICS:END -->
