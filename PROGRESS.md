# Build progress

Single source of truth for where this build stands. **Update this file at the
end of every working session**, before the context runs out.

Spec: `PROJECT-01-supply-chain-forecasting.md` (the original brief).

---

## Current state

**All four weeks built and run end to end, plus hierarchical reconciliation.
123 tests pass. Only the container deploy is outstanding, and it is blocked
on a credential rather than on work.**

| Week | Deliverable | Status |
|---|---|---|
| 1 | Ingestion + warehouse + SQL analytics | ✅ complete |
| 2 | Feature store + model ladder + backtest harness | ✅ complete |
| 3 | Optimizer + counterfactual + shadow prices | ✅ complete |
| 4 | API + dashboard + docs | ✅ complete (container unverified) |
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

1. **Deploy for a live URL — BLOCKED on a credential, not on work.**
   Docker cannot be installed from this session:
   - Docker Desktop needs Windows admin; the session user is not an admin, so
     `winget install Docker.DockerDesktop` stops at a UAC prompt.
   - Docker Engine in WSL2 needs the Ubuntu `sudo` password.
   - Rootless Docker needs `uidmap` + `iptables`, neither installed, and
     installing them needs sudo.

   Everything else is ready: WSL2 (Ubuntu 24.04) is installed, virtualization is
   on, cgroup2 is present, subuid/subgid are configured, 954 GB free.

   **The user runs this once, in a WSL terminal:**
   ```bash
   sudo apt-get update && sudo apt-get install -y uidmap iptables ca-certificates curl
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER && sudo service docker start
   ```
   Then: `uv run python scripts/stage_deploy.py && docker compose up --build`.

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
uv run sci warehouse build   # ~4 min -> 163 MB DuckDB
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
  `build_serving_bundle.py`, `stage_deploy.py`
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

## Key numbers (for README / interview prep)

- 91% of series are intermittent or lumpy (Syntetos-Boylan)
- 45.1% of revenue is in the AZ box (high value, high variability), 96% of it intermittent
- Revenue is NOT 80/20: it takes 36.8% of series to reach 80% of revenue
- Stockout proxy: 3.99% of panel days; sensitivity 2.37%–10.86% across alpha 0.001–0.05
- SNAP lifts FOODS +7.7%, HOBBIES +1.3%, HOUSEHOLD +1.9%
- Naive elasticity: FOODS −0.43, HOUSEHOLD −0.36, HOBBIES −0.26 (all weak R², not causal)
- Weekend lift: +19% to +26% depending on category

---

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
- Not a Windows admin -> Docker Desktop's installer stops at UAC.
- WSL2 (Ubuntu 24.04) is installed and virtualization is on, but `sudo` needs a
  password that cannot be supplied non-interactively.
- Rootless Docker needs `uidmap` and `iptables`; neither is present and both
  need sudo to install.

Prerequisites that ARE in place: WSL2, cgroup2, subuid/subgid configured for the
user, 954 GB free. The one-time command the user must run is in
"What is genuinely left" above.
