# Build progress

Single source of truth for where this build stands. **Update this file at the
end of every working session**, before the context runs out.

Spec: `PROJECT-01-supply-chain-forecasting.md` (the original brief).

---

## Current state

**Week 1: COMPLETE.** Week 2: in progress.

| Week | Deliverable | Status |
|---|---|---|
| 1 | Ingestion + warehouse + SQL analytics | ✅ complete |
| 2 | Feature store + model ladder + backtest harness | 🔨 in progress |
| 3 | Optimizer + counterfactual | ⬜ not started |
| 4 | Dashboard + deploy + docs | ⬜ not started |

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
uv run pytest                # 26 tests
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
- `ingest/` — `m5_loader.py`, `weather_client.py`, `fred_client.py` (dormant)
- `sql/ddl/` → `sql/transform/` → `sql/analytics/` (pipeline order matters)
- `tests/` — 26 tests, all passing

---

## Decisions already made — do not silently revisit

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
