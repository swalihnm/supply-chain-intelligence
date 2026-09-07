# Supply Chain Intelligence

> Demand forecasting and inventory optimization over retail SKU-store series.
> **Headline number: TBD** — filled in once the counterfactual simulation runs (week 3).

**Status:** week 1 complete — 46.9M-row panel in DuckDB, censoring handled,
analytics running. Weeks 2–4 (models, optimizer, serving) not started.

---

## The business problem

A retailer holds inventory at thousands of store-SKU combinations. Hold too
much and capital sits on shelves; hold too little and sales are lost. The
decision is *how much to stock at each store-SKU to hit a 95% service level at
the lowest total cost*.

This project answers that end to end: it forecasts demand as a **distribution**
(not a point estimate), converts those distributions into a replenishment
policy, and simulates that policy against reconstructed observed behaviour to
put a defended cost number on the difference.

## Architecture

```
sources/     M5 (Kaggle), Open-Meteo, FRED
   |
ingest/      idempotent, checksummed pulls -> parquet
   |
warehouse/   DuckDB star schema
   |
features/    point-in-time-correct feature store
   |
models/      baseline -> statistical -> global ML -> quantiles -> reconciliation
   |
optimize/    (s, S) policy under capacity / budget / MOQ constraints
   |
serve/       FastAPI + Streamlit
```

## Results

**The cost result is a week-3 deliverable and does not exist yet.**

| | Service level | Total cost | Stockout days | Avg inventory |
|---|---|---|---|---|
| Reconstructed baseline | — | — | — | — |
| Optimized policy | — | — | — | — |

What week 1 established, in full at [docs/04-results.md](docs/04-results.md):

- **46,881,677-row panel** built from 59.2M raw rows. The 20.8% dropped are
  SKU-store pairs the store did not carry yet — verified, not assumed:
  the price-NULL block is contiguous and leading for **100.00%** of all 30,490
  series, and no row anywhere has positive sales with a NULL price.
- **91% of series are intermittent or lumpy** (Syntetos-Boylan). ETS and ARIMA
  are the wrong family for nearly the whole catalogue.
- **45% of revenue sits in the AZ box** — high value *and* highly erratic, 96%
  of it intermittent. This is the direct argument for forecasting the
  distribution rather than the mean.
- **3.99% of panel days are likely stockouts**, detected with a per-series
  adaptive threshold. A first attempt using lifetime demand rates flagged 18.3%
  and was measuring dormancy; [the fix is written up in
  full](docs/02-methodology.md#p-is-local-not-lifetime).
- **SNAP days lift FOODS +7.7%** but HOBBIES only +1.3% — an end-to-end
  validation of the calendar → store → state join that nothing was tuned to
  produce.

## Methods

See [docs/02-methodology.md](docs/02-methodology.md).

## Limitations

See [docs/03-assumptions-and-limitations.md](docs/03-assumptions-and-limitations.md).
Read this before the results table.

## How to run locally

```bash
uv sync --extra dev
cp .env.example .env      # then edit SCI_DATA_DIR and SCI_M5_SOURCE
uv run sci ingest all     # M5 from local zip, weather from Open-Meteo
uv run sci warehouse build
uv run sci analyze        # lists the analytics reports
uv run sci analyze abc_xyz_segmentation --csv
```

Requires Python 3.11 or 3.12. M5 must be downloaded manually from
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
(accept the competition rules first); point `SCI_M5_SOURCE` at the zip.

A full rebuild from the raw zip takes about 4 minutes and produces a 163 MB
DuckDB file. Set `SCI_DATA_DIR` somewhere outside any synced folder.
