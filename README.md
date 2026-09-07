# Supply Chain Intelligence

> Demand forecasting and inventory optimization over retail SKU-store series.
> **Headline number: TBD** — filled in once the counterfactual simulation runs (week 3).

**Status:** week 1 in progress — ingestion and warehouse.

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

_Pending — week 3._

| | Service level | Total cost | Stockout days | Avg inventory |
|---|---|---|---|---|
| Reconstructed baseline | — | — | — | — |
| Optimized policy | — | — | — | — |

## Methods

See [docs/02-methodology.md](docs/02-methodology.md).

## Limitations

See [docs/03-assumptions-and-limitations.md](docs/03-assumptions-and-limitations.md).
Read this before the results table.

## How to run locally

```bash
uv sync --extra dev
cp .env.example .env      # then edit SCI_DATA_DIR and SCI_M5_SOURCE
uv run sci ingest m5
uv run sci warehouse build
```

Requires Python 3.11 or 3.12.
