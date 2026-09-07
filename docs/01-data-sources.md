# Data sources

Four independent sources, joined into one panel. Nothing here is a single flat
CSV download.

| Source | What it gives | Grain | Key required | Status |
|---|---|---|---|---|
| M5 / Walmart (Kaggle) | Transaction spine: units sold, prices, calendar, SNAP + event flags | item × store × day | Kaggle account (manual download) | **ingested** (59,181,090 rows) |
| Open-Meteo Archive | Daily temperature, precipitation, snowfall, wind | state × day | none | **ingested** (6,210 rows) |
| FRED | CPI, gas price, national + state unemployment | series × native freq | free API key | client written, dormant |
| Google Trends | Category search interest | category × state × week | none (rate-limited) | deferred — first in the spec's cut order |

## M5 — the transaction spine

Download from
<https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>. You must
accept the competition rules on that page before the download works.

Point `SCI_M5_SOURCE` in `.env` at either the downloaded `.zip` or a folder you
unpacked it into. The loader handles both.

Files used:

- `sales_train_evaluation.csv` — 30,490 rows × 1,941 day columns (`d_1`..`d_1941`)
- `calendar.csv` — 1,969 days, maps `d` → real date, carries SNAP and event flags
- `sell_prices.csv` — weekly price per item × store

`sales_train_validation.csv` is the same data truncated at `d_1913` and is
**not** used. Using it by accident would silently drop the last 28 days.

### The unpivot

The sales file arrives in wide form. `ingest/m5_loader.py` unpivots it to
`(item_id, store_id, d, units)` — about 59.2M rows. This runs through the Polars
streaming engine and sinks directly to parquet, so it never materialises in
memory.

## Open-Meteo — weather

Keyless. `ingest/weather_client.py` pulls 2010-12-01 → 2016-07-31, a margin
either side of the M5 span (2011-01-29 → 2016-06-19) so lag and rolling features
are not truncated at the boundaries.

Raw JSON responses are cached under `raw/weather/` so re-runs never re-hit the
API.

**See [03-assumptions-and-limitations.md](03-assumptions-and-limitations.md) for
the store-city approximation. It matters.**

## FRED — macro

Free key from <https://fred.stlouisfed.org/docs/api/api_key.html>. Put it in
`.env` as `FRED_API_KEY`. Without it the step is skipped and the rest of the
pipeline runs unaffected.

Series pulled: `CPIAUCSL`, `GASREGW`, `UNRATE`, plus `CAUR` / `TXUR` / `WIUR`
for state-level unemployment.

Series are stored at their **native frequency**. Resampling to daily is a
feature decision — forward-fill only, never backward — and lives in the feature
layer where the point-in-time rules are enforced. Doing it at ingest would bake
a leakage risk into the warehouse.

## Idempotency and provenance

Every ingest step records the SHA-256 of its input and the row count of its
output in `staging/_manifest.json`. Re-running a step whose input has not
changed is a no-op. `--force` overrides.

This is what makes the pipeline cheap to iterate on and gives every table an
auditable provenance trail.
