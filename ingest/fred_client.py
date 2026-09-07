"""Macro covariates from FRED (US Federal Reserve economic data).

Requires a free API key in FRED_API_KEY. When the key is absent this module
raises `FredKeyMissing` and the CLI skips the step rather than failing the run -
macro is a nice-to-have covariate, not part of the transaction spine.

Series are stored at their NATIVE frequency. Resampling to daily is a feature
decision (forward-fill only, never interpolate backwards) and therefore belongs
in the feature layer, where the point-in-time rules are enforced. Doing it here
would bake a leakage risk into the warehouse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sci.config import settings
from sci.log import get_logger
from sci.manifest import Manifest, text_sha256

log = get_logger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

OBS_START = "2010-12-01"
OBS_END = "2016-07-31"


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    label: str
    freq: str
    # state_id when the series is state-specific, else None for national
    state_id: str | None = None


SERIES = (
    FredSeries("CPIAUCSL", "cpi_all_urban", "monthly"),
    FredSeries("GASREGW", "gas_price_regular", "weekly"),
    FredSeries("UNRATE", "unemployment_rate", "monthly"),
    FredSeries("CAUR", "unemployment_rate_state", "monthly", state_id="CA"),
    FredSeries("TXUR", "unemployment_rate_state", "monthly", state_id="TX"),
    FredSeries("WIUR", "unemployment_rate_state", "monthly", state_id="WI"),
)


class FredKeyMissing(RuntimeError):
    pass


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch(series_id: str, api_key: str, start: str, end: str) -> dict:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(FRED_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def _to_frame(payload: dict, series: FredSeries) -> pl.DataFrame:
    obs = payload.get("observations", [])
    if not obs:
        return pl.DataFrame(
            schema={
                "series_id": pl.Utf8,
                "label": pl.Utf8,
                "state_id": pl.Utf8,
                "freq": pl.Utf8,
                "date": pl.Date,
                "value": pl.Float64,
            }
        )
    return (
        pl.DataFrame(obs)
        .select(
            pl.lit(series.series_id).alias("series_id"),
            pl.lit(series.label).alias("label"),
            pl.lit(series.state_id, dtype=pl.Utf8).alias("state_id"),
            pl.lit(series.freq).alias("freq"),
            pl.col("date").str.to_date("%Y-%m-%d"),
            # FRED encodes a missing observation as the string "."
            pl.when(pl.col("value") == ".")
            .then(None)
            .otherwise(pl.col("value"))
            .cast(pl.Float64)
            .alias("value"),
        )
        .drop_nulls("value")
    )


def ingest_fred(force: bool = False, start: str = OBS_START, end: str = OBS_END) -> int:
    """Fetch all macro series and stage them as one long parquet table."""
    api_key = settings.fred_api_key.strip()
    if not api_key:
        raise FredKeyMissing(
            "FRED_API_KEY is empty. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and put it in .env. "
            "Macro covariates are optional - the rest of the pipeline runs without them."
        )

    settings.ensure_dirs()
    raw = settings.raw_dir / "fred"
    raw.mkdir(parents=True, exist_ok=True)
    stage = settings.staging_dir / "external"
    stage.mkdir(parents=True, exist_ok=True)
    out = stage / "macro_series.parquet"

    mf = Manifest(settings.staging_dir / "_manifest.json")
    key = "macro_series"
    input_sha = text_sha256(start, end, *(s.series_id for s in SERIES))
    if not force and mf.is_current(key, input_sha, out):
        rows = (mf.get(key) or {}).get("rows", 0)
        log.info("%-16s up to date (%s rows) - skipping", key, f"{rows:,}")
        return rows

    frames: list[pl.DataFrame] = []
    for series in SERIES:
        cache: Path = raw / f"{series.series_id}_{start}_{end}.json"
        if cache.exists() and not force:
            log.info("fred %s: using cached raw pull", series.series_id)
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            log.info("fred %s: fetching", series.series_id)
            payload = _fetch(series.series_id, api_key, start, end)
            cache.write_text(json.dumps(payload), encoding="utf-8")
        frames.append(_to_frame(payload, series))

    df = pl.concat(frames).sort("series_id", "date")
    df.write_parquet(out, compression="zstd")
    mf.record(key, input_sha, out, df.height, series=[s.series_id for s in SERIES])
    log.info("%-16s wrote %s rows across %d series", key, f"{df.height:,}", len(SERIES))
    return df.height
