"""Daily historical weather per M5 state, from the Open-Meteo archive API.

No API key required.

ASSUMPTION (documented in docs/03-assumptions-and-limitations.md): M5 does not
disclose the city of each store, only its state. We therefore pull weather for
one representative metro per state and join on `state_id`. This is a real
approximation: a CA_1 store in San Diego and a CA_3 store in Sacramento would
see different weather, and we are giving them the same. The alternative -
inventing store cities - would be worse because it would look precise while
being fabricated.
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

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# M5 spans 2011-01-29 (d_1) through 2016-06-19 (d_1941). We pull a small margin
# on each side so lag/rolling features at the boundaries are not truncated.
START_DATE = "2010-12-01"
END_DATE = "2016-07-31"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]


@dataclass(frozen=True)
class StateSite:
    state_id: str
    city: str
    latitude: float
    longitude: float
    timezone: str


SITES = (
    StateSite("CA", "Los Angeles", 34.0522, -118.2437, "America/Los_Angeles"),
    StateSite("TX", "Houston", 29.7604, -95.3698, "America/Chicago"),
    StateSite("WI", "Madison", 43.0731, -89.4012, "America/Chicago"),
)


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch(site: StateSite, start: str, end: str) -> dict:
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": site.timezone,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(ARCHIVE_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def _to_frame(payload: dict, site: StateSite) -> pl.DataFrame:
    daily = payload["daily"]
    df = pl.DataFrame(daily).with_columns(
        pl.col("time").str.to_date("%Y-%m-%d").alias("date"),
        pl.lit(site.state_id).alias("state_id"),
        pl.lit(site.city).alias("weather_city"),
    )
    rename = {v: v.replace("_2m", "").replace("_10m", "") for v in DAILY_VARS}
    df = df.drop("time").rename(rename)
    numeric = [rename[v] for v in DAILY_VARS]
    return df.select(
        "state_id",
        "date",
        "weather_city",
        *[pl.col(c).cast(pl.Float32) for c in numeric],
    ).sort("state_id", "date")


def ingest_weather(force: bool = False, start: str = START_DATE, end: str = END_DATE) -> int:
    """Fetch daily weather for every M5 state and stage it as one parquet table."""
    settings.ensure_dirs()
    raw = settings.raw_dir / "weather"
    raw.mkdir(parents=True, exist_ok=True)
    stage = settings.staging_dir / "external"
    stage.mkdir(parents=True, exist_ok=True)
    out = stage / "weather_daily.parquet"

    mf = Manifest(settings.staging_dir / "_manifest.json")
    key = "weather_daily"
    input_sha = text_sha256(start, end, ",".join(DAILY_VARS), *(s.city for s in SITES))
    if not force and mf.is_current(key, input_sha, out):
        rows = (mf.get(key) or {}).get("rows", 0)
        log.info("%-16s up to date (%s rows) - skipping", key, f"{rows:,}")
        return rows

    frames: list[pl.DataFrame] = []
    for site in SITES:
        cache: Path = raw / f"{site.state_id}_{start}_{end}.json"
        if cache.exists() and not force:
            log.info("weather %s: using cached raw pull", site.state_id)
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            log.info("weather %s: fetching %s (%s..%s)", site.state_id, site.city, start, end)
            payload = _fetch(site, start, end)
            cache.write_text(json.dumps(payload), encoding="utf-8")
        frames.append(_to_frame(payload, site))

    df = pl.concat(frames).sort("state_id", "date")
    df.write_parquet(out, compression="zstd")
    mf.record(key, input_sha, out, df.height, sites=[s.state_id for s in SITES])
    log.info("%-16s wrote %s rows", key, f"{df.height:,}")
    return df.height
