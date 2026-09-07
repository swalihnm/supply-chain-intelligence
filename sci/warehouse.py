"""DuckDB warehouse: execute the hand-written DDL, load staging parquet.

Loads are INSERT ... SELECT straight off the parquet files. DuckDB reads
parquet natively, so the 59M-row sales fact never passes through Python.

Tables whose staging parquet is absent are skipped with a warning rather than
failing the build, so the warehouse is useful before every source has landed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from sci.config import REPO_ROOT, settings
from sci.log import get_logger

log = get_logger(__name__)

DDL_DIR = REPO_ROOT / "sql" / "ddl"
TRANSFORM_DIR = REPO_ROOT / "sql" / "transform"
ANALYTICS_DIR = REPO_ROOT / "sql" / "analytics"


@dataclass(frozen=True)
class LoadSpec:
    table: str
    parquet: Path
    columns: tuple[str, ...]
    required: bool = True


def _specs() -> list[LoadSpec]:
    m5 = settings.staging_dir / "m5"
    ext = settings.staging_dir / "external"
    return [
        LoadSpec(
            "core.dim_date",
            m5 / "calendar.parquet",
            (
                "d",
                "date",
                "wm_yr_wk",
                "weekday",
                "wday",
                "month",
                "year",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ),
        ),
        LoadSpec("core.dim_item", m5 / "dim_item.parquet", ("item_id", "dept_id", "cat_id")),
        LoadSpec("core.dim_store", m5 / "dim_store.parquet", ("store_id", "state_id")),
        LoadSpec(
            "core.fact_sales", m5 / "sales_long.parquet", ("item_id", "store_id", "d", "units")
        ),
        LoadSpec(
            "core.fact_price",
            m5 / "sell_prices.parquet",
            ("item_id", "store_id", "wm_yr_wk", "sell_price"),
        ),
        LoadSpec(
            "core.fact_weather",
            ext / "weather_daily.parquet",
            (
                "state_id",
                "date",
                "weather_city",
                "temperature_max",
                "temperature_min",
                "temperature_mean",
                "apparent_temperature_max",
                "precipitation_sum",
                "snowfall_sum",
                "windspeed_max",
            ),
            required=False,
        ),
        LoadSpec(
            "core.fact_macro",
            ext / "macro_series.parquet",
            ("series_id", "label", "state_id", "freq", "date", "value"),
            required=False,
        ),
    ]


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    settings.ensure_dirs()
    return duckdb.connect(str(settings.duckdb_path), read_only=read_only)


def run_ddl(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Execute every sql/ddl/*.sql in filename order."""
    applied = []
    for path in sorted(DDL_DIR.glob("*.sql")):
        log.info("ddl  %s", path.name)
        con.execute(path.read_text(encoding="utf-8"))
        applied.append(path.name)
    if not applied:
        raise FileNotFoundError(f"No DDL files found in {DDL_DIR}")
    return applied


def load_tables(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Load each staging parquet into its warehouse table. Returns row counts."""
    counts: dict[str, int] = {}
    for spec in _specs():
        if not spec.parquet.exists():
            level = log.warning if spec.required else log.info
            level(
                "%-18s SKIPPED - %s not staged yet",
                spec.table.split(".")[-1],
                spec.parquet.name,
            )
            continue
        cols = ", ".join(spec.columns)
        con.execute(f"DELETE FROM {spec.table}")
        con.execute(
            f"INSERT INTO {spec.table} ({cols}) SELECT {cols} FROM read_parquet(?)",
            [str(spec.parquet)],
        )
        n = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
        counts[spec.table] = n
        log.info("%-18s loaded %s rows", spec.table.split(".")[-1], f"{n:,}")
    return counts


def run_transforms(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Execute sql/transform/*.sql in filename order, after the loads.

    These build the derived tables the analytics and model layers read:
    core.panel (availability-filtered), core.stockout_flags, core.series_class.
    They cannot live in the DDL because they are CTAS over loaded data.
    """
    counts: dict[str, int] = {}
    for path in sorted(TRANSFORM_DIR.glob("*.sql")):
        log.info("transform  %s", path.name)
        con.execute(path.read_text(encoding="utf-8"))
        # Report the row count of whatever table the script created.
        for table in _created_tables(path.read_text(encoding="utf-8")):
            n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            counts[table] = n
            log.info("%-22s %s rows", table, f"{n:,}")
    return counts


def _created_tables(sql: str) -> list[str]:
    """Pull `CREATE OR REPLACE TABLE <name> AS` targets out of a transform script."""
    import re

    return re.findall(r"CREATE\s+OR\s+REPLACE\s+TABLE\s+([\w.]+)\s+AS", sql, flags=re.IGNORECASE)


def build(force_ddl: bool = True, transforms: bool = True) -> dict[str, int]:
    """Full warehouse build: DDL, loads, then derived transforms."""
    with connect() as con:
        if force_ddl:
            run_ddl(con)
        counts = load_tables(con)
        if transforms:
            counts.update(run_transforms(con))
        return counts
