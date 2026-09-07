"""Warehouse contract tests.

The grain assertions here are the reason fact_sales carries no PRIMARY KEY:
the uniqueness constraint is still enforced, just in the test suite rather than
by a 59M-row index maintained on every insert.
"""

from __future__ import annotations

import duckdb
import pytest

from sci.config import settings
from sci.warehouse import DDL_DIR, run_ddl

EXPECTED_TABLES = {
    "dim_date",
    "dim_item",
    "dim_store",
    "fact_sales",
    "fact_price",
    "fact_weather",
    "fact_macro",
}


@pytest.fixture
def mem_con():
    con = duckdb.connect(":memory:")
    run_ddl(con)
    yield con
    con.close()


def test_ddl_files_exist():
    assert sorted(p.name for p in DDL_DIR.glob("*.sql"))[0] == "001_star_schema.sql"


def test_ddl_creates_the_full_star_schema(mem_con):
    names = {
        r[0]
        for r in mem_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'core'"
        ).fetchall()
    }
    assert names >= EXPECTED_TABLES


def test_fact_weather_grain_is_enforced(mem_con):
    row = ("CA", "2011-01-29", "Los Angeles", 20.0, 10.0, 15.0, 19.0, 0.0, 0.0, 5.0)
    ins = "INSERT INTO core.fact_weather VALUES (?,?,?,?,?,?,?,?,?,?)"
    mem_con.execute(ins, list(row))
    with pytest.raises(duckdb.ConstraintException):
        mem_con.execute(ins, list(row))


def test_dim_date_rejects_duplicate_day_index(mem_con):
    row = ("1", "2011-01-29", 11101, "Saturday", 1, 1, 2011, None, None, None, None, 0, 0, 0)
    ins = "INSERT INTO core.dim_date VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    mem_con.execute(ins, list(row))
    with pytest.raises(duckdb.ConstraintException):
        mem_con.execute(ins, list(row))


# --- tests that need a real built warehouse -------------------------------

pytestmark_needs_db = pytest.mark.skipif(
    not settings.duckdb_path.exists(), reason="warehouse not built yet"
)


@pytestmark_needs_db
def test_built_warehouse_sales_grain_is_unique():
    """(item_id, store_id, d) must be unique in fact_sales. Skipped until M5 lands."""
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        total, distinct = con.execute(
            "SELECT count(*), count(DISTINCT (item_id, store_id, d)) FROM core.fact_sales"
        ).fetchone()
    finally:
        con.close()
    if total == 0:
        pytest.skip("fact_sales is empty - M5 not ingested yet")
    assert total == distinct, f"fact_sales has {total - distinct} duplicate grain rows"


@pytestmark_needs_db
def test_built_warehouse_weather_covers_every_state():
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        states = {
            r[0] for r in con.execute("SELECT DISTINCT state_id FROM core.fact_weather").fetchall()
        }
    finally:
        con.close()
    if not states:
        pytest.skip("fact_weather is empty")
    assert states == {"CA", "TX", "WI"}
