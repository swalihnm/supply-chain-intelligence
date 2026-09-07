"""Contract tests for the transform and analytics layers.

These need a built warehouse and skip cleanly without one, so the suite stays
green on a fresh clone before `sci warehouse build` has run.
"""

from __future__ import annotations

import duckdb
import pytest

from sci.config import settings
from sci.warehouse import ANALYTICS_DIR, TRANSFORM_DIR

pytestmark = pytest.mark.skipif(
    not settings.duckdb_path.exists(), reason="warehouse not built (run: sci warehouse build)"
)


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        if c.execute("SELECT count(*) FROM core.panel").fetchone()[0] == 0:
            pytest.skip("core.panel is empty")
        yield c
    finally:
        c.close()


# --- transform layer -------------------------------------------------------


def test_panel_excludes_unavailable_rows(con):
    """core.panel must be strictly smaller than fact_sales: the availability filter."""
    raw = con.execute("SELECT count(*) FROM core.fact_sales").fetchone()[0]
    panel = con.execute("SELECT count(*) FROM core.panel").fetchone()[0]
    assert panel < raw
    # Roughly 21% of rows are pre-launch. Bracket it loosely so the test catches
    # the filter silently disappearing, not normal data drift.
    assert 0.70 < panel / raw < 0.90


def test_panel_has_no_null_prices(con):
    """A NULL price means unavailable, which panel is defined to exclude."""
    n = con.execute("SELECT count(*) FROM core.panel WHERE sell_price IS NULL").fetchone()[0]
    assert n == 0


def test_panel_grain_is_unique(con):
    total, distinct = con.execute(
        "SELECT count(*), count(DISTINCT (item_id, store_id, d)) FROM core.panel"
    ).fetchone()
    assert total == distinct


def test_panel_is_contiguous_in_d_per_series(con):
    """The stockout rule uses a ROWS window as a proxy for a DAYS window.

    That is only valid if each series' rows are consecutive in `d`. If a future
    change breaks contiguity, the 56-day trailing window silently becomes a
    56-observation window and the stockout thresholds go quietly wrong.
    """
    bad = con.execute(
        """
        SELECT count(*) FROM (
            SELECT item_id, store_id, count(*) AS n, max(d) - min(d) + 1 AS span
            FROM core.panel GROUP BY 1, 2
        ) WHERE n <> span
        """
    ).fetchone()[0]
    assert bad == 0


def test_stockout_flags_are_a_subset_of_zero_runs(con):
    z = con.execute("SELECT count(*) FROM core.zero_runs").fetchone()[0]
    f = con.execute("SELECT count(*) FROM core.stockout_flags").fetchone()[0]
    assert 0 < f < z


def test_stockout_censor_types_respect_the_28_day_split(con):
    bad = con.execute(
        """
        SELECT count(*) FROM core.stockout_flags
        WHERE (censor_type = 'stockout' AND run_len > 28)
           OR (censor_type = 'dormant'  AND run_len <= 28)
        """
    ).fetchone()[0]
    assert bad == 0


def test_stockout_rate_is_plausible(con):
    """A stockout rate far outside single digits means the rule has broken.

    An earlier lifetime-rate version flagged 18.3% of the panel with a mean run
    of 40 days, which is dormancy, not stockout. This pins the corrected
    behaviour so that regression cannot return unnoticed.
    """
    pct = con.execute(
        """
        SELECT 100.0 * sum(run_len) FILTER (WHERE censor_type = 'stockout')
               / (SELECT count(*) FROM core.panel)
        FROM core.stockout_flags
        """
    ).fetchone()[0]
    assert 1.0 < pct < 9.0, f"stockout rate {pct:.2f}% is outside the plausible band"


def test_series_class_covers_every_series(con):
    n = con.execute("SELECT count(*) FROM core.series_class").fetchone()[0]
    panel_series = con.execute(
        "SELECT count(DISTINCT (item_id, store_id)) FROM core.panel"
    ).fetchone()[0]
    assert n == panel_series == 30_490


def test_abc_classes_partition_revenue_as_defined(con):
    """A must carry ~80% of revenue, A+B ~95%. That is the definition, so it is
    a real check that cum_revenue_share was computed over the right ordering."""
    rows = dict(
        con.execute(
            """
            SELECT abc_class, 100.0 * sum(total_revenue) / sum(sum(total_revenue)) OVER ()
            FROM core.series_class GROUP BY 1
            """
        ).fetchall()
    )
    assert 78 < rows["A"] < 82
    assert 93 < rows["A"] + rows["B"] < 97


# --- analytics layer -------------------------------------------------------


@pytest.mark.parametrize("script", sorted(ANALYTICS_DIR.glob("*.sql")), ids=lambda p: p.stem)
def test_analytics_script_runs_and_returns_rows(con, script):
    df = con.sql(script.read_text(encoding="utf-8")).df()
    assert len(df) > 0, f"{script.name} returned no rows"


def test_transform_scripts_all_create_tables():
    """Every transform must materialise something, else the runner reports nothing."""
    import re

    for path in sorted(TRANSFORM_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        assert re.search(r"CREATE\s+OR\s+REPLACE\s+TABLE", sql, re.IGNORECASE), path.name


def test_alpha_sensitivity_is_monotonic(con):
    """Looser alpha must flag at least as many days.

    This caught a real bug: the report originally recomputed thresholds over the
    already-filtered core.stockout_flags, so every alpha above the build-time
    0.005 returned an identical figure and the parameter looked inert.
    """
    rows = con.execute(
        """
        WITH g AS (SELECT unnest([0.001, 0.005, 0.01, 0.05]) AS alpha)
        SELECT g.alpha,
               sum(z.run_len) FILTER (
                   WHERE z.run_len >= ceil(ln(g.alpha) / ln(nullif(1.0 - z.p_used, 0.0)))
                     AND z.run_len <= 28
               ) AS days
        FROM core.zero_runs z CROSS JOIN g
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    days = [d for _, d in rows]
    assert days == sorted(days), f"flagged days not monotonic in alpha: {rows}"
    assert days[0] < days[-1], "alpha has no effect at all - the filter is not being applied"
