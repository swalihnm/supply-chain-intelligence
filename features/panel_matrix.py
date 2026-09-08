"""Dense (n_series x n_days) matrices built once from core.panel.

WHY A DENSE MATRIX AT ALL
-------------------------
core.panel is 46.9M rows and ragged: each series exists only inside its own
availability window. Every baseline, every metric and every backtest fold needs
random access by (series, day). Doing that against a 46.9M-row long table means
a group-by per operation; doing it against a (30490, 1941) array is a slice.

The full grid is 59.2M cells. As float32 that is 237 MB per matrix, which fits
comfortably in memory and turns the whole model ladder from an ETL problem into
vectorised numpy.

Unavailable days are NaN rather than 0. That distinction is the entire point of
the availability filter and losing it here would undo the work in
sql/transform/010_panel.sql - `np.nanmean` skips a NaN and averages in a 0.

Built once and cached to artifacts/panel_matrices.npz. Rebuild with force=True
after any change to the transform layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.protocol import FIRST_D, LAST_D
from sci.config import settings
from sci.log import get_logger
from sci.warehouse import connect

log = get_logger(__name__)

N_DAYS = LAST_D - FIRST_D + 1
CACHE_NAME = "panel_matrices.npz"
CHUNK_SERIES = 8000  # keeps the arrow fetch under ~1 GB


@dataclass
class PanelMatrices:
    """Everything the model layer reads, aligned on a common index.

    Row i of every matrix is series i, described by row i of `series`.
    Column j is day FIRST_D + j.
    """

    units: np.ndarray  # (S, T) float32, NaN outside availability
    price: np.ndarray  # (S, T) float32, NaN outside availability
    stockout: np.ndarray  # (S, T) bool - censored, demand existed but was unseen
    dormant: np.ndarray  # (S, T) bool - censored, product effectively not on sale
    series: pl.DataFrame  # S rows: item_id, store_id, cat_id, dept_id, state_id, classes
    calendar: pl.DataFrame  # T rows: d, date, wday, month, year, snap_*, events

    @property
    def n_series(self) -> int:
        return self.units.shape[0]

    @property
    def n_days(self) -> int:
        return self.units.shape[1]

    @property
    def available(self) -> np.ndarray:
        return ~np.isnan(self.units)

    def censored(self) -> np.ndarray:
        """Any day whose observed zero should not be read as observed demand."""
        return self.stockout | self.dormant

    def training_units(self) -> np.ndarray:
        """Units with censored days masked to NaN.

        This is what models should train on. A stockout zero is not demand, and
        a dormant zero is not even an offer to sell. Leaving either in teaches
        the model to forecast supply failures.
        """
        out = self.units.copy()
        out[self.censored()] = np.nan
        return out

    def slice_days(self, start_d: int, end_d: int) -> tuple[np.ndarray, ...]:
        """(units, price, stockout, dormant) for an inclusive day range."""
        a, b = start_d - FIRST_D, end_d - FIRST_D + 1
        return self.units[:, a:b], self.price[:, a:b], self.stockout[:, a:b], self.dormant[:, a:b]


def _cache_path() -> Path:
    return settings.artifacts_dir / CACHE_NAME


def _load_series_index(con) -> pl.DataFrame:
    """Stable series ordering. Sorted so the index is reproducible across runs."""
    return con.execute(
        """
        SELECT
            row_number() OVER (ORDER BY item_id, store_id) - 1 AS series_idx,
            item_id, store_id, dept_id, cat_id, state_id,
            abc_class, xyz_class, sb_class,
            adi, cv_units, total_revenue, launch_d, last_available_d
        FROM core.series_class
        ORDER BY item_id, store_id
        """
    ).pl()


def _load_calendar(con) -> pl.DataFrame:
    return con.execute(
        f"""
        SELECT d, date, wm_yr_wk, weekday, wday, month, year,
               event_name_1, event_type_1, event_name_2, event_type_2,
               snap_CA, snap_TX, snap_WI
        FROM core.dim_date
        WHERE d BETWEEN {FIRST_D} AND {LAST_D}
        ORDER BY d
        """
    ).pl()


def build(force: bool = False) -> PanelMatrices:
    """Materialise the dense matrices, using the cache when it is current."""
    settings.ensure_dirs()
    cache = _cache_path()

    con = connect(read_only=True)
    try:
        series = _load_series_index(con)
        calendar = _load_calendar(con)

        if cache.exists() and not force:
            log.info("panel matrices: loading cache %s", cache.name)
            z = np.load(cache)
            if z["units"].shape == (series.height, N_DAYS):
                return PanelMatrices(
                    units=z["units"],
                    price=z["price"],
                    stockout=z["stockout"],
                    dormant=z["dormant"],
                    series=series,
                    calendar=calendar,
                )
            log.warning("cache shape %s does not match warehouse - rebuilding", z["units"].shape)

        n_series = series.height
        log.info("panel matrices: building %d series x %d days", n_series, N_DAYS)

        units = np.full((n_series, N_DAYS), np.nan, dtype=np.float32)
        price = np.full((n_series, N_DAYS), np.nan, dtype=np.float32)
        stockout = np.zeros((n_series, N_DAYS), dtype=bool)
        dormant = np.zeros((n_series, N_DAYS), dtype=bool)

        # Registered as a view rather than a TEMP TABLE: the connection is
        # read-only, and a view over the Arrow buffer joins just as well.
        con.register("_sidx", series.select("series_idx", "item_id", "store_id").to_arrow())

        for lo in range(0, n_series, CHUNK_SERIES):
            hi = min(lo + CHUNK_SERIES, n_series)
            log.info("  panel chunk series %d-%d", lo, hi - 1)
            chunk = con.execute(
                f"""
                SELECT s.series_idx, p.d, p.units, p.sell_price
                FROM core.panel p
                JOIN _sidx s ON s.item_id = p.item_id AND s.store_id = p.store_id
                WHERE s.series_idx >= {lo} AND s.series_idx < {hi}
                  AND p.d BETWEEN {FIRST_D} AND {LAST_D}
                """
            ).fetchnumpy()
            si = np.asarray(chunk["series_idx"], dtype=np.int32)
            dd = np.asarray(chunk["d"], dtype=np.int32) - FIRST_D
            units[si, dd] = np.asarray(chunk["units"], dtype=np.float32)
            price[si, dd] = np.asarray(chunk["sell_price"], dtype=np.float32)
            del chunk, si, dd

        # Censoring flags are stored as runs; expand them to days.
        log.info("  expanding censoring runs")
        runs = con.execute(
            """
            SELECT s.series_idx, f.run_start_d, f.run_end_d, f.censor_type
            FROM core.stockout_flags f
            JOIN _sidx s ON s.item_id = f.item_id AND s.store_id = f.store_id
            """
        ).fetchnumpy()
        r_idx = np.asarray(runs["series_idx"], dtype=np.int32)
        r_lo = np.asarray(runs["run_start_d"], dtype=np.int32) - FIRST_D
        r_hi = np.asarray(runs["run_end_d"], dtype=np.int32) - FIRST_D
        r_type = list(runs["censor_type"])
        for i, lo_d, hi_d, t in zip(r_idx, r_lo, r_hi, r_type, strict=True):
            target = stockout if t == "stockout" else dormant
            target[i, lo_d : hi_d + 1] = True

        log.info("panel matrices: caching to %s", cache.name)
        np.savez_compressed(cache, units=units, price=price, stockout=stockout, dormant=dormant)

        return PanelMatrices(
            units=units,
            price=price,
            stockout=stockout,
            dormant=dormant,
            series=series,
            calendar=calendar,
        )
    finally:
        con.close()
