"""Pack everything the API and dashboard need into one file.

Usage:  uv run python scripts/build_serving_bundle.py

The full panel matrices are ~500 MB in memory, which is the wrong thing to put
in a container that only needs to answer questions about the forecast horizon.
This extracts the slice that actually gets served:

    history       last 365 days of clean demand, for live re-simulation
    demand/price  the simulation window (warm-up + scored)
    forecasts     point and quantile predictions for the test horizon
    series        item/store/class metadata

Result is roughly 30 MB compressed, so /simulate can run a REAL simulation over
all 30,490 series on request rather than interpolating a precomputed table.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from evaluation.protocol import FIRST_D, test_fold
from evaluation.simulate_counterfactual import WARMUP_DAYS
from features import panel_matrix
from sci.config import settings
from sci.log import get_logger

log = get_logger(__name__)

HISTORY_DAYS = 365
BUNDLE_NAME = "serving_bundle.npz"


def main() -> None:
    settings.ensure_dirs()
    art = settings.artifacts_dir
    pm = panel_matrix.build()
    fold = test_fold()

    sim_start_d = fold.valid_start_d - WARMUP_DAYS
    a = sim_start_d - FIRST_D
    b = fold.valid_end_d - FIRST_D + 1

    demand = pm.units[:, a:b].astype(np.float32)
    price = pm.price[:, a:b].astype(np.float32)
    med_price = np.nanmedian(pm.price, axis=1).astype(np.float32)
    price = np.where(np.isnan(price), med_price[:, None], price)

    hist_end = fold.train_end_d - FIRST_D + 1
    history = pm.training_units()[:, max(0, hist_end - HISTORY_DAYS) : hist_end].astype(np.float32)

    fc_path = art / "forecast_test.npz"
    payload: dict[str, np.ndarray] = {
        "history": history,
        "demand": demand,
        "price": price,
        "median_price": med_price,
        "sim_start_d": np.array([sim_start_d]),
        "valid_start_d": np.array([fold.valid_start_d]),
        "valid_end_d": np.array([fold.valid_end_d]),
        "warmup_days": np.array([WARMUP_DAYS]),
    }

    if fc_path.exists():
        fc = np.load(fc_path)
        for k in fc.files:
            payload[f"forecast_{k}"] = fc[k].astype(np.float32)
        log.info("included forecasts: %s", ", ".join(fc.files))
    else:
        log.warning("forecast_test.npz not found - bundle will have no forecasts")

    # Actual units over the scored window, for the forecast-explorer chart.
    payload["actual_test"] = pm.units[
        :, fold.valid_start_d - FIRST_D : fold.valid_end_d - FIRST_D + 1
    ].astype(np.float32)

    out = art / BUNDLE_NAME
    np.savez_compressed(out, **payload)

    # Series metadata travels as parquet - it is a table, not an array.
    pm.series.write_parquet(art / "serving_series.parquet")
    pm.calendar.write_parquet(art / "serving_calendar.parquet")

    size_mb = out.stat().st_size / 1e6
    log.info("wrote %s (%.1f MB) + serving_series.parquet", out.name, size_mb)

    counts = {
        "series": pm.n_series,
        "history_days": history.shape[1],
        "sim_days": demand.shape[1],
    }
    print(pl.DataFrame([counts]))


if __name__ == "__main__":
    main()
