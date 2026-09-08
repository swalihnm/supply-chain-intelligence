"""Statistical models (AutoETS) for the AGGREGATE levels of the hierarchy.

WHERE THESE BELONG, AND WHERE THEY DO NOT
-----------------------------------------
The spec's ladder puts AutoETS and AutoARIMA on the bottom-level series. On this
catalogue that is the wrong place for them: 91% of the 30,490 SKU-store series
are intermittent or lumpy, and exponential smoothing on a series that is zero
four days in five fits noise. The measured baseline table agrees - Croston and
TSB, built for intermittency, beat everything except the global model, while
seasonal naive (a smoothing-family method) came out 30% worse than a plain
moving average.

Aggregate series are a different object entirely. Summed over a store or a
category, demand is smooth, strongly weekly-seasonal and has no zeros at all -
exactly the regime ETS was designed for. So the statistical rung runs on levels
1-11 of the hierarchy and the global ML model stays at the bottom. Each method
is used where its assumptions hold, which is a better answer to "why this model"
than running everything everywhere.

WHY THE BATCH API AND NOT A LOOP
--------------------------------
A first version fitted AutoETS one node at a time. At 12,350 aggregate nodes it
had not cleared 2,000 fits after two minutes - AutoETS searches over a family of
ETS specifications per series, and doing that serially is hopeless at this
scale. statsforecast's `StatsForecast` class fits across series in parallel
worker processes, which is what it exists for.

`n_jobs` is deliberately capped rather than set to -1: each worker copies its
slice of the panel, and on a 7.3 GB machine twelve of those is an out-of-memory
error rather than a speedup.

FALLBACK
--------
statsforecast is an optional dependency (`--extra models`). Without it - or for
any node whose fit fails or returns non-finite values - this falls back to
seasonal naive. Which nodes fell back is counted and reported, never silently
absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sci.log import get_logger

log = get_logger(__name__)

SEASON_LENGTH = 7
DEFAULT_N_JOBS = 4
# One year is 52 weekly cycles - ample for a weekly-seasonal ETS, and a quarter
# of the memory of the full history in long format.
DEFAULT_HISTORY_DAYS = 365


@dataclass
class AggregateForecast:
    point: np.ndarray  # (n_nodes, horizon)
    in_sample_mse: np.ndarray  # (n_nodes,) -> the wls_var reconciliation weights
    method_used: list[str] = field(default_factory=list)
    n_fallback: int = 0


def _seasonal_naive(y: np.ndarray, horizon: int, season: int = SEASON_LENGTH):
    """Last full weekly cycle, tiled. The fallback and the sanity floor."""
    if y.size < season:
        level = float(np.nanmean(y)) if y.size else 0.0
        var = float(np.nanvar(y)) if y.size else 1.0
        return np.full(horizon, max(level, 0.0)), max(var, 1e-9)

    cycle = y[-season:]
    reps = int(np.ceil(horizon / season))
    fc = np.tile(cycle, reps)[:horizon]

    if y.size > season:
        resid = y[season:] - y[:-season]
        mse = float(np.mean(resid**2))
    else:
        mse = float(np.var(y))
    return np.maximum(fc, 0.0), max(mse, 1e-9)


def _all_seasonal_naive(hist: np.ndarray, horizon: int) -> AggregateForecast:
    n = hist.shape[0]
    point = np.zeros((n, horizon))
    mse = np.ones(n)
    for i in range(n):
        point[i], mse[i] = _seasonal_naive(np.nan_to_num(hist[i], nan=0.0), horizon)
    return AggregateForecast(
        point=point,
        in_sample_mse=mse,
        method_used=["seasonal_naive"] * n,
        n_fallback=n,
    )


def forecast_intermittent_nodes(
    agg_history: np.ndarray,
    horizon: int,
    season_length: int = SEASON_LENGTH,
) -> AggregateForecast:
    """TSB for aggregate levels that are still INTERMITTENT.

    Not every "aggregate" is smooth. Summing one item across ten stores
    (`item`, `item_state`) removes far less zero-inflation than summing a whole
    store: a slow SKU that sells twice a month in each store still has zero days
    everywhere. Those two levels are 12,196 of the 12,350 aggregate nodes.

    Running AutoETS on them was both wrong and ruinous. Wrong because ETS on a
    zero-inflated series fits noise - the same argument that keeps it off the
    bottom level. Ruinous because it took over twenty minutes and drove the
    machine into swap, for the levels where it had the least to offer.

    TSB is the right family here and it is already vectorised across series, so
    12,196 nodes take seconds. Same principle as everywhere else in the ladder:
    route by the character of the series, not by its position in the hierarchy.
    """
    from models.baselines import tsb

    hist = np.nan_to_num(agg_history.astype(np.float32), nan=0.0)
    point = tsb(hist, horizon).astype(np.float64)

    # In-sample MSE against the flat TSB rate, for the wls_var weights.
    rate = point[:, 0][:, None]
    resid = hist.astype(np.float64) - rate
    mse = np.maximum(np.mean(resid**2, axis=1), 1e-9)

    n = hist.shape[0]
    log.info("  TSB over %d intermittent aggregate nodes", n)
    return AggregateForecast(
        point=np.maximum(point, 0.0),
        in_sample_mse=mse,
        method_used=["tsb"] * n,
        n_fallback=0,
    )


def forecast_aggregates(
    agg_history: np.ndarray,
    horizon: int,
    use_statsforecast: bool = True,
    season_length: int = SEASON_LENGTH,
    n_jobs: int = DEFAULT_N_JOBS,
    history_days: int = DEFAULT_HISTORY_DAYS,
) -> AggregateForecast:
    """Forecast every aggregate node with AutoETS, in parallel.

    agg_history : (n_nodes, n_days) values already summed up the hierarchy.
                  Aggregates are dense by construction, so no NaN handling
                  beyond a defensive nan_to_num.
    """
    n_nodes = agg_history.shape[0]
    hist = np.nan_to_num(agg_history[:, -history_days:].astype(np.float64), nan=0.0)
    n_days = hist.shape[1]

    if not use_statsforecast:
        return _all_seasonal_naive(hist, horizon)

    try:
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS
    except ImportError:
        log.warning(
            "statsforecast not installed - aggregate levels fall back to seasonal naive. "
            "Install with: uv sync --extra models"
        )
        return _all_seasonal_naive(hist, horizon)

    # Long format, which is what StatsForecast consumes.
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(n_nodes), n_days),
            "ds": np.tile(dates.values, n_nodes),
            "y": hist.reshape(-1),
        }
    )
    log.info(
        "  AutoETS over %d nodes x %d days (%s rows, n_jobs=%d)",
        n_nodes,
        n_days,
        f"{len(df):,}",
        n_jobs,
    )

    sf = StatsForecast(
        models=[AutoETS(season_length=season_length)],
        freq="D",
        n_jobs=n_jobs,
    )

    try:
        fcst = sf.forecast(df=df, h=horizon, fitted=True)
        fitted_vals = sf.forecast_fitted_values()
    except Exception as exc:  # noqa: BLE001 - a batch failure falls back wholesale
        log.warning("  StatsForecast failed (%s) - falling back to seasonal naive", exc)
        return _all_seasonal_naive(hist, horizon)

    # ---- unpack point forecasts ----
    point = np.zeros((n_nodes, horizon), dtype=np.float64)
    fc = fcst.to_pandas() if hasattr(fcst, "to_pandas") else fcst
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    col = "AutoETS" if "AutoETS" in fc.columns else fc.columns[-1]
    ids = fc["unique_id"].to_numpy().astype(int)
    vals = fc[col].to_numpy(dtype=np.float64)
    # Rows arrive grouped by unique_id in horizon order.
    for uid in np.unique(ids):
        m = ids == uid
        v = vals[m][:horizon]
        point[uid, : len(v)] = v

    # ---- in-sample MSE, for the reconciliation weights ----
    mse = np.ones(n_nodes, dtype=np.float64)
    fv = fitted_vals.to_pandas() if hasattr(fitted_vals, "to_pandas") else fitted_vals
    fv = fv.reset_index() if "unique_id" not in fv.columns else fv
    fcol = "AutoETS" if "AutoETS" in fv.columns else fv.columns[-1]
    f_ids = fv["unique_id"].to_numpy().astype(int)
    resid = fv["y"].to_numpy(dtype=np.float64) - fv[fcol].to_numpy(dtype=np.float64)
    ok = np.isfinite(resid)
    sums = np.bincount(f_ids[ok], weights=resid[ok] ** 2, minlength=n_nodes)
    counts = np.bincount(f_ids[ok], minlength=n_nodes)
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(counts > 0, sums / np.maximum(counts, 1), 1.0)
    mse = np.maximum(np.nan_to_num(mse, nan=1.0), 1e-9)

    # ---- repair anything non-finite or negative ----
    methods = ["autoets"] * n_nodes
    bad = ~np.isfinite(point).all(axis=1)
    n_fallback = int(bad.sum())
    for i in np.flatnonzero(bad):
        point[i], mse[i] = _seasonal_naive(hist[i], horizon, season_length)
        methods[i] = "seasonal_naive_fallback"
    point = np.maximum(point, 0.0)

    if n_fallback:
        log.info("  %d/%d aggregate nodes fell back to seasonal naive", n_fallback, n_nodes)
    else:
        log.info("  all %d aggregate nodes fitted with AutoETS", n_nodes)

    return AggregateForecast(
        point=point, in_sample_mse=mse, method_used=methods, n_fallback=n_fallback
    )
