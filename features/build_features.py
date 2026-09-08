"""Point-in-time-correct feature construction for the global model.

LEAKAGE RULES, ENFORCED BY CONSTRUCTION
---------------------------------------
Every lag and rolling statistic is offset by at least HORIZON (28) days. A
feature for target day t reads only observations from day t-28 or earlier.

That is a *direct* multi-step design rather than a recursive one. It costs some
accuracy at short lead times - the model never sees yesterday - but it removes
recursion entirely, so there is no error compounding and no way for a
one-step-ahead feature to sneak into a 28-step-ahead forecast. On a 28-day
replenishment horizon the trade is worth it, and it makes the leakage audit a
matter of arithmetic rather than trust.

WHAT IS LEGITIMATELY KNOWN AT FORECAST TIME
-------------------------------------------
  Calendar (weekday, month, SNAP, holidays)  known years ahead. Safe.
  Price                                      M5 publishes prices for the forecast
                                             weeks, and retailers do plan prices
                                             ahead. Used as-is; recorded as an
                                             assumption because a real deployment
                                             would need the planned price file.
  Weather                                    NOT known 28 days out. This is the
                                             trap. See below.

THE WEATHER DECISION
--------------------
Feeding actual future weather into a 28-day forecast is leakage, and it is the
kind that quietly inflates a headline number. No one has a usable 28-day
forecast of daily rainfall.

So the model gets CLIMATOLOGY instead: the historical average for that state and
day-of-year, computed only from days strictly before the forecast origin. That
is genuinely knowable in advance - it is just seasonality. It is also weaker
than real weather data would be, which is the honest cost of not cheating.

One actual-weather feature is kept: the anomaly at t-28, which is observed by
forecast time and carries some persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl

from evaluation.protocol import FIRST_D, HORIZON
from features.panel_matrix import PanelMatrices
from sci.log import get_logger

log = get_logger(__name__)

# Lags, all >= HORIZON so nothing recursive is required.
LAGS = (28, 35, 42, 49, 56, 364)
ROLL_WINDOWS = (7, 28, 56, 112)

CATEGORICAL = ("cat_code", "dept_code", "store_code", "state_code", "sb_code", "wday", "month")


@dataclass
class DesignMatrix:
    X: np.ndarray  # (n_rows, n_features) float32
    y: np.ndarray  # (n_rows,) float32
    series_idx: np.ndarray  # (n_rows,) int32
    d: np.ndarray  # (n_rows,) int32
    feature_names: list[str]

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]


def _codes(series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Integer codes for the categorical series attributes."""
    out = {}
    for col, name in [
        ("cat_id", "cat_code"),
        ("dept_id", "dept_code"),
        ("store_id", "store_code"),
        ("state_id", "state_code"),
        ("sb_class", "sb_code"),
    ]:
        vals = series[col].to_numpy()
        uniq = {v: i for i, v in enumerate(sorted(set(vals.tolist())))}
        out[name] = np.array([uniq[v] for v in vals], dtype=np.float32)
    return out


def _climatology(pm: PanelMatrices, origin_d: int) -> tuple[np.ndarray, np.ndarray]:
    """(state, day_of_year) mean temperature and precipitation.

    Built ONLY from days strictly before `origin_d`, so a fold cannot see the
    weather of its own validation window even in averaged form.

    Returns two (n_states, 367) arrays indexed by day-of-year.
    """
    from sci.warehouse import connect

    cal = pm.calendar
    cutoff_date = cal.filter(pl.col("d") == origin_d)["date"][0]

    con = connect(read_only=True)
    try:
        w = con.execute(
            "SELECT state_id, date, temperature_mean, precipitation_sum FROM core.fact_weather"
        ).pl()
    finally:
        con.close()

    w = w.filter(pl.col("date") < cutoff_date).with_columns(
        pl.col("date").dt.ordinal_day().alias("doy")
    )
    agg = w.group_by(["state_id", "doy"]).agg(
        pl.col("temperature_mean").mean().alias("t"),
        pl.col("precipitation_sum").mean().alias("p"),
    )

    states = sorted(w["state_id"].unique().to_list())
    s_index = {s: i for i, s in enumerate(states)}
    temp = np.full((len(states), 367), np.nan, dtype=np.float32)
    prec = np.full((len(states), 367), np.nan, dtype=np.float32)
    for row in agg.iter_rows(named=True):
        temp[s_index[row["state_id"]], row["doy"]] = row["t"]
        prec[s_index[row["state_id"]], row["doy"]] = row["p"]

    # Fill any doy with no history from the state's own mean.
    for i in range(len(states)):
        temp[i] = np.nan_to_num(temp[i], nan=float(np.nanmean(temp[i])))
        prec[i] = np.nan_to_num(prec[i], nan=float(np.nanmean(prec[i])))
    return temp, prec


def _actual_weather(pm: PanelMatrices) -> np.ndarray:
    """(n_states, n_days) actual mean temperature, for the t-28 anomaly feature."""
    from sci.warehouse import connect

    con = connect(read_only=True)
    try:
        w = con.execute(
            """
            SELECT w.state_id, dt.d, w.temperature_mean
            FROM core.fact_weather w
            JOIN core.dim_date dt ON dt.date = w.date
            ORDER BY w.state_id, dt.d
            """
        ).pl()
    finally:
        con.close()

    states = sorted(w["state_id"].unique().to_list())
    s_index = {s: i for i, s in enumerate(states)}
    out = np.full((len(states), pm.n_days), np.nan, dtype=np.float32)
    for row in w.iter_rows(named=True):
        j = row["d"] - FIRST_D
        if 0 <= j < pm.n_days:
            out[s_index[row["state_id"]], j] = row["temperature_mean"]
    return out


def _append_feature(
    feats: list[np.ndarray],
    names: list[str],
    n_rows: int,
    name: str,
    arr: np.ndarray,
) -> None:
    """Accumulate one feature block.

    Module-level and explicitly parameterised rather than a closure inside the
    chunk loop: a closure would capture the loop's accumulators by reference,
    which is a real footgun even where (as here) it happens to be called within
    the same iteration.
    """
    feats.append(arr.astype(np.float32).reshape(n_rows, -1))
    names.append(name)


def _rolling(u0: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix sums for O(1) rolling windows. Returns (sum_cs, count_cs).

    Column j of the result is the cumulative total over days [0, j). Padding
    with a leading zero column makes window arithmetic a plain subtraction with
    no special case at the start of the series.
    """
    n = u0.shape[0]
    s = np.zeros((n, u0.shape[1] + 1), dtype=np.float32)
    c = np.zeros((n, u0.shape[1] + 1), dtype=np.float32)
    np.cumsum(u0, axis=1, out=s[:, 1:])
    np.cumsum(valid, axis=1, out=c[:, 1:])
    return s, c


def build_xy(
    pm: PanelMatrices,
    start_d: int,
    end_d: int,
    origin_d: int | None = None,
    row_sample: float = 1.0,
    seed: int = 0,
    chunk: int = 6000,
    include_y: bool = True,
) -> DesignMatrix:
    """Build the design matrix for target days in [start_d, end_d].

    origin_d is the last day the model is allowed to know about. It defaults to
    start_d - 1, which is what every backtest fold wants. Climatology is built
    from days before origin_d.

    Rows are produced in series chunks to stay inside a small memory budget;
    with 7 GB of RAM the full 30,490 x 1,941 grid cannot be featurised at once.
    """
    origin_d = origin_d if origin_d is not None else start_d - 1
    rng = np.random.default_rng(seed)

    cal = pm.calendar
    target_j = np.arange(start_d - FIRST_D, end_d - FIRST_D + 1)

    # Calendar vectors, aligned to matrix columns.
    wday = cal["wday"].to_numpy().astype(np.float32)
    month = cal["month"].to_numpy().astype(np.float32)
    dom = cal["date"].dt.day().to_numpy().astype(np.float32)
    doy = cal["date"].dt.ordinal_day().to_numpy().astype(np.int32)
    has_event = (~cal["event_name_1"].is_null()).to_numpy().astype(np.float32)
    snap_by_state = {
        "CA": cal["snap_CA"].to_numpy().astype(np.float32),
        "TX": cal["snap_TX"].to_numpy().astype(np.float32),
        "WI": cal["snap_WI"].to_numpy().astype(np.float32),
    }

    codes = _codes(pm.series)
    state_ids = pm.series["state_id"].to_numpy()
    launch_d = pm.series["launch_d"].to_numpy().astype(np.float32)

    clim_t, clim_p = _climatology(pm, origin_d)
    actual_t = _actual_weather(pm)
    states_sorted = sorted(set(state_ids.tolist()))
    state_row = {s: i for i, s in enumerate(states_sorted)}

    train_units = pm.training_units()

    names: list[str] = []
    Xs, ys, sidx, dsel = [], [], [], []

    n_series = pm.n_series
    for lo in range(0, n_series, chunk):
        hi = min(lo + chunk, n_series)
        u = train_units[lo:hi]
        price = pm.price[lo:hi]

        valid = (~np.isnan(u)).astype(np.float32)
        u0 = np.nan_to_num(u, nan=0.0)
        s_cs, c_cs = _rolling(u0, valid)

        p_valid = (~np.isnan(price)).astype(np.float32)
        p0 = np.nan_to_num(price, nan=0.0)
        ps_cs, pc_cs = _rolling(p0, p_valid)

        nonzero = (u0 > 0).astype(np.float32)
        nz_cs, _ = _rolling(nonzero, valid)

        feats: list[np.ndarray] = []
        chunk_names: list[str] = []
        n_rows_c = hi - lo

        add = partial(_append_feature, feats, chunk_names, n_rows_c)

        # --- lagged units ------------------------------------------------
        for lag in LAGS:
            src = target_j - lag
            ok = src >= 0
            col = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
            col[:, ok] = u0[:, src[ok]]
            add(f"lag_{lag}", col)

        # --- rolling stats over windows ENDING at t-HORIZON --------------
        end_j = target_j - HORIZON + 1  # exclusive prefix index
        for w in ROLL_WINDOWS:
            beg_j = end_j - w
            ok = beg_j >= 0
            mean = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
            frac = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
            if ok.any():
                tot = s_cs[:, end_j[ok]] - s_cs[:, beg_j[ok]]
                cnt = c_cs[:, end_j[ok]] - c_cs[:, beg_j[ok]]
                nzt = nz_cs[:, end_j[ok]] - nz_cs[:, beg_j[ok]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean[:, ok] = np.where(cnt > 0, tot / cnt, np.nan)
                    frac[:, ok] = np.where(cnt > 0, nzt / cnt, np.nan)
            add(f"roll_mean_{w}", mean)
            add(f"nonzero_frac_{w}", frac)

        # --- same-weekday history ----------------------------------------
        # A plain rolling mean averages the weekly shape away, and the first
        # model built without these put 81.7% of its gain on roll_mean_112 while
        # giving `wday` 0.93% - it was reconstructing a long moving average and
        # ignoring the strongest signal in retail data (weekends run +19% to
        # +26%). These read the same weekday 8 times back, all at t-28 or older.
        dow_stack = np.full((n_rows_c, len(target_j), 8), np.nan, dtype=np.float32)
        for k in range(8):
            src = target_j - HORIZON - 7 * k
            ok = src >= 0
            dow_stack[:, ok, k] = u0[:, src[ok]]
        with np.errstate(invalid="ignore"):
            dow_mean = np.nanmean(dow_stack, axis=2)
        add("dow_mean_8", dow_mean)
        del dow_stack

        # The weekly SHAPE, separated from the level: how much busier is this
        # weekday than the series' own recent average?
        beg56 = end_j - 56
        ok = beg56 >= 0
        base56 = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
        if ok.any():
            t56 = s_cs[:, end_j[ok]] - s_cs[:, beg56[ok]]
            c56 = c_cs[:, end_j[ok]] - c_cs[:, beg56[ok]]
            with np.errstate(invalid="ignore", divide="ignore"):
                base56[:, ok] = np.where(c56 > 0, t56 / c56, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            add("dow_ratio_8", np.where(base56 > 0, dow_mean / base56, np.nan))

        # --- price -------------------------------------------------------
        cur_price = price[:, target_j]
        add("price", cur_price)

        beg_j = end_j - 28
        ok = beg_j >= 0
        pmean = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
        if ok.any():
            pt = ps_cs[:, end_j[ok]] - ps_cs[:, beg_j[ok]]
            pc = pc_cs[:, end_j[ok]] - pc_cs[:, beg_j[ok]]
            with np.errstate(invalid="ignore", divide="ignore"):
                pmean[:, ok] = np.where(pc > 0, pt / pc, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            add("price_rel_28", np.where(pmean > 0, cur_price / pmean, np.nan))

        src = target_j - 7
        ok = src >= 0
        prev_p = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
        prev_p[:, ok] = price[:, src[ok]]
        with np.errstate(invalid="ignore", divide="ignore"):
            add("price_change_7", np.where(prev_p > 0, cur_price / prev_p - 1.0, np.nan))

        # --- calendar (broadcast across series) --------------------------
        add("wday", np.tile(wday[target_j], (n_rows_c, 1)))
        add("month", np.tile(month[target_j], (n_rows_c, 1)))
        add("day_of_month", np.tile(dom[target_j], (n_rows_c, 1)))
        add("has_event", np.tile(has_event[target_j], (n_rows_c, 1)))

        snap = np.empty((n_rows_c, len(target_j)), dtype=np.float32)
        for si, st in enumerate(state_ids[lo:hi]):
            snap[si] = snap_by_state[st][target_j]
        add("snap", snap)

        # --- weather: climatology only, plus an observed t-28 anomaly ----
        s_rows = np.array([state_row[s] for s in state_ids[lo:hi]], dtype=np.int32)
        doy_t = doy[target_j]
        add("temp_climatology", clim_t[s_rows][:, doy_t])
        add("precip_climatology", clim_p[s_rows][:, doy_t])

        src = target_j - HORIZON
        ok = src >= 0
        anom = np.full((n_rows_c, len(target_j)), np.nan, dtype=np.float32)
        if ok.any():
            act = actual_t[s_rows][:, src[ok]]
            base = clim_t[s_rows][:, doy[src[ok]]]
            anom[:, ok] = act - base
        add("temp_anomaly_lag28", anom)

        # --- series attributes -------------------------------------------
        for name, arr in codes.items():
            add(name, np.tile(arr[lo:hi][:, None], (1, len(target_j))))
        add(
            "days_since_launch",
            (target_j[None, :] + FIRST_D) - launch_d[lo:hi][:, None],
        )

        # --- assemble ------------------------------------------------------
        n_feat = len(feats)
        Xc = np.empty((n_rows_c * len(target_j), n_feat), dtype=np.float32)
        for k, f in enumerate(feats):
            Xc[:, k] = f.reshape(-1)
        del feats

        yc = pm.units[lo:hi][:, target_j].reshape(-1)
        si_c = np.repeat(np.arange(lo, hi, dtype=np.int32), len(target_j))
        d_c = np.tile((target_j + FIRST_D).astype(np.int32), n_rows_c)

        # Drop rows the model must not learn from: unavailable days, and
        # censored days where the observed zero is not real demand.
        cens = pm.censored()[lo:hi][:, target_j].reshape(-1)
        keep = ~np.isnan(yc)
        if include_y:
            keep &= ~cens
        if row_sample < 1.0:
            keep &= rng.random(keep.shape[0]) < row_sample

        Xs.append(Xc[keep])
        ys.append(yc[keep])
        sidx.append(si_c[keep])
        dsel.append(d_c[keep])
        if not names:
            names = chunk_names
        del Xc, yc, si_c, d_c, keep, u, price, s_cs, c_cs, ps_cs, pc_cs, nz_cs

    return DesignMatrix(
        X=np.concatenate(Xs),
        y=np.concatenate(ys).astype(np.float32),
        series_idx=np.concatenate(sidx),
        d=np.concatenate(dsel),
        feature_names=names,
    )
