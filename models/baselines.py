"""Baseline forecasters. Not throwaways.

Reporting how much a gradient-boosted global model beats a *well-tuned*
seasonal naive is the honest framing. Reporting how much it beats nothing is
not. So these are implemented properly and kept in the ladder to the end.

Every function has the same signature:

    f(train: np.ndarray, horizon: int, **params) -> np.ndarray

`train` is (n_series, n_train_days) with NaN for days that must not be read as
demand - outside the availability window, or censored by the stockout proxy.
The return is (n_series, horizon).

All of them are vectorised across series. Croston and TSB need a sequential
pass over time, but the inner operation is a 30,490-wide vector update, so the
whole catalogue fits in one loop of 1,941 steps.

INCLUDING A ZERO FORECAST IS DELIBERATE
---------------------------------------
72.6% of these series are intermittent. On a scale-free metric, "always predict
zero" is a genuinely competitive baseline for the sparsest of them, and any
model that cannot beat it on a given segment has learned nothing there. Leaving
it out would hide that.
"""

from __future__ import annotations

import numpy as np


def _last_valid(train: np.ndarray) -> np.ndarray:
    """Most recent non-NaN value per series, 0 where a series has none."""
    valid = ~np.isnan(train)
    has_any = valid.any(axis=1)
    # Index of the last True along axis 1.
    last_idx = train.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    out = np.zeros(train.shape[0], dtype=np.float32)
    rows = np.arange(train.shape[0])
    out[has_any] = train[rows[has_any], last_idx[has_any]]
    return np.nan_to_num(out, nan=0.0)


def zero(train: np.ndarray, horizon: int) -> np.ndarray:
    """Predict no demand. The floor any real model must clear."""
    return np.zeros((train.shape[0], horizon), dtype=np.float32)


def naive(train: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    return np.repeat(_last_valid(train)[:, None], horizon, axis=1)


def seasonal_naive(train: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last full seasonal cycle.

    Weekly (season=7) for retail: the day-of-week signal is the single strongest
    pattern in this data, worth +19% to +26% on weekends.

    NaN inside the last cycle falls back to the series' recent mean rather than
    to zero, so a censored Saturday does not propagate a false zero into every
    future Saturday.
    """
    n_series = train.shape[0]
    cycle = train[:, -season:].astype(np.float32).copy()

    with np.errstate(invalid="ignore", divide="ignore"):
        recent = np.nanmean(train[:, -4 * season :], axis=1)
    recent = np.nan_to_num(recent, nan=0.0)

    gaps = np.isnan(cycle)
    cycle[gaps] = np.repeat(recent[:, None], season, axis=1)[gaps]

    reps = int(np.ceil(horizon / season))
    return np.tile(cycle, (1, reps))[:, :horizon].reshape(n_series, horizon)


def moving_average(train: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    """Flat forecast at the mean of the last `window` observed days."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(train[:, -window:], axis=1)
    mean = np.nan_to_num(mean, nan=0.0)
    return np.repeat(mean[:, None], horizon, axis=1).astype(np.float32)


def croston(train: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Classic Croston (1972) for intermittent demand.

    Smooths demand SIZE and the INTERVAL between demands separately, then
    forecasts the rate size/interval. The separation is the whole idea: a series
    selling 10 units every 20 days and one selling 0.5 units daily have the same
    mean, and averaging them together is what makes naive methods fail on
    intermittent data.

    Known to be biased upward (Syntetos-Boylan 2001). TSB below is the usual
    correction; both are reported so the difference is visible rather than
    assumed.
    """
    n_series, n_t = train.shape
    z = np.zeros(n_series, dtype=np.float64)  # smoothed demand size
    p = np.ones(n_series, dtype=np.float64)  # smoothed interval
    q = np.ones(n_series, dtype=np.float64)  # periods since last demand
    started = np.zeros(n_series, dtype=bool)

    for t in range(n_t):
        y = train[:, t]
        observed = ~np.isnan(y)
        occurred = observed & (y > 0)

        # First demand initialises the state rather than smoothing toward it
        # from zero, which would otherwise take ~1/alpha periods to recover.
        first = occurred & ~started
        z[first] = y[first]
        p[first] = q[first]
        started |= first

        upd = occurred & ~first
        z[upd] = alpha * y[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * q[upd] + (1 - alpha) * p[upd]

        q[occurred] = 1.0
        # Only count days the product was actually on sale toward the interval.
        q[observed & ~occurred] += 1.0

    rate = np.where(started, z / np.maximum(p, 1e-9), 0.0)
    return np.repeat(rate[:, None], horizon, axis=1).astype(np.float32)


def tsb(train: np.ndarray, horizon: int, alpha: float = 0.1, beta: float = 0.05) -> np.ndarray:
    """Teunter-Syntetos-Babai (2011).

    Smooths demand PROBABILITY instead of interval, and updates that probability
    on every period rather than only on demand periods. That is what lets it
    decay toward zero for a dying SKU - Croston, updating only when demand
    occurs, keeps forecasting the last known rate forever on an obsolete item.

    Given that 18.4% of these series are lumpy and many fade out, this matters
    more here than the textbook comparison suggests.
    """
    n_series, n_t = train.shape
    z = np.zeros(n_series, dtype=np.float64)  # smoothed demand size
    prob = np.zeros(n_series, dtype=np.float64)  # smoothed demand probability
    started = np.zeros(n_series, dtype=bool)

    for t in range(n_t):
        y = train[:, t]
        observed = ~np.isnan(y)
        occurred = observed & (y > 0)

        first = occurred & ~started
        z[first] = y[first]
        prob[first] = 1.0
        started |= first

        upd = occurred & ~first
        z[upd] += alpha * (y[upd] - z[upd])

        # Probability updates on EVERY observed period, up or down.
        obs_started = observed & started
        target = np.where(occurred, 1.0, 0.0)
        prob[obs_started] += beta * (target[obs_started] - prob[obs_started])

    rate = np.where(started, prob * z, 0.0)
    return np.repeat(rate[:, None], horizon, axis=1).astype(np.float32)


def seasonal_croston(train: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Croston rate reshaped by the series' own day-of-week profile.

    Croston produces a flat rate, which throws away the strongest signal in
    retail data. This scales that rate by a normalised weekly profile so the
    total over the horizon is preserved but the shape is not flat.

    Not a standard method - it is a cheap combination worth including because it
    isolates how much of any ML gain is just day-of-week structure.
    """
    rate = croston(train, 1, alpha=alpha)[:, 0]

    # Weekly profile from the last 8 weeks of training, normalised to mean 1.
    tail = train[:, -56:]
    n_t = tail.shape[1]
    profile = np.ones((train.shape[0], 7), dtype=np.float64)
    for k in range(7):
        cols = np.arange(n_t - 1 - k, -1, -7)  # every 7th day back from the end
        if len(cols) == 0:
            continue
        with np.errstate(invalid="ignore"):
            profile[:, 6 - k] = np.nanmean(tail[:, cols], axis=1)
    profile = np.nan_to_num(profile, nan=0.0)
    denom = profile.mean(axis=1, keepdims=True)
    profile = np.divide(profile, denom, out=np.ones_like(profile), where=denom > 0)

    reps = int(np.ceil(horizon / 7))
    shape = np.tile(profile, (1, reps))[:, :horizon]
    return (rate[:, None] * shape).astype(np.float32)


REGISTRY = {
    "zero": zero,
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "croston": croston,
    "tsb": tsb,
    "seasonal_croston": seasonal_croston,
}
