"""Forecast accuracy metrics.

All functions are vectorised over series: inputs are 2-D arrays shaped
(n_series, horizon), and NaN marks a day that should not be scored (outside the
series' availability window). Scoring a day the product did not exist would be
meaningless, and averaging it in as a zero error would flatter every model.

Three metrics, because they answer different questions:

  RMSSE/WRMSSE  the M5 official metric. Scale-free, so it can be averaged across
                series with wildly different volumes, and dollar-weighted so
                that being wrong about a high-revenue SKU costs more.
  MASE          scale-free absolute error. Reported alongside RMSSE because
                squared error is dominated by a handful of spike days on
                intermittent series, and the two disagreeing is informative.
  pinball       the one that actually matters downstream. The optimizer consumes
                quantiles, so a model with a great point forecast and badly
                calibrated tails is useless to it.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# scaling denominators
# --------------------------------------------------------------------------


def naive_scale_squared(train: np.ndarray) -> np.ndarray:
    """Mean squared one-step naive error per series - the RMSSE denominator.

    M5 defines this as mean((y_t - y_{t-1})^2) over the training period. Because
    `train` comes from core.panel it is already restricted to each series'
    availability window, so the leading zeros that the official M5 definition
    explicitly excludes are gone before we get here.

    Returns NaN for series with too little history to form a difference.
    """
    diff = np.diff(train, axis=1)
    with np.errstate(invalid="ignore"):
        scale = np.nanmean(diff**2, axis=1)
    # A flat series gives a zero denominator; RMSSE is undefined there.
    scale = np.where(scale > 0, scale, np.nan)
    return scale


def naive_scale_absolute(train: np.ndarray) -> np.ndarray:
    """Mean absolute one-step naive error per series - the MASE denominator."""
    diff = np.diff(train, axis=1)
    with np.errstate(invalid="ignore"):
        scale = np.nanmean(np.abs(diff), axis=1)
    return np.where(scale > 0, scale, np.nan)


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------


def rmsse(actual: np.ndarray, pred: np.ndarray, scale_sq: np.ndarray) -> np.ndarray:
    """Root mean squared scaled error, per series."""
    err_sq = (actual - pred) ** 2
    with np.errstate(invalid="ignore"):
        mse = np.nanmean(err_sq, axis=1)
    return np.sqrt(mse / scale_sq)


def mase(actual: np.ndarray, pred: np.ndarray, scale_abs: np.ndarray) -> np.ndarray:
    """Mean absolute scaled error, per series."""
    with np.errstate(invalid="ignore"):
        mae = np.nanmean(np.abs(actual - pred), axis=1)
    return mae / scale_abs


def wrmsse(per_series_rmsse: np.ndarray, weights: np.ndarray) -> float:
    """Dollar-weighted mean of per-series RMSSE - the M5 headline number.

    Weights are renormalised over the series that produced a finite RMSSE, so
    dropping undefined series does not silently shrink the total.
    """
    ok = np.isfinite(per_series_rmsse) & np.isfinite(weights)
    if not ok.any():
        return float("nan")
    w = weights[ok]
    w = w / w.sum()
    return float((w * per_series_rmsse[ok]).sum())


def m5_weights(train: np.ndarray, price: np.ndarray, last_n: int = 28) -> np.ndarray:
    """M5 series weights: dollar sales over the last `last_n` days of training.

    Normalised to sum to 1. A series with no revenue in the window gets zero
    weight, which is correct - it contributes nothing to the business.
    """
    units = train[:, -last_n:]
    prices = price[:, -last_n:]
    with np.errstate(invalid="ignore"):
        revenue = np.nansum(units * prices, axis=1)
    total = revenue.sum()
    if total <= 0:
        return np.full(train.shape[0], np.nan)
    return revenue / total


# --------------------------------------------------------------------------
# probabilistic metrics
# --------------------------------------------------------------------------


def pinball_loss(actual: np.ndarray, pred_q: np.ndarray, q: float) -> np.ndarray:
    """Pinball (quantile) loss at level q, per series.

    Asymmetric on purpose: at q=0.95 under-forecasting is penalised 19x more
    heavily than over-forecasting, which is exactly the asymmetry a service
    level target encodes.
    """
    diff = actual - pred_q
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    with np.errstate(invalid="ignore"):
        return np.nanmean(loss, axis=1)


def mean_pinball(
    actual: np.ndarray, pred_by_q: dict[float, np.ndarray], weights: np.ndarray | None = None
) -> dict[float, float]:
    """Pinball loss averaged across series, per quantile level."""
    out = {}
    for q, pred in pred_by_q.items():
        per_series = pinball_loss(actual, pred, q)
        ok = np.isfinite(per_series)
        if not ok.any():
            out[q] = float("nan")
            continue
        if weights is None:
            out[q] = float(per_series[ok].mean())
        else:
            w = weights[ok]
            w = w / w.sum()
            out[q] = float((w * per_series[ok]).sum())
    return out


def coverage(actual: np.ndarray, pred_q: np.ndarray) -> float:
    """Empirical fraction of actuals at or below the predicted quantile.

    This is the calibration check that matters. If the 0.95 quantile only covers
    80% of outcomes, the optimizer will hit a 95% service target on paper and
    miss it in reality - the forecast error becomes a silent inventory shortfall.
    """
    mask = np.isfinite(actual) & np.isfinite(pred_q)
    if not mask.any():
        return float("nan")
    return float((actual[mask] <= pred_q[mask]).mean())


def calibration_table(
    actual: np.ndarray, pred_by_q: dict[float, np.ndarray]
) -> list[dict[str, float]]:
    """Nominal vs empirical coverage for each quantile level."""
    rows = []
    for q in sorted(pred_by_q):
        emp = coverage(actual, pred_by_q[q])
        rows.append(
            {
                "quantile": q,
                "nominal_coverage": q,
                "empirical_coverage": emp,
                "gap": emp - q,
            }
        )
    return rows


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------


def score_point_forecast(
    actual: np.ndarray,
    pred: np.ndarray,
    train: np.ndarray,
    price: np.ndarray,
) -> dict[str, float]:
    """Everything a point forecast is judged on, in one call."""
    scale_sq = naive_scale_squared(train)
    scale_abs = naive_scale_absolute(train)
    weights = m5_weights(train, price)

    per_rmsse = rmsse(actual, pred, scale_sq)
    per_mase = mase(actual, pred, scale_abs)

    return {
        "wrmsse": wrmsse(per_rmsse, weights),
        "rmsse_mean": float(np.nanmean(per_rmsse)),
        "mase_mean": float(np.nanmean(per_mase)),
        "n_series_scored": int(np.isfinite(per_rmsse).sum()),
    }
