"""Probabilistic forecasting: quantiles, and the calibration check on them.

WHY QUANTILES AND NOT A POINT FORECAST
--------------------------------------
The optimizer does not order the expected demand. It orders enough to meet a
service level - the 95th percentile of demand over the lead time. Safety stock
IS a statement about the tail of the predictive distribution, so a point
forecast contains none of the information the decision actually needs.

This matters more here than in a typical forecasting problem because 45.1% of
revenue sits in the AZ box: high value AND high variability. For those series
the mean is a poor summary and the spread is the whole question.

TWO ROUTES TO A DISTRIBUTION, BOTH KEPT
---------------------------------------
1. Direct quantile regression (LightGBM `objective=quantile`). One model per
   level. Flexible, makes no distributional assumption, but the levels are
   fitted independently and can cross.
2. Conformal calibration of the point model's residuals. Distribution-free with
   finite-sample coverage guarantees under exchangeability, and cheap - one
   extra pass over a calibration fold.

Both are produced and compared, because the interesting question is not which
has the better pinball loss but whether either is actually CALIBRATED. A 0.95
quantile that only covers 88% of outcomes will make the optimizer hit its
service target on paper and miss it in the store.

QUANTILE CROSSING
-----------------
Independently fitted quantile models can produce q0.9 > q0.95 for some rows.
That is incoherent and would make safety stock non-monotone in service level,
so the levels are sorted per row afterwards. The rate at which crossing occurs
is reported rather than hidden - it is a fit-quality signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from evaluation import metrics
from evaluation.protocol import QUANTILES, Fold
from features.build_features import DesignMatrix
from features.panel_matrix import PanelMatrices
from models import global_ml
from sci.log import get_logger

log = get_logger(__name__)


@dataclass
class QuantileForecast:
    """Predictions at each level for one fold, shaped (n_series, horizon)."""

    fold: str
    by_quantile: dict[float, np.ndarray]
    point: np.ndarray
    crossing_rate: float = 0.0
    models: dict[float, global_ml.GlobalModel] = field(default_factory=dict)

    def levels(self) -> list[float]:
        return sorted(self.by_quantile)


def enforce_monotone(by_q: dict[float, np.ndarray]) -> tuple[dict[float, np.ndarray], float]:
    """Sort predictions across quantile levels so they cannot cross.

    Returns the repaired dict and the fraction of cells that needed repair.
    """
    levels = sorted(by_q)
    stack = np.stack([by_q[q] for q in levels], axis=-1)  # (S, H, Q)

    finite = np.isfinite(stack).all(axis=-1)
    diffs = np.diff(stack, axis=-1)
    crossed = (diffs < -1e-9).any(axis=-1) & finite
    rate = float(crossed.sum() / max(finite.sum(), 1))

    fixed = np.sort(stack, axis=-1)
    return {q: fixed[..., i] for i, q in enumerate(levels)}, rate


def train_quantiles(
    pm: PanelMatrices,
    fold: Fold,
    quantiles: tuple[float, ...] = QUANTILES,
    num_boost_round: int = 300,
    dm: DesignMatrix | None = None,
) -> QuantileForecast:
    """Fit one booster per quantile level, plus the Tweedie point model.

    The design matrix is built once and reused across every level - featurising
    is the expensive part and it is identical for all of them.
    """
    dm = dm if dm is not None else global_ml._training_matrix(pm, fold, seed=0)

    point_model = global_ml.train(pm, fold, num_boost_round=num_boost_round, dm=dm)
    point = global_ml.predict(point_model, pm, fold)

    by_q: dict[float, np.ndarray] = {}
    models: dict[float, global_ml.GlobalModel] = {}
    for q in quantiles:
        m = global_ml.train(pm, fold, quantile=q, num_boost_round=num_boost_round, dm=dm)
        by_q[q] = global_ml.predict(m, pm, fold)
        models[q] = m

    by_q, rate = enforce_monotone(by_q)
    if rate > 0:
        log.info("  quantile crossing repaired on %.3f%% of cells", 100 * rate)

    return QuantileForecast(
        fold=fold.name, by_quantile=by_q, point=point, crossing_rate=rate, models=models
    )


# --------------------------------------------------------------------------
# conformal calibration
# --------------------------------------------------------------------------


def conformal_quantiles(
    point_cal: np.ndarray,
    actual_cal: np.ndarray,
    point_new: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
    by_group: np.ndarray | None = None,
) -> dict[float, np.ndarray]:
    """Split-conformal quantiles from residuals on a calibration fold.

    For each level q, take the empirical q-quantile of the calibration residuals
    (actual - point) and add it to the new point forecast. Distribution-free,
    and with `by_group` the residual quantile is computed within a group so that
    an intermittent series is not calibrated using a smooth series' errors.

    The exchangeability assumption is not innocent here: residuals are computed
    on one time window and applied to the next, and demand distributions drift.
    This is why it is compared against the direct quantile models rather than
    trusted on its own.
    """
    resid = actual_cal - point_cal
    out: dict[float, np.ndarray] = {}

    if by_group is None:
        flat = resid[np.isfinite(resid)]
        for q in quantiles:
            out[q] = np.clip(point_new + np.quantile(flat, q), 0, None)
        return out

    groups = np.unique(by_group)
    for q in quantiles:
        adj = np.zeros_like(point_new, dtype=np.float32)
        for g in groups:
            rows = by_group == g
            r = resid[rows]
            r = r[np.isfinite(r)]
            adj[rows] = np.quantile(r, q) if r.size > 50 else 0.0
        out[q] = np.clip(point_new + adj, 0, None)
    return out


# --------------------------------------------------------------------------
# calibration reporting
# --------------------------------------------------------------------------


def calibration_report(
    actual: np.ndarray, by_q: dict[float, np.ndarray], label: str = ""
) -> pl.DataFrame:
    """Nominal vs empirical coverage. The check that decides if the optimizer
    can trust the service level it thinks it is buying."""
    rows = metrics.calibration_table(actual, by_q)
    df = pl.DataFrame(rows)
    if label:
        df = df.with_columns(pl.lit(label).alias("method"))
    return df


def pinball_report(
    actual: np.ndarray,
    by_q: dict[float, np.ndarray],
    weights: np.ndarray | None = None,
    label: str = "",
) -> pl.DataFrame:
    losses = metrics.mean_pinball(actual, by_q, weights=weights)
    df = pl.DataFrame({"quantile": list(losses), "pinball_loss": [losses[q] for q in losses]})
    if label:
        df = df.with_columns(pl.lit(label).alias("method"))
    return df


def calibration_by_segment(
    actual: np.ndarray, by_q: dict[float, np.ndarray], segment: np.ndarray
) -> pl.DataFrame:
    """Coverage broken down by series class.

    A model can be well calibrated on average and badly calibrated exactly where
    the money is. Reporting only the pooled number would hide that.
    """
    rows = []
    for g in np.unique(segment):
        m = segment == g
        for q in sorted(by_q):
            rows.append(
                {
                    "segment": str(g),
                    "quantile": q,
                    "empirical_coverage": metrics.coverage(actual[m], by_q[q][m]),
                    "n_series": int(m.sum()),
                }
            )
    return pl.DataFrame(rows).with_columns(
        (pl.col("empirical_coverage") - pl.col("quantile")).alias("gap")
    )
