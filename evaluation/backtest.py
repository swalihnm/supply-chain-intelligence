"""The backtest harness.

Runs any forecaster over the frozen protocol folds and returns a tidy result
frame. Nothing here decides what a fold is - that lives in protocol.py and is
imported, never redefined.

WHAT GETS SCORED, AND WHAT DOES NOT
-----------------------------------
Models TRAIN on `training_units` (censored days masked to NaN), because a
stockout zero is not demand.

Models are SCORED against raw `units`, because that is what actually happened
and what the business paid for. Scoring against imputed values would let a model
be rewarded for matching our own imputation.

Days outside a series' availability window are excluded from both. NaN
propagates through the metrics, which use nan-aware reductions throughout.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from evaluation import metrics
from evaluation.protocol import FIRST_D, Fold, all_folds
from features.panel_matrix import PanelMatrices
from sci.log import get_logger

log = get_logger(__name__)

Forecaster = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class FoldResult:
    model: str
    fold: str
    wrmsse: float
    rmsse_mean: float
    mase_mean: float
    n_series_scored: int
    fit_seconds: float
    per_series_rmsse: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)


def _fold_arrays(pm: PanelMatrices, fold: Fold) -> tuple[np.ndarray, ...]:
    """Training and validation slices for one fold.

    train_clean : censored days masked - what the model sees
    train_raw   : unmasked - used only for the metric scaling denominators,
                  which M5 defines over observed history
    actual      : raw validation units - what the model is scored against
    """
    t_end = fold.train_end_d - FIRST_D + 1
    v_lo = fold.valid_start_d - FIRST_D
    v_hi = fold.valid_end_d - FIRST_D + 1

    train_clean = pm.training_units()[:, :t_end]
    train_raw = pm.units[:, :t_end]
    price_train = pm.price[:, :t_end]
    actual = pm.units[:, v_lo:v_hi]
    return train_clean, train_raw, price_train, actual


def run_model(
    pm: PanelMatrices,
    name: str,
    fn: Forecaster,
    folds: list[Fold] | None = None,
    include_test: bool = False,
) -> list[FoldResult]:
    """Backtest one forecaster across the protocol folds."""
    folds = folds if folds is not None else all_folds(include_test=include_test)
    results: list[FoldResult] = []

    for fold in folds:
        train_clean, train_raw, price_train, actual = _fold_arrays(pm, fold)

        t0 = time.perf_counter()
        pred = fn(train_clean, fold.horizon)
        elapsed = time.perf_counter() - t0

        if pred.shape != actual.shape:
            raise ValueError(
                f"{name} returned {pred.shape}, expected {actual.shape} for {fold.name}"
            )

        # Demand cannot be negative. Clipping here rather than inside each model
        # keeps the models honest about what they actually produce.
        pred = np.clip(pred, 0, None)

        scale_sq = metrics.naive_scale_squared(train_raw)
        scale_abs = metrics.naive_scale_absolute(train_raw)
        weights = metrics.m5_weights(train_raw, price_train)

        per_rmsse = metrics.rmsse(actual, pred, scale_sq)
        per_mase = metrics.mase(actual, pred, scale_abs)

        results.append(
            FoldResult(
                model=name,
                fold=fold.name,
                wrmsse=metrics.wrmsse(per_rmsse, weights),
                rmsse_mean=float(np.nanmean(per_rmsse)),
                mase_mean=float(np.nanmean(per_mase)),
                n_series_scored=int(np.isfinite(per_rmsse).sum()),
                fit_seconds=elapsed,
                per_series_rmsse=per_rmsse,
                weights=weights,
            )
        )
        log.info(
            "%-18s %-5s  WRMSSE %.4f  MASE %.4f  (%.1fs)",
            name,
            fold.name,
            results[-1].wrmsse,
            results[-1].mase_mean,
            elapsed,
        )
    return results


def run_ladder(
    pm: PanelMatrices,
    models: dict[str, Forecaster],
    include_test: bool = False,
) -> pl.DataFrame:
    """Backtest several forecasters and return one tidy frame."""
    rows = []
    keep: list[FoldResult] = []
    for name, fn in models.items():
        for r in run_model(pm, name, fn, include_test=include_test):
            keep.append(r)
            rows.append(
                {
                    "model": r.model,
                    "fold": r.fold,
                    "wrmsse": r.wrmsse,
                    "rmsse_mean": r.rmsse_mean,
                    "mase_mean": r.mase_mean,
                    "n_series_scored": r.n_series_scored,
                    "fit_seconds": r.fit_seconds,
                }
            )
    return pl.DataFrame(rows), keep


def summarise(df: pl.DataFrame) -> pl.DataFrame:
    """Mean across CV folds, ranked. The test fold is reported separately."""
    cv = df.filter(pl.col("fold") != "test")
    return (
        cv.group_by("model")
        .agg(
            pl.col("wrmsse").mean().alias("wrmsse"),
            pl.col("wrmsse").std().alias("wrmsse_sd"),
            pl.col("mase_mean").mean().alias("mase"),
            pl.col("fit_seconds").sum().alias("total_seconds"),
            pl.len().alias("folds"),
        )
        .sort("wrmsse")
        .with_columns(pl.col("wrmsse").rank().cast(pl.Int32).alias("rank"))
    )


def error_by_segment(
    pm: PanelMatrices, results: list[FoldResult], by: str = "sb_class"
) -> pl.DataFrame:
    """Decompose per-series RMSSE by segmentation class.

    Required analysis, not optional: a headline WRMSSE hides that a model can be
    excellent on smooth series and useless on intermittent ones, which is
    precisely the split that decides whether the optimizer can trust it.
    """
    seg = pm.series[by].to_numpy()
    rows = []
    for r in results:
        finite = np.isfinite(r.per_series_rmsse)
        for cls in np.unique(seg):
            m = finite & (seg == cls)
            if not m.any():
                continue
            w = r.weights[m]
            w = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else None
            rows.append(
                {
                    "model": r.model,
                    "fold": r.fold,
                    by: str(cls),
                    "n_series": int(m.sum()),
                    "rmsse_mean": float(np.nanmean(r.per_series_rmsse[m])),
                    "wrmsse": float((w * r.per_series_rmsse[m]).sum()) if w is not None else np.nan,
                }
            )
    out = pl.DataFrame(rows)
    return (
        out.filter(pl.col("fold") != "test")
        .group_by(["model", by])
        .agg(
            pl.col("rmsse_mean").mean(),
            pl.col("wrmsse").mean(),
            pl.col("n_series").first(),
        )
        .sort([by, "rmsse_mean"])
    )
