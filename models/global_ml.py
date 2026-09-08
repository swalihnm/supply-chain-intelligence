"""Global gradient-boosted model: ONE model across all 30,490 series.

WHY GLOBAL RATHER THAN 30,490 LOCAL MODELS
------------------------------------------
1. Data per series is thin and mostly zero. A local model for an intermittent
   SKU-store fits a handful of non-zero observations and overfits them. Pooling
   lets a series borrow strength from the thousands of others that behave like
   it.
2. Effects are shared. Day-of-week, SNAP, holiday and price-elasticity patterns
   are properties of the category and store, not of one SKU. Estimating them
   30,490 separate times wastes the data and gets a noisier answer every time.
3. New and short-history series work immediately. A local model needs history
   that a newly-launched SKU does not have; a global model just reads its
   category, store and price features. With a median launch at d=160 and half
   the catalogue starting after d=100, this is not a corner case here.
4. It is operable. One artifact to train, validate, ship and monitor.

The cost is that a genuinely idiosyncratic series gets pulled toward the crowd.
The error decomposition by Syntetos-Boylan class is what makes that visible
instead of assumed.

OBJECTIVE
---------
Tweedie, not squared error. The target is a zero-inflated non-negative count -
68% of observed days are zero - and Tweedie with 1 < p < 2 is a compound
Poisson-gamma that puts a point mass at zero and a continuous positive tail.
Squared error on this target pulls predictions toward a conditional mean that
under-predicts the spikes and over-predicts the zeros.

MEMORY
------
This machine has ~7 GB of RAM, so the full 46.9M-row training grid cannot be
featurised at once. Training samples rows across a two-year window rather than
truncating to recent history: seasonality matters more than having every row,
and a GBDT does not need contiguous rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from evaluation.protocol import HORIZON, Fold
from features.build_features import CATEGORICAL, DesignMatrix, build_xy
from features.panel_matrix import PanelMatrices
from sci.log import get_logger

log = get_logger(__name__)

# Two years of target days, sampled. 728 = 26 * 28.
TRAIN_DAYS_BACK = 728
TRAIN_ROW_SAMPLE = 0.18
FEATURE_CHUNK = 2000

BASE_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 8,
    "verbosity": -1,
    "force_row_wise": True,
}


@dataclass
class GlobalModel:
    booster: lgb.Booster
    feature_names: list[str]
    params: dict
    n_train_rows: int
    train_seconds: float
    quantile: float | None = None
    importance: dict[str, float] = field(default_factory=dict)


def _training_matrix(pm: PanelMatrices, fold: Fold, seed: int) -> DesignMatrix:
    start_d = max(2, fold.train_end_d - TRAIN_DAYS_BACK + 1)
    log.info(
        "  featurising training days d%d-%d (sample %.0f%%)",
        start_d,
        fold.train_end_d,
        100 * TRAIN_ROW_SAMPLE,
    )
    return build_xy(
        pm,
        start_d=start_d,
        end_d=fold.train_end_d,
        origin_d=fold.train_end_d,
        row_sample=TRAIN_ROW_SAMPLE,
        seed=seed,
        chunk=FEATURE_CHUNK,
        include_y=True,
    )


def train(
    pm: PanelMatrices,
    fold: Fold,
    objective: str = "tweedie",
    quantile: float | None = None,
    num_boost_round: int = 400,
    seed: int = 0,
    dm: DesignMatrix | None = None,
) -> GlobalModel:
    """Fit one global booster on everything up to fold.train_end_d.

    Pass `dm` to reuse an already-built design matrix across several quantile
    models - featurising is the expensive part and it is identical for all of
    them.
    """
    dm = dm if dm is not None else _training_matrix(pm, fold, seed)

    params = dict(BASE_PARAMS)
    if quantile is not None:
        params.update({"objective": "quantile", "alpha": quantile, "metric": "quantile"})
        params.pop("tweedie_variance_power", None)
    else:
        params["objective"] = objective

    cat_idx = [dm.feature_names.index(c) for c in CATEGORICAL if c in dm.feature_names]

    log.info(
        "  training %s on %s rows x %d features",
        f"q{quantile}" if quantile is not None else objective,
        f"{dm.n_rows:,}",
        len(dm.feature_names),
    )
    ds = lgb.Dataset(
        dm.X,
        label=dm.y,
        feature_name=dm.feature_names,
        categorical_feature=cat_idx,
        free_raw_data=False,
    )
    t0 = time.perf_counter()
    booster = lgb.train(params, ds, num_boost_round=num_boost_round)
    elapsed = time.perf_counter() - t0

    gains = booster.feature_importance(importance_type="gain")
    total = gains.sum() or 1.0
    importance = {
        n: float(g / total)
        for n, g in sorted(zip(dm.feature_names, gains, strict=True), key=lambda kv: -kv[1])
    }

    log.info("  trained in %.1fs", elapsed)
    return GlobalModel(
        booster=booster,
        feature_names=dm.feature_names,
        params=params,
        n_train_rows=dm.n_rows,
        train_seconds=elapsed,
        quantile=quantile,
        importance=importance,
    )


def predict(model: GlobalModel, pm: PanelMatrices, fold: Fold) -> np.ndarray:
    """Forecast the fold's validation window as a dense (n_series, horizon) array.

    Days outside a series' availability window stay NaN, matching how the
    actuals are shaped, so the metrics skip them consistently.
    """
    dm = build_xy(
        pm,
        start_d=fold.valid_start_d,
        end_d=fold.valid_end_d,
        origin_d=fold.train_end_d,
        row_sample=1.0,
        chunk=FEATURE_CHUNK,
        include_y=False,
    )
    raw = model.booster.predict(dm.X)

    out = np.full((pm.n_series, fold.horizon), np.nan, dtype=np.float32)
    out[dm.series_idx, dm.d - fold.valid_start_d] = np.clip(raw, 0, None)
    return out


def as_forecaster(model: GlobalModel, pm: PanelMatrices, fold: Fold):
    """Adapter so a trained global model plugs into the backtest harness.

    The harness passes (train_array, horizon); a global model needs the fold
    context instead, so it is closed over here. The train array is ignored on
    purpose - the model was already fitted on exactly that window.
    """

    def _f(train: np.ndarray, horizon: int) -> np.ndarray:
        if horizon != fold.horizon:
            raise ValueError(f"horizon {horizon} != fold horizon {fold.horizon}")
        return predict(model, pm, fold)

    return _f


def train_and_predict_folds(
    pm: PanelMatrices,
    folds: list[Fold],
    quantiles: tuple[float, ...] = (),
    num_boost_round: int = 400,
) -> dict[str, dict]:
    """Fit a fresh model per fold - never one model scored on its own training data.

    Returns {fold_name: {"point": pred, q: pred, ..., "model": GlobalModel}}.
    """
    out: dict[str, dict] = {}
    for fold in folds:
        log.info("fold %s", fold)
        dm = _training_matrix(pm, fold, seed=0)
        entry: dict = {}

        point = train(pm, fold, num_boost_round=num_boost_round, dm=dm)
        entry["point"] = predict(point, pm, fold)
        entry["model"] = point

        for q in quantiles:
            qm = train(pm, fold, quantile=q, num_boost_round=num_boost_round, dm=dm)
            entry[q] = predict(qm, pm, fold)
            entry[f"model_q{q}"] = qm

        del dm
        out[fold.name] = entry
    return out


def horizon_guard() -> int:
    """The lag floor. Exposed so tests can assert features never dip below it."""
    return HORIZON
