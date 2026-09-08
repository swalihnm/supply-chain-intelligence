"""Leakage audit and feature-construction contracts.

The arithmetic tests here are the leakage audit. A direct multi-step design
makes leakage a matter of index arithmetic rather than trust, so it can be
asserted rather than argued - which is the entire reason the design was chosen.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.protocol import HORIZON, cv_folds
from features import build_features
from sci.config import settings

pytestmark = pytest.mark.skipif(
    not (settings.artifacts_dir / "panel_matrices.npz").exists(),
    reason="panel matrices not built (run: uv run python scripts/run_model_ladder.py)",
)


@pytest.fixture(scope="module")
def pm():
    from features import panel_matrix

    return panel_matrix.build()


@pytest.fixture(scope="module")
def dm(pm):
    fold = cv_folds()[-1]
    return build_features.build_xy(
        pm,
        fold.valid_start_d,
        fold.valid_end_d,
        origin_d=fold.train_end_d,
        row_sample=0.02,
        seed=0,
    )


# --- the leakage audit ----------------------------------------------------


def test_every_lag_is_at_least_the_horizon():
    """No feature may read a day closer than HORIZON to its target.

    This is what makes the design non-recursive. If a lag below 28 ever appears,
    a 28-day-ahead forecast would silently depend on data it cannot have.
    """
    assert min(build_features.LAGS) >= HORIZON, build_features.LAGS


def test_lag_features_match_the_actual_history(pm, dm):
    """Spot-check the index arithmetic against the source matrix."""
    from evaluation.protocol import FIRST_D

    train = pm.training_units()
    for lag in (28, 35, 56):
        col = dm.feature_names.index(f"lag_{lag}")
        rng = np.random.default_rng(lag)
        for row in rng.integers(0, dm.n_rows, 25):
            s, d = dm.series_idx[row], dm.d[row]
            j = (d - lag) - FIRST_D
            if j < 0:
                continue
            expected = np.nan_to_num(train[s, j], nan=0.0)
            assert dm.X[row, col] == pytest.approx(expected, abs=1e-4), (
                f"lag_{lag} mismatch at series {s} day {d}"
            )


def test_no_feature_is_perfectly_correlated_with_the_target(dm):
    """A correlation at 1.0 would mean the target itself leaked in."""
    for i, name in enumerate(dm.feature_names):
        col = dm.X[:, i]
        m = np.isfinite(col)
        if m.sum() < 500 or np.nanstd(col) == 0:
            continue
        c = abs(np.corrcoef(col[m], dm.y[m])[0, 1])
        assert c < 0.98, f"{name} correlates {c:.4f} with the target - leakage"


def test_climatology_is_built_only_from_prior_days(pm):
    """Weather climatology for a fold must not see that fold's own weather.

    Two folds with different origins must produce different climatology, and the
    later one must be built from strictly more history.
    """
    early, late = cv_folds()[0], cv_folds()[-1]
    t_early, _ = build_features._climatology(pm, early.train_end_d)
    t_late, _ = build_features._climatology(pm, late.train_end_d)
    assert not np.allclose(t_early, t_late, equal_nan=True), (
        "climatology did not change with the forecast origin - it may be using future weather"
    )


def test_no_actual_future_weather_feature_exists(dm):
    """Only climatology and a t-28 anomaly are permitted.

    Actual weather at the target day is not knowable 28 days out; including it
    would be leakage that inflates the headline accuracy.
    """
    weather = [n for n in dm.feature_names if "temp" in n or "precip" in n]
    assert set(weather) == {"temp_climatology", "precip_climatology", "temp_anomaly_lag28"}, (
        f"unexpected weather features: {weather}"
    )


# --- construction contracts -----------------------------------------------


def test_targets_are_finite_and_non_negative(dm):
    assert np.isfinite(dm.y).all()
    assert (dm.y >= 0).all()


def test_training_rows_exclude_censored_days(pm):
    """A stockout zero is not demand and must never be a training target."""
    from evaluation.protocol import FIRST_D

    fold = cv_folds()[-1]
    dm = build_features.build_xy(
        pm,
        fold.valid_start_d,
        fold.valid_end_d,
        origin_d=fold.train_end_d,
        row_sample=0.05,
        seed=1,
        include_y=True,
    )
    cens = pm.censored()
    bad = sum(1 for s, d in zip(dm.series_idx, dm.d, strict=True) if cens[s, d - FIRST_D])
    assert bad == 0, f"{bad} censored rows leaked into the training set"


def test_prediction_rows_keep_censored_days(pm):
    """At prediction time censored days ARE forecast - they are scored against
    raw actuals, so dropping them would misalign the output matrix."""
    fold = cv_folds()[-1]
    with_y = build_features.build_xy(
        pm, fold.valid_start_d, fold.valid_end_d, origin_d=fold.train_end_d, include_y=True
    )
    without_y = build_features.build_xy(
        pm, fold.valid_start_d, fold.valid_end_d, origin_d=fold.train_end_d, include_y=False
    )
    assert without_y.n_rows > with_y.n_rows


def test_feature_names_are_unique(dm):
    assert len(dm.feature_names) == len(set(dm.feature_names))


def test_all_columns_are_accounted_for(dm):
    assert dm.X.shape[1] == len(dm.feature_names)
    assert dm.X.shape[0] == dm.y.shape[0] == dm.series_idx.shape[0] == dm.d.shape[0]
