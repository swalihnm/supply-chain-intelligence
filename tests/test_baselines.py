"""Baseline forecaster behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from models import baselines


@pytest.fixture
def simple():
    # Two series, 70 days. One dense, one intermittent.
    dense = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 10)
    sparse = np.zeros(70)
    sparse[::10] = 5.0
    return np.vstack([dense, sparse]).astype(np.float32)


@pytest.mark.parametrize("name", sorted(baselines.REGISTRY))
def test_shape_and_non_negativity(simple, name):
    pred = baselines.REGISTRY[name](simple, 28)
    assert pred.shape == (2, 28), name
    assert np.isfinite(pred).all(), f"{name} produced non-finite values"
    assert (pred >= 0).all(), f"{name} produced negative demand"


def test_zero_predicts_zero(simple):
    assert (baselines.zero(simple, 28) == 0).all()


def test_naive_repeats_the_last_value(simple):
    pred = baselines.naive(simple, 5)
    assert pred[0, 0] == pytest.approx(simple[0, -1])
    assert (pred[0] == pred[0, 0]).all()


def test_seasonal_naive_repeats_the_weekly_cycle(simple):
    pred = baselines.seasonal_naive(simple, 14, season=7)
    np.testing.assert_allclose(pred[0, :7], simple[0, -7:])
    np.testing.assert_allclose(pred[0, 7:14], simple[0, -7:])


def test_seasonal_naive_fills_gaps_with_recent_mean_not_zero():
    """A censored day in the last cycle must not propagate a false zero."""
    x = np.tile([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], 8).astype(np.float32)
    x[-1] = np.nan
    pred = baselines.seasonal_naive(x[None, :], 7, season=7)
    assert pred[0, -1] > 0, "gap became a zero instead of the recent mean"
    assert pred[0, -1] == pytest.approx(4.0, abs=0.5)


def test_moving_average_is_the_window_mean(simple):
    pred = baselines.moving_average(simple, 3, window=7)
    assert pred[0, 0] == pytest.approx(np.nanmean(simple[0, -7:]), rel=1e-5)


def test_croston_produces_a_flat_positive_rate(simple):
    pred = baselines.croston(simple, 10)
    assert (pred[:, 0][:, None] == pred).all(), "Croston should be flat over the horizon"
    assert pred[1, 0] > 0, "Croston found no demand on an intermittent series"


def test_croston_rate_is_near_the_intermittent_series_mean():
    """A series selling 5 units every 10 days has a rate near 0.5/day."""
    x = np.zeros((1, 200), dtype=np.float32)
    x[0, ::10] = 5.0
    rate = baselines.croston(x, 1)[0, 0]
    assert 0.3 < rate < 0.8, f"rate {rate} is far from the true 0.5"


def test_tsb_decays_toward_zero_for_a_dead_sku():
    """The reason TSB is in the ladder at all.

    Croston only updates on demand periods, so it keeps forecasting the last
    known rate forever after a SKU dies. TSB updates probability every period
    and should fall well below it.
    """
    x = np.zeros((1, 400), dtype=np.float32)
    x[0, :100:5] = 4.0  # sold regularly, then stopped dead at day 100

    croston_rate = baselines.croston(x, 1)[0, 0]
    tsb_rate = baselines.tsb(x, 1)[0, 0]
    assert tsb_rate < croston_rate, (
        f"TSB {tsb_rate:.4f} did not decay below Croston {croston_rate:.4f} on a dead SKU"
    )


def test_all_baselines_tolerate_an_all_nan_series():
    x = np.full((1, 60), np.nan, dtype=np.float32)
    for name, fn in baselines.REGISTRY.items():
        pred = fn(x, 7)
        assert np.isfinite(pred).all(), f"{name} produced NaN on an empty series"


def test_seasonal_croston_preserves_the_croston_level():
    """The weekly reshaping must change the shape, not the total."""
    x = np.tile([2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0], 30).astype(np.float32)[None, :]
    flat = baselines.croston(x, 28)[0].sum()
    shaped = baselines.seasonal_croston(x, 28)[0].sum()
    assert shaped == pytest.approx(flat, rel=0.05)
