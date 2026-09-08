"""Metric correctness against hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import metrics


def test_perfect_forecast_scores_zero():
    actual = np.array([[1.0, 2.0, 3.0, 4.0]])
    train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    scale_sq = metrics.naive_scale_squared(train)
    assert metrics.rmsse(actual, actual, scale_sq)[0] == pytest.approx(0.0)


def test_naive_scale_matches_hand_calculation():
    # diffs are 1, 1, 1, 1 -> mean squared 1.0, mean absolute 1.0
    train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert metrics.naive_scale_squared(train)[0] == pytest.approx(1.0)
    assert metrics.naive_scale_absolute(train)[0] == pytest.approx(1.0)


def test_naive_scale_ignores_nan_days():
    """NaN marks a day outside availability - it must not become a huge diff."""
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    b = np.array([[1.0, 2.0, np.nan, 3.0, 4.0]])
    # b's finite diffs are 1, nan, nan, 1 -> nanmean of squares is 1.0
    assert metrics.naive_scale_squared(b)[0] == pytest.approx(1.0)
    assert metrics.naive_scale_squared(a)[0] == pytest.approx(1.0)


def test_flat_series_gives_undefined_scale_not_infinity():
    """A constant series has zero naive error; RMSSE is undefined, not inf."""
    train = np.array([[3.0, 3.0, 3.0, 3.0]])
    assert np.isnan(metrics.naive_scale_squared(train)[0])


def test_wrmsse_is_a_weighted_mean():
    per = np.array([1.0, 3.0])
    w = np.array([0.25, 0.75])
    assert metrics.wrmsse(per, w) == pytest.approx(0.25 * 1.0 + 0.75 * 3.0)


def test_wrmsse_renormalises_after_dropping_undefined_series():
    per = np.array([2.0, np.nan])
    w = np.array([0.5, 0.5])
    # The surviving series carries all the weight, so the answer is 2.0 not 1.0.
    assert metrics.wrmsse(per, w) == pytest.approx(2.0)


def test_m5_weights_are_revenue_shares_and_sum_to_one():
    train = np.array([[1.0, 1.0], [3.0, 3.0]])
    price = np.array([[1.0, 1.0], [1.0, 1.0]])
    w = metrics.m5_weights(train, price, last_n=2)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.75)


def test_pinball_penalises_asymmetrically():
    """At q=0.95, under-forecasting must cost far more than over-forecasting.

    That asymmetry is the whole reason the optimizer uses quantiles.
    """
    actual = np.array([[10.0]])
    under = metrics.pinball_loss(actual, np.array([[8.0]]), 0.95)[0]
    over = metrics.pinball_loss(actual, np.array([[12.0]]), 0.95)[0]
    assert under > over
    assert under / over == pytest.approx(0.95 / 0.05)


def test_pinball_at_median_is_symmetric():
    actual = np.array([[10.0]])
    under = metrics.pinball_loss(actual, np.array([[8.0]]), 0.5)[0]
    over = metrics.pinball_loss(actual, np.array([[12.0]]), 0.5)[0]
    assert under == pytest.approx(over)


def test_coverage_counts_actuals_at_or_below_the_quantile():
    actual = np.array([[1.0, 2.0, 3.0, 4.0]])
    pred = np.array([[2.0, 2.0, 2.0, 2.0]])
    assert metrics.coverage(actual, pred) == pytest.approx(0.5)


def test_coverage_skips_nan_cells():
    actual = np.array([[1.0, np.nan, 3.0]])
    pred = np.array([[2.0, 2.0, 2.0]])
    # Only cells 0 and 2 count: 1 <= 2 is True, 3 <= 2 is False.
    assert metrics.coverage(actual, pred) == pytest.approx(0.5)


def test_calibration_table_reports_the_gap():
    actual = np.array([[1.0, 2.0, 3.0, 4.0]])
    rows = metrics.calibration_table(actual, {0.5: np.array([[2.0] * 4])})
    assert rows[0]["gap"] == pytest.approx(0.0)
