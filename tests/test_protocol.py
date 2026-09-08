"""The backtest protocol is a contract. These tests are what keep it one."""

from __future__ import annotations

from itertools import pairwise

from evaluation.protocol import (
    DEV_END_D,
    HORIZON,
    LAST_D,
    N_FOLDS,
    TEST_END_D,
    TEST_START_D,
    all_folds,
    cv_folds,
    test_fold,
)


def test_fold_count():
    assert len(cv_folds()) == N_FOLDS


def test_every_fold_has_the_declared_horizon():
    for f in all_folds(include_test=True):
        assert f.horizon == HORIZON, f"{f.name} has horizon {f.horizon}"


def test_training_never_overlaps_its_own_validation():
    """The single most important property. A fold that trains on its own
    validation window produces a meaningless score that looks excellent."""
    for f in all_folds(include_test=True):
        assert f.train_end_d < f.valid_start_d, f"{f.name} trains into its validation window"
        assert f.train_end_d == f.valid_start_d - 1, f"{f.name} has an unexplained gap"


def test_validation_windows_are_disjoint():
    """No observation may be scored twice across folds."""
    seen: set[int] = set()
    for f in all_folds(include_test=True):
        days = set(range(f.valid_start_d, f.valid_end_d + 1))
        assert not (days & seen), f"{f.name} overlaps an earlier fold"
        seen |= days


def test_cv_folds_never_touch_the_locked_test_set():
    """Development must not see the hold-out, in training OR validation."""
    for f in cv_folds():
        assert f.valid_end_d <= DEV_END_D, f"{f.name} validates inside the test window"
        assert f.train_end_d < TEST_START_D, f"{f.name} trains inside the test window"


def test_test_fold_is_the_final_window():
    t = test_fold()
    assert t.valid_start_d == TEST_START_D
    assert t.valid_end_d == TEST_END_D == LAST_D
    assert t.train_end_d == DEV_END_D


def test_folds_are_ordered_oldest_first():
    folds = cv_folds()
    starts = [f.valid_start_d for f in folds]
    assert starts == sorted(starts)


def test_later_folds_train_on_more_data():
    """Rolling origin: each fold should see everything the previous one did."""
    folds = cv_folds()
    for a, b in pairwise(folds):
        assert b.train_end_d > a.train_end_d
