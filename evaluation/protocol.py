"""The backtest protocol. Defined BEFORE any model is fitted, and frozen.

This module deliberately contains no modelling code and imports nothing that
fits anything. It exists so the evaluation rules cannot be quietly adjusted
after seeing a result - which is the single easiest way to fool yourself in a
forecasting project, and the thing an interviewer will probe hardest.

THE RULES
---------
1. Rolling-origin cross validation, 5 folds, 28-day horizon each.
2. Strict temporal separation. Training data for a fold ends the day before the
   validation window opens. No shuffling, ever, anywhere.
3. The final 28 days (d 1914-1941) are a LOCKED TEST SET. They are not touched
   during model development. They are scored exactly once, at the end, and
   whatever that number says is the number that gets reported.
4. Metrics: WRMSSE (the M5 official metric), MASE, and pinball loss.
   Pinball is not optional - the optimizer consumes quantiles, not point
   forecasts, so a model with a great WRMSSE and badly calibrated tails is
   useless downstream.
5. Quantile calibration is checked separately: does the 90th percentile actually
   get exceeded 10% of the time?

WHY 28 DAYS
-----------
It matches the M5 competition horizon, so WRMSSE is comparable to published
numbers, and it is a realistic replenishment planning horizon.

WHY THE FOLDS OVERLAP IN TRAINING BUT NOT VALIDATION
----------------------------------------------------
Each fold trains on everything up to its cutoff, so later folds see more data -
which is what actually happens in production. Validation windows are disjoint,
so no observation is scored twice.
"""

from __future__ import annotations

from dataclasses import dataclass

# M5 day index bounds present in core.panel.
FIRST_D = 1
LAST_D = 1941  # 2016-05-22

HORIZON = 28
N_FOLDS = 5

# The locked test set. Nothing may look at these days until the final run.
TEST_START_D = LAST_D - HORIZON + 1  # 1914
TEST_END_D = LAST_D  # 1941

# Everything available for development.
DEV_END_D = TEST_START_D - 1  # 1913

QUANTILES = (0.5, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class Fold:
    """One rolling-origin split.

    train_end_d is inclusive. valid_start_d is train_end_d + 1, so there is no
    gap and no overlap - the model forecasts the days immediately following
    everything it was allowed to see.
    """

    name: str
    train_end_d: int
    valid_start_d: int
    valid_end_d: int

    @property
    def horizon(self) -> int:
        return self.valid_end_d - self.valid_start_d + 1

    def __str__(self) -> str:
        return (
            f"{self.name}: train d<={self.train_end_d}, "
            f"valid d{self.valid_start_d}-{self.valid_end_d}"
        )


def cv_folds() -> list[Fold]:
    """The 5 development folds, oldest first.

    The last fold validates on d 1886-1913, immediately before the locked test
    set. Earlier folds step back one horizon at a time.
    """
    folds = []
    for i in range(N_FOLDS):
        # i = 0 is the oldest fold.
        steps_back = N_FOLDS - 1 - i
        valid_end = DEV_END_D - steps_back * HORIZON
        valid_start = valid_end - HORIZON + 1
        folds.append(
            Fold(
                name=f"cv{i + 1}",
                train_end_d=valid_start - 1,
                valid_start_d=valid_start,
                valid_end_d=valid_end,
            )
        )
    return folds


def test_fold() -> Fold:
    """The locked hold-out. Score it once."""
    return Fold(
        name="test",
        train_end_d=DEV_END_D,
        valid_start_d=TEST_START_D,
        valid_end_d=TEST_END_D,
    )


def all_folds(include_test: bool = False) -> list[Fold]:
    folds = cv_folds()
    if include_test:
        folds.append(test_fold())
    return folds


def describe() -> str:
    lines = [
        "Backtest protocol (frozen)",
        f"  panel spans d{FIRST_D}-{LAST_D}",
        f"  horizon {HORIZON} days, {N_FOLDS} rolling-origin folds",
        f"  development range: d{FIRST_D}-{DEV_END_D}",
        f"  LOCKED TEST: d{TEST_START_D}-{TEST_END_D} (scored once)",
        f"  quantiles: {QUANTILES}",
        "",
    ]
    lines += [f"  {f}" for f in cv_folds()]
    lines.append(f"  {test_fold()}")
    return "\n".join(lines)
