"""Expected cost per (series, service level), from the FORECAST only.

WHY THIS MODULE EXISTS - THE OVERFITTING IT FIXES
-------------------------------------------------
The allocation step chooses a service level for each of 30,490 series. The first
version scored every candidate level by SIMULATING it on the realised test
demand and then picked the per-series minimum. That is selection on the outcome:
each series gets the level that happened to work best on the very days the
result is then reported over. It produced an "unconstrained optimum" of 77,554
against a 127,930 baseline - a 39% saving that is mostly hindsight.

Everything here is computed from the forecast-derived lead-time demand
distribution instead. No realised demand is touched. The chosen allocation is
then evaluated on realised demand exactly once, which is a genuine out-of-sample
counterfactual rather than a lookup of the answer.

WHY A UNIFORM SERVICE LEVEL IS THE WRONG SHAPE ANYWAY
-----------------------------------------------------
The measured frontier falls monotonically: total cost keeps dropping as the
uniform target rises to 99.9%. That is not a bug, it is the arithmetic of the
cost model - over a ~12-day exposure window a held unit costs about $0.025 while
a lost unit costs about $1.50, so the newsvendor critical ratio sits near 0.98
and there is no interior optimum below it.

Which means the interesting question was never "what uniform service level?" but
"which series deserve the stock?" - and that has a different answer for a
fast-moving A-item and a lumpy C-item. Allocating per series is the decision the
project is actually about.

THE APPROXIMATIONS, STATED
--------------------------
Average on-hand over a review cycle is approximated as

    I_avg ~ S - E[D_(T+L)] + mu*T/2

- the order-up-to level, less what is consumed over the protection window, plus
half a review period of cycle stock. This is the standard periodic-review
approximation. It ignores the case-pack rounding (which raises real inventory
slightly) and assumes every review triggers an order (which overstates the order
count for slow movers). Both push in the same direction for every candidate
level, so the RANKING across levels - which is all the allocation uses - is far
more robust than the absolute numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimize.cost_model import CostParams
from optimize.lead_time_demand import LEAD_TIME_MAX, LEAD_TIME_MIN, simulate
from sci.log import get_logger

log = get_logger(__name__)


@dataclass
class LevelCosts:
    """Expected 28-day cost and footprint for every series at one service level."""

    service_level: float
    expected_cost: np.ndarray
    holding: np.ndarray
    shortfall: np.ndarray
    ordering: np.ndarray
    inventory_units: np.ndarray  # average units on hand - the shelf-space footprint
    capital: np.ndarray  # average dollars tied up
    reorder_point: np.ndarray
    order_up_to: np.ndarray


def grid(
    history: np.ndarray,
    forecast_daily_mean: np.ndarray,
    price: np.ndarray,
    params: CostParams,
    service_levels: tuple[float, ...],
    review_period: int = 7,
    horizon_days: int = 28,
    n_sims: int = 1500,
    seed: int = 0,
) -> list[LevelCosts]:
    """Expected cost for every (series, service level) pair.

    The two lead-time simulations are run ONCE and reused across all levels -
    the distribution does not depend on the service level, only the quantile
    read off it does.
    """
    lt_only = simulate(
        history,
        forecast_daily_mean,
        review_period=review_period,
        n_sims=n_sims,
        seed=seed,
        cover_review_period=False,
    )
    cycle = simulate(
        history,
        forecast_daily_mean,
        review_period=review_period,
        n_sims=n_sims,
        seed=seed + 1,
        cover_review_period=True,
    )

    mu = np.clip(forecast_daily_mean, 0, None)
    e_cycle_demand = cycle.mean()
    cycles = horizon_days / review_period

    unit_cost = params.unit_cost(price)
    unit_margin = params.unit_margin(price)

    out: list[LevelCosts] = []
    for q in service_levels:
        s = np.ceil(lt_only.quantile(q))
        big_s = np.maximum(np.ceil(cycle.quantile(q)), s + 1.0)

        # Average on-hand over a cycle (see the module docstring).
        i_avg = np.clip(big_s - e_cycle_demand + mu * review_period / 2.0, 0.0, None)

        shortfall_units = cycle.expected_shortfall(big_s)

        holding = params.holding_rate_daily * horizon_days * unit_cost * i_avg
        shortfall = params.stockout_penalty * unit_margin * shortfall_units * cycles
        ordering = np.full_like(holding, params.order_cost * cycles)

        out.append(
            LevelCosts(
                service_level=q,
                expected_cost=holding + shortfall + ordering,
                holding=holding,
                shortfall=shortfall,
                ordering=ordering,
                inventory_units=i_avg,
                capital=i_avg * unit_cost,
                reorder_point=s,
                order_up_to=big_s,
            )
        )
        log.info(
            "  expected cost at sl=%.4f: total %.0f, mean on-hand %.2f units",
            q,
            float(holding.sum() + shortfall.sum() + ordering.sum()),
            float(i_avg.mean()),
        )

    return out


def argmin_levels(costs: list[LevelCosts]) -> tuple[np.ndarray, np.ndarray]:
    """Per-series cost-minimising service level, chosen on EXPECTED cost.

    Returns (chosen_service_levels, expected_cost_at_choice).
    """
    stack = np.stack([c.expected_cost for c in costs], axis=1)
    levels = np.array([c.service_level for c in costs])
    k = np.nanargmin(stack, axis=1)
    return levels[k], stack[np.arange(stack.shape[0]), k]


def mean_lead_time() -> float:
    return (LEAD_TIME_MIN + LEAD_TIME_MAX) / 2.0
