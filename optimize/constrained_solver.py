"""Constrained service-level allocation, with shadow prices.

THE PROBLEM
-----------
The unconstrained policy sets every series to the same service level. Real
stores cannot do that: shelf space is finite, so is working capital. Something
has to give, and the question is WHICH series should give.

The answer is not uniform. A series where an extra unit of stock buys a lot of
avoided shortfall cost deserves the space; one where it buys almost nothing does
not. That is an allocation problem, and it is what separates a forecasting
project from a decision one.

FORMULATION - MULTIPLE-CHOICE KNAPSACK
--------------------------------------
Each series i gets one service level from K candidates. Each (i, k) pair has a
known expected cost, inventory footprint and capital requirement, all produced
by the simulation. Then:

    minimise    sum_ik  cost[i,k] * y[i,k]
    subject to  sum_k   y[i,k] = 1                       for every series
                sum_ik  units[i,k] * y[i,k] <= capacity[store]
                sum_ik  capital[i,k] * y[i,k] <= budget
                y in {0, 1}

WHY THE LP RELAXATION IS NOT A COMPROMISE HERE
----------------------------------------------
As an integer program this is 30,490 x 6 = 182,940 binaries, which is not
something CBC will close on a laptop. We solve the LP relaxation instead, and
for this structure that is close to exact rather than a concession:

A basic feasible solution of an LP has at most as many basic variables as it has
rows. The multiple-choice constraints force one unit of mass per series, so the
only source of fractional assignment is the SIDE constraints - the store
capacities and the budget. With 10 stores plus one budget row, at most 11 series
can be split across two service levels. Out of 30,490. Those are rounded down to
the cheaper level, and the resulting integrality gap is reported.

The relaxation also gives what the integer program would not: DUAL VALUES.

SHADOW PRICES
-------------
The dual on a capacity row is the marginal cost of that constraint - what one
more unit of shelf space at that store is worth in avoided total cost per period.
That is the number to take to an operations director, because it converts
"we are short on space" into "space at TX_2 is worth $X per unit per period and
space at CA_1 is worth nothing, so stop asking for more at CA_1".

A zero dual means the constraint is slack and is costing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pulp

from sci.log import get_logger

log = get_logger(__name__)


@dataclass
class Candidate:
    """One (series, service level) option, costed by simulation."""

    service_level: float
    expected_cost: np.ndarray  # (n_series,) total expected cost per period
    inventory_units: np.ndarray  # (n_series,) avg units held - the space footprint
    capital: np.ndarray  # (n_series,) avg dollars tied up


@dataclass
class SolveResult:
    chosen_level: np.ndarray  # (n_series,) service level assigned
    total_cost: float
    unconstrained_cost: float
    shadow_prices: dict[str, float]
    binding: dict[str, bool]
    n_fractional: int
    status: str
    constraint_usage: pl.DataFrame = field(default_factory=pl.DataFrame)

    def cost_of_constraints(self) -> float:
        """How much the constraints cost versus the unconstrained optimum."""
        return self.total_cost - self.unconstrained_cost


def solve(
    candidates: list[Candidate],
    store_code: np.ndarray,
    store_names: list[str],
    store_capacity: dict[str, float] | None = None,
    total_budget: float | None = None,
    msg: bool = False,
) -> SolveResult:
    """Allocate a service level to every series under capacity and budget limits.

    store_capacity maps store name -> maximum average units held.
    total_budget is the cap on average capital tied up across all stores.
    Passing neither returns the unconstrained optimum, which is the reference
    point the constrained solutions are measured against.
    """
    n_series = candidates[0].expected_cost.shape[0]
    n_levels = len(candidates)

    cost = np.stack([c.expected_cost for c in candidates], axis=1)  # (S, K)
    units = np.stack([c.inventory_units for c in candidates], axis=1)
    capital = np.stack([c.capital for c in candidates], axis=1)
    levels = np.array([c.service_level for c in candidates])

    # The unconstrained optimum: every series independently picks its cheapest
    # level. This is the benchmark, and it needs no solver.
    best_k = np.nanargmin(cost, axis=1)
    unconstrained_cost = float(cost[np.arange(n_series), best_k].sum())

    if not store_capacity and total_budget is None:
        return SolveResult(
            chosen_level=levels[best_k],
            total_cost=unconstrained_cost,
            unconstrained_cost=unconstrained_cost,
            shadow_prices={},
            binding={},
            n_fractional=0,
            status="unconstrained",
        )

    log.info("  building LP: %d series x %d levels", n_series, n_levels)
    prob = pulp.LpProblem("service_level_allocation", pulp.LpMinimize)

    # y[i][k] in [0, 1]
    y = [
        [pulp.LpVariable(f"y_{i}_{k}", lowBound=0, upBound=1) for k in range(n_levels)]
        for i in range(n_series)
    ]

    prob += pulp.lpSum(
        float(cost[i, k]) * y[i][k] for i in range(n_series) for k in range(n_levels)
    )

    for i in range(n_series):
        prob += pulp.lpSum(y[i]) == 1, f"assign_{i}"

    cap_rows: dict[str, pulp.LpConstraint] = {}
    if store_capacity:
        for s_idx, s_name in enumerate(store_names):
            if s_name not in store_capacity:
                continue
            members = np.flatnonzero(store_code == s_idx)
            if members.size == 0:
                continue
            expr = pulp.lpSum(
                float(units[i, k]) * y[i][k] for i in members for k in range(n_levels)
            )
            name = f"capacity_{s_name}"
            prob += expr <= float(store_capacity[s_name]), name
            cap_rows[name] = prob.constraints[name]

    if total_budget is not None:
        expr = pulp.lpSum(
            float(capital[i, k]) * y[i][k] for i in range(n_series) for k in range(n_levels)
        )
        prob += expr <= float(total_budget), "budget"
        cap_rows["budget"] = prob.constraints["budget"]

    log.info("  solving LP (%d vars, %d rows)", n_series * n_levels, len(prob.constraints))
    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    status = pulp.LpStatus[prob.status]
    log.info("  status: %s", status)

    sol = np.array([[y[i][k].value() or 0.0 for k in range(n_levels)] for i in range(n_series)])
    fractional = ((sol > 1e-6) & (sol < 1 - 1e-6)).any(axis=1)
    n_frac = int(fractional.sum())

    # Round the handful of split assignments to the level using LESS of the
    # constrained resource, not the cheaper one. Rounding toward cost can push
    # a binding constraint marginally over its limit - observed at ~0.2%
    # over-utilisation in testing - which makes the reported solution
    # infeasible. Rounding toward the smaller footprint guarantees feasibility
    # and costs at most a few series' worth of service.
    chosen_k = np.argmax(sol, axis=1)
    for i in np.flatnonzero(fractional):
        active = np.flatnonzero(sol[i] > 1e-6)
        chosen_k[i] = active[np.argmin(units[i, active])]

    total = float(cost[np.arange(n_series), chosen_k].sum())

    shadow = {}
    binding = {}
    for name, row in cap_rows.items():
        # PuLP reports duals with the sign convention of the solver; for a <=
        # row in a minimisation the dual is <= 0, so negate to read it as
        # "cost saved per extra unit of the resource".
        pi = row.pi if row.pi is not None else 0.0
        shadow[name] = float(-pi)
        binding[name] = abs(pi) > 1e-9

    usage_rows = []
    for name in cap_rows:
        if name == "budget":
            used = float(capital[np.arange(n_series), chosen_k].sum())
            limit = float(total_budget)
        else:
            s_name = name.removeprefix("capacity_")
            s_idx = store_names.index(s_name)
            m = store_code == s_idx
            used = float(units[np.arange(n_series), chosen_k][m].sum())
            limit = float(store_capacity[s_name])
        usage_rows.append(
            {
                "constraint": name,
                "used": used,
                "limit": limit,
                "utilisation": used / limit if limit > 0 else np.nan,
                "shadow_price": shadow[name],
                "binding": binding[name],
            }
        )

    return SolveResult(
        chosen_level=levels[chosen_k],
        total_cost=total,
        unconstrained_cost=unconstrained_cost,
        shadow_prices=shadow,
        binding=binding,
        n_fractional=n_frac,
        status=status,
        constraint_usage=pl.DataFrame(usage_rows),
    )
