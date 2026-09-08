"""Cost model, lead-time demand, and policy behaviour."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from optimize import constrained_solver, lead_time_demand, policy
from optimize.cost_model import CostParams, evaluate, sensitivity_grid, service_level

# --- cost model -----------------------------------------------------------


def test_unit_cost_and_margin_split_the_price():
    p = CostParams(gross_margin=0.25)
    price = np.array([10.0])
    assert p.unit_cost(price)[0] == pytest.approx(7.5)
    assert p.unit_margin(price)[0] == pytest.approx(2.5)
    assert p.unit_cost(price)[0] + p.unit_margin(price)[0] == pytest.approx(price[0])


def test_cost_components_are_additive():
    b = evaluate(
        on_hand=np.array([[10.0]]),
        unmet=np.array([[2.0]]),
        n_orders=np.array([3.0]),
        price=np.array([[10.0]]),
        params=CostParams(),
    )
    assert b.total == pytest.approx(b.holding + b.stockout + b.ordering)
    assert b.ordering == pytest.approx(3 * 0.50)
    assert b.stockout == pytest.approx(2 * 2.5 * 1.5)


def test_holding_cost_scales_with_the_rate():
    kw = dict(
        on_hand=np.array([[100.0]]),
        unmet=np.array([[0.0]]),
        n_orders=np.array([0.0]),
        price=np.array([[10.0]]),
    )
    a = evaluate(**kw, params=CostParams(holding_rate_annual=0.25))
    b = evaluate(**kw, params=CostParams(holding_rate_annual=0.50))
    assert b.holding == pytest.approx(2 * a.holding)


def test_fill_rate_and_cycle_service_are_different_quantities():
    """Cycle service counts DAYS without shortfall; fill rate counts UNITS.

    Neither is reliably larger - it depends on the shape of the failures. One
    big shortfall on one day gives a HIGH fill rate and a LOW cycle service;
    small shortfalls spread over every day give the opposite. Both are reported
    everywhere so a service claim cannot quietly pick the flattering one.
    """
    demand = np.array([[10.0, 10.0, 10.0, 10.0]])

    # Rare but large: one day loses half its demand.
    rare_big = service_level(demand, np.array([[0.0, 0.0, 0.0, 5.0]]))
    assert rare_big["fill_rate"] == pytest.approx(1 - 5 / 40)  # 0.875
    assert rare_big["cycle_service"] == pytest.approx(0.75)
    assert rare_big["cycle_service"] < rare_big["fill_rate"]

    # Frequent but tiny: every day loses one unit.
    frequent_small = service_level(demand, np.array([[1.0, 1.0, 1.0, 1.0]]))
    assert frequent_small["fill_rate"] == pytest.approx(1 - 4 / 40)  # 0.90
    assert frequent_small["cycle_service"] == pytest.approx(0.0)
    assert frequent_small["cycle_service"] < frequent_small["fill_rate"]

    # And the reverse ordering is reachable too.
    tiny_on_one_day = service_level(demand, np.array([[0.0, 0.0, 0.0, 0.1]]))
    assert tiny_on_one_day["cycle_service"] == pytest.approx(0.75)
    assert tiny_on_one_day["fill_rate"] > tiny_on_one_day["cycle_service"]


def test_sensitivity_grid_is_a_full_factorial():
    g = sensitivity_grid()
    assert len(g) == 3 * 3 * 3 * 1
    assert len({p.label() for p in g}) == len(g)


# --- lead time demand -----------------------------------------------------


@pytest.fixture
def history():
    rng = np.random.default_rng(0)
    # 3 series x 365 days: dense, intermittent, and thin-history.
    h = np.zeros((3, 365), dtype=np.float32)
    h[0] = rng.poisson(5.0, 365)
    h[1] = rng.poisson(0.2, 365)
    h[2] = np.nan
    h[2, -20:] = rng.poisson(1.0, 20)
    return h


def test_lead_time_quantiles_are_monotone(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    ltd = lead_time_demand.simulate(history, fmean, n_sims=500, seed=0)
    qs = [ltd.quantile(q) for q in (0.5, 0.9, 0.95, 0.99)]
    for a, b in pairwise(qs):
        assert (b >= a - 1e-9).all(), "lead-time demand quantiles are not monotone"


def test_thin_history_falls_back_to_poisson_and_is_flagged(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    ltd = lead_time_demand.simulate(history, fmean, n_sims=300, seed=0)
    assert not ltd.used_bootstrap[2], "series with 20 valid days should not bootstrap"
    assert ltd.used_bootstrap[0] and ltd.used_bootstrap[1]


def test_lead_time_demand_is_never_negative(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    ltd = lead_time_demand.simulate(history, fmean, n_sims=300, seed=0)
    assert (ltd.samples >= 0).all()


def test_bootstrap_mean_tracks_the_forecast_level(history):
    """The scaling step should centre the draws on what the model expects."""
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    ltd = lead_time_demand.simulate(
        history, fmean, review_period=7, n_sims=2000, seed=0, cover_review_period=True
    )
    # window is 7 + U(3,7) -> mean 12 days
    expected = fmean[0] * 12
    assert ltd.mean()[0] == pytest.approx(expected, rel=0.2)


def test_censored_days_are_imputed_not_zero_filled():
    """A NaN inside the sampled history must not be resampled as zero demand.

    Zero-filling would reintroduce exactly the censoring bias the stockout proxy
    exists to remove, and would drag every safety stock down.
    """
    clean = np.full((1, 365), 4.0, dtype=np.float32)
    holed = clean.copy()
    holed[0, 100:160] = np.nan  # a long censored stretch

    fmean = np.array([4.0], dtype=np.float32)
    a = lead_time_demand.simulate(clean, fmean, n_sims=1500, seed=1).mean()[0]
    b = lead_time_demand.simulate(holed, fmean, n_sims=1500, seed=1).mean()[0]
    assert b == pytest.approx(a, rel=0.1), "censored days depressed lead-time demand"


# --- policy ---------------------------------------------------------------


def test_order_up_to_always_exceeds_the_reorder_point(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    p = policy.build(history, fmean, service_level=0.95, n_sims=500)
    assert (p.order_up_to > p.reorder_point).all()


def test_safety_stock_rises_with_service_level(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    low = policy.build(history, fmean, service_level=0.80, n_sims=1000, seed=3)
    high = policy.build(history, fmean, service_level=0.99, n_sims=1000, seed=3)
    assert high.safety_stock[0] > low.safety_stock[0]
    assert (high.reorder_point >= low.reorder_point).all()


def test_safety_stock_is_never_negative(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    p = policy.build(history, fmean, service_level=0.5, n_sims=400)
    assert (p.safety_stock >= 0).all()


def test_round_to_pack_rounds_up_and_respects_moq():
    qty = np.array([0.0, 1.0, 13.0])
    pack = np.array([12.0, 12.0, 12.0])
    moq = np.array([0.0, 0.0, 24.0])
    out = policy.round_to_pack(qty, pack, moq)
    assert out[0] == 0.0, "no order should stay no order"
    assert out[1] == 12.0, "must round UP to a whole case"
    assert out[2] == 24.0, "MOQ must win over the case rounding"


def test_default_case_packs_are_larger_for_cheap_items():
    packs = policy.default_case_packs(np.array([1.0, 3.0, 7.0, 15.0, 30.0]))
    assert list(packs) == [12.0, 6.0, 4.0, 2.0, 1.0]


def test_normal_approx_is_available_as_a_comparison_arm(history):
    fmean = np.array([5.0, 0.2, 1.0], dtype=np.float32)
    p = policy.build_normal_approx(history, fmean, service_level=0.95)
    assert (p.order_up_to > p.reorder_point).all()
    assert (p.safety_stock >= 0).all()


# --- constrained solver ---------------------------------------------------


def _candidates(n=40, k=4):
    rng = np.random.default_rng(0)
    out = []
    for i, sl in enumerate((0.85, 0.90, 0.95, 0.99)[:k]):
        units = rng.uniform(1, 10, n) * (1 + 0.4 * i)
        out.append(
            constrained_solver.Candidate(
                service_level=sl,
                expected_cost=rng.uniform(1, 5, n) + 0.3 * i,
                inventory_units=units,
                capital=units * 3.0,
            )
        )
    return out


def test_unconstrained_solve_needs_no_solver():
    cands = _candidates()
    code = np.zeros(40, dtype=np.int32)
    r = constrained_solver.solve(cands, code, ["S1"])
    assert r.status == "unconstrained"
    assert r.total_cost == pytest.approx(r.unconstrained_cost)


def test_constraints_never_reduce_cost_below_the_unconstrained_optimum():
    cands = _candidates()
    code = np.zeros(40, dtype=np.int32)
    units = np.stack([c.inventory_units for c in cands], axis=1)
    cost = np.stack([c.expected_cost for c in cands], axis=1)
    k = np.argmin(cost, axis=1)
    cap = {"S1": 0.7 * float(units[np.arange(40), k].sum())}

    r = constrained_solver.solve(cands, code, ["S1"], store_capacity=cap)
    assert r.status == "Optimal"
    assert r.total_cost >= r.unconstrained_cost - 1e-6
    assert r.cost_of_constraints() >= -1e-6


def test_rounding_keeps_the_solution_feasible():
    """Fractional assignments are rounded toward the SMALLER footprint, so a
    binding capacity row cannot end up over its limit."""
    cands = _candidates(n=60)
    code = np.zeros(60, dtype=np.int32)
    units = np.stack([c.inventory_units for c in cands], axis=1)
    cost = np.stack([c.expected_cost for c in cands], axis=1)
    k = np.argmin(cost, axis=1)
    limit = 0.7 * float(units[np.arange(60), k].sum())

    r = constrained_solver.solve(cands, code, ["S1"], store_capacity={"S1": limit})
    used = float(r.constraint_usage["used"][0])
    assert used <= limit + 1e-6, f"used {used} exceeds limit {limit}"


def test_shadow_price_is_non_negative_on_a_binding_row():
    cands = _candidates()
    code = np.zeros(40, dtype=np.int32)
    units = np.stack([c.inventory_units for c in cands], axis=1)
    cost = np.stack([c.expected_cost for c in cands], axis=1)
    k = np.argmin(cost, axis=1)
    cap = {"S1": 0.7 * float(units[np.arange(40), k].sum())}
    r = constrained_solver.solve(cands, code, ["S1"], store_capacity=cap)
    assert all(v >= -1e-9 for v in r.shadow_prices.values())
