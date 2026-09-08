"""Counterfactual simulation: what the optimized policy would have cost.

This is the deliverable. Everything before it is machinery.

HOW THE BASELINE IS RECONSTRUCTED - READ THIS BEFORE THE RESULT
---------------------------------------------------------------
M5 contains no inventory data. There are no stock levels, no purchase orders, no
receipts. So the "observed baseline" is NOT observed - it is reconstructed, and
the reconstruction is an assumption that the headline number rests on entirely.

The reconstruction: a (s, S) policy of the same structure, but driven by what a
conventional replenishment system actually does -

    * a 28-day moving-average forecast, not a learned model
    * safety stock from the textbook normal approximation, z*sigma*sqrt(T+L)

and with its service-level parameter CALIBRATED so that the simulated fill rate
reproduces the fill rate implied by the stockout proxy on real data. That
calibration is what ties the counterfactual to observed behaviour rather than to
a strawman: the baseline is tuned to fail as often as the real stores failed.

THE DECOMPOSITION
-----------------
Reporting one number would hide where the gain comes from, so three policies run
on identical demand:

    baseline        moving-average forecast + normal safety stock
    forecast_only   global model forecast   + normal safety stock
    optimized       global model forecast   + empirical bootstrap safety stock

baseline -> forecast_only isolates the value of the FORECAST.
forecast_only -> optimized isolates the value of dropping NORMALITY.

If almost all the gain sits in one step, that is worth knowing and worth saying.

WARM-UP
-------
The simulation runs 28 warm-up days before the 28 scored days and reports only
the scored window. Starting every policy at its own order-up-to level would hand
an advantage to whichever policy sets S highest, and the warm-up lets each
settle into its own steady state first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimize.cost_model import CostBreakdown, CostParams, evaluate, service_level
from optimize.policy import Policy, round_to_pack
from sci.log import get_logger

log = get_logger(__name__)

WARMUP_DAYS = 28


@dataclass
class SimResult:
    name: str
    on_hand: np.ndarray  # (n_series, n_scored_days)
    unmet: np.ndarray
    demand: np.ndarray
    n_orders: np.ndarray  # (n_series,)
    order_units: np.ndarray
    cost: CostBreakdown
    service: dict[str, float]

    def summary(self) -> dict[str, float]:
        out = {
            "policy": self.name,
            "avg_on_hand_units": float(np.nanmean(self.on_hand)),
            "total_on_hand_unit_days": float(np.nansum(self.on_hand)),
            "orders_placed": float(np.nansum(self.n_orders)),
            "units_ordered": float(np.nansum(self.order_units)),
        }
        out.update(self.service)
        out.update(self.cost.as_dict())
        return out


def simulate_policy(
    demand: np.ndarray,
    policy: Policy,
    price: np.ndarray,
    params: CostParams,
    name: str,
    lead_time_min: int = 3,
    lead_time_max: int = 7,
    warmup: int = WARMUP_DAYS,
    seed: int = 0,
) -> SimResult:
    """Run the (s, S) policy day by day over `demand`.

    demand : (n_series, warmup + n_scored) actual units demanded
    Only the last (n_days - warmup) columns are scored.
    """
    rng = np.random.default_rng(seed)
    n_series, n_days = demand.shape
    T = policy.review_period

    on_hand = policy.order_up_to.astype(np.float64).copy()
    pipeline = np.zeros((n_series, lead_time_max + 2), dtype=np.float64)

    oh_rec = np.zeros((n_series, n_days), dtype=np.float32)
    unmet_rec = np.zeros((n_series, n_days), dtype=np.float32)
    order_rec = np.zeros((n_series, n_days), dtype=np.float32)

    for t in range(n_days):
        # receipts scheduled for today
        on_hand += pipeline[:, 0]
        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = 0.0

        d = np.nan_to_num(demand[:, t], nan=0.0)
        served = np.minimum(on_hand, d)
        unmet_rec[:, t] = d - served
        on_hand -= served
        oh_rec[:, t] = on_hand

        # review day
        if t % T == 0:
            position = on_hand + pipeline.sum(axis=1)
            trigger = position <= policy.reorder_point
            raw_qty = np.where(trigger, policy.order_up_to - position, 0.0)
            qty = round_to_pack(raw_qty, policy.case_pack, policy.min_order_qty)
            qty = np.where(trigger, qty, 0.0)

            lt = rng.integers(lead_time_min, lead_time_max + 1, size=n_series)
            # Scatter each order into its own arrival slot.
            for lag in range(lead_time_min, lead_time_max + 1):
                m = (lt == lag) & (qty > 0)
                if m.any():
                    pipeline[m, lag] += qty[m]
            order_rec[:, t] = qty

    sl = slice(warmup, n_days)
    oh, un, dm, orders = oh_rec[:, sl], unmet_rec[:, sl], demand[:, sl], order_rec[:, sl]

    # A day the series was not on sale is not a service failure.
    mask = np.isnan(demand[:, sl])
    un = np.where(mask, np.nan, un)
    oh = np.where(mask, np.nan, oh)

    cost = evaluate(
        on_hand=oh,
        unmet=un,
        n_orders=(orders > 0).sum(axis=1),
        price=price[:, sl],
        params=params,
    )
    return SimResult(
        name=name,
        on_hand=oh,
        unmet=un,
        demand=dm,
        n_orders=(orders > 0).sum(axis=1),
        order_units=orders.sum(axis=1),
        cost=cost,
        service=service_level(dm, un),
    )


def calibrate_policy_to_fill(
    demand: np.ndarray,
    build_policy_fn,
    price: np.ndarray,
    params: CostParams,
    target_fill_rate: float,
    lo: float = 0.50,
    hi: float = 0.9999,
    tol: float = 5e-4,
    max_iter: int = 14,
    seed: int = 0,
    label: str = "policy",
) -> tuple[float, float]:
    """Find the service-level PARAMETER whose simulated FILL RATE hits a target.

    Used for every policy, not just the baseline, and that is the point.

    A service-level parameter and an achieved fill rate are different
    quantities. The parameter sets a quantile of lead-time demand, which
    controls the probability of a shortfall in a replenishment CYCLE; the fill
    rate is the fraction of demanded UNITS served. The second is always lower,
    and the gap between them differs by method - the normal approximation and
    the empirical bootstrap do not convert parameter to fill rate the same way
    at all on intermittent demand.

    So comparing "baseline at parameter 0.999" against "optimized at parameter
    0.95" compares two policies operating at different service levels, and any
    cost difference is mostly just that. Calibrating EVERY policy to the same
    achieved fill rate is what makes the cost comparison mean something:
    same service, different cost.

    BISECTION, NOT A GRID
    ---------------------
    Fill rate is monotone increasing in the parameter, so bisection converges to
    a tight match. A coarse grid does not, and here that is not a rounding
    detail: a lost unit costs roughly 26x holding one for the whole window, so a
    residual 0.5-point fill gap swamps every real difference between methods. An
    earlier grid version left the optimized policy 0.56 points below the
    baseline and reported it as 9% more expensive - which was the service gap
    being priced, not the method.

    Returns (chosen_parameter, achieved_fill_rate).
    """

    def _fill_at(sl: float) -> float:
        pol = build_policy_fn(sl)
        res = simulate_policy(demand, pol, price, params, name=f"cal_{sl}", seed=seed)
        return res.service["fill_rate"]

    f_lo, f_hi = _fill_at(lo), _fill_at(hi)
    log.info(
        "  %s: bracket [%.4f, %.4f] -> fill [%.4f, %.4f], target %.4f",
        label,
        lo,
        hi,
        f_lo,
        f_hi,
        target_fill_rate,
    )

    # If the target lies outside what this policy family can reach, take the
    # closest endpoint and say so rather than silently returning a bad match.
    if target_fill_rate <= f_lo:
        log.warning("  %s cannot go below fill %.4f - clamping to parameter %.4f", label, f_lo, lo)
        return lo, f_lo
    if target_fill_rate >= f_hi:
        log.warning("  %s cannot reach fill %.4f - clamping to parameter %.4f", label, f_hi, hi)
        return hi, f_hi

    best, best_fill = hi, f_hi
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _fill_at(mid)
        if abs(f_mid - target_fill_rate) < abs(best_fill - target_fill_rate):
            best, best_fill = mid, f_mid
        log.info("  %s: iter %2d parameter %.5f -> fill %.5f", label, i + 1, mid, f_mid)
        if abs(f_mid - target_fill_rate) <= tol:
            break
        if f_mid < target_fill_rate:
            lo = mid
        else:
            hi = mid

    log.info(
        "  %s calibrated to parameter %.5f (fill %.5f, gap %+.5f)",
        label,
        best,
        best_fill,
        best_fill - target_fill_rate,
    )
    return best, best_fill


def bootstrap_cost_delta(
    result_a: SimResult,
    result_b: SimResult,
    price: np.ndarray,
    params: CostParams,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Confidence interval on the cost difference, resampling SERIES.

    Series are the independent unit here, not days: demand within a series is
    autocorrelated across days, so resampling days would treat correlated
    observations as independent and produce an interval that is far too narrow.

    Resampling series also matches the business question - "would this hold on a
    different assortment of SKUs" - rather than "would this hold on a different
    Tuesday".
    """
    rng = np.random.default_rng(seed)
    n_series = result_a.on_hand.shape[0]

    unit_cost = params.unit_cost(price)
    unit_margin = params.unit_margin(price)
    h = params.holding_rate_daily

    def per_series_cost(r: SimResult) -> np.ndarray:
        hold = np.nansum(r.on_hand * unit_cost, axis=1) * h
        short = np.nansum(r.unmet * unit_margin, axis=1) * params.stockout_penalty
        order = r.n_orders * params.order_cost
        return hold + short + order

    ca, cb = per_series_cost(result_a), per_series_cost(result_b)
    deltas = np.empty(n_boot)
    pct = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_series, n_series)
        a, b = ca[idx].sum(), cb[idx].sum()
        deltas[i] = b - a
        pct[i] = 100.0 * (b - a) / a if a > 0 else np.nan

    return {
        "cost_a": float(ca.sum()),
        "cost_b": float(cb.sum()),
        "delta": float(cb.sum() - ca.sum()),
        "delta_pct": float(100.0 * (cb.sum() - ca.sum()) / ca.sum()),
        "delta_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_ci_hi": float(np.percentile(deltas, 97.5)),
        "delta_pct_ci_lo": float(np.nanpercentile(pct, 2.5)),
        "delta_pct_ci_hi": float(np.nanpercentile(pct, 97.5)),
        "n_boot": n_boot,
    }
