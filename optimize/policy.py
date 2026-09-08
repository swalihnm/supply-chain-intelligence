"""(s, S) periodic-review replenishment policy, derived from forecast quantiles.

THE POLICY
----------
Inventory is reviewed every T days (default 7 - grocery stores order weekly).
At a review, if the inventory position has fallen to or below the reorder point
s, an order is placed to bring it up to the order-up-to level S. Orders arrive
after a random lead time L.

    s = quantile_SL( demand over L )        protects until an order placed now arrives
    S = quantile_SL( demand over T + L )    protects until the order AFTER that arrives

TWO WINDOWS, NOT ONE
--------------------
S covers T + L rather than L. This is the mistake that quietly starves periodic
review systems: once you order, you wait L days for delivery AND another T days
before you can order again, so the exposure between replenishment opportunities
is T + L. Sizing S on L alone under-stocks by roughly a review period of demand
on every cycle.

SAFETY STOCK COMES FROM THE EMPIRICAL DISTRIBUTION
--------------------------------------------------
    safety_stock = s - E[demand over L]

It is a difference between a simulated quantile and a simulated mean, not
z * sigma * sqrt(L). No normality is assumed anywhere - see
optimize/lead_time_demand.py for why that matters when 72.6% of series are
intermittent.

INVENTORY POSITION, NOT ON-HAND
-------------------------------
The review compares s against inventory POSITION = on hand + already on order.
Comparing against on-hand alone would re-order every review while stock is in
transit, which produces a sawtooth of duplicate orders and wildly overstates
both holding cost and order count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimize.lead_time_demand import LeadTimeDemand, simulate
from sci.log import get_logger

log = get_logger(__name__)

DEFAULT_REVIEW_PERIOD = 7
DEFAULT_SERVICE_LEVEL = 0.95


@dataclass
class Policy:
    """Per-series replenishment parameters, all in units."""

    reorder_point: np.ndarray  # s
    order_up_to: np.ndarray  # S
    safety_stock: np.ndarray
    expected_lt_demand: np.ndarray  # E[demand over L]
    expected_cycle_demand: np.ndarray  # E[demand over T + L]
    service_level: float
    review_period: int
    case_pack: np.ndarray
    min_order_qty: np.ndarray

    def n_series(self) -> int:
        return self.reorder_point.shape[0]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "reorder_point": self.reorder_point,
            "order_up_to": self.order_up_to,
            "safety_stock": self.safety_stock,
            "expected_lt_demand": self.expected_lt_demand,
            "expected_cycle_demand": self.expected_cycle_demand,
        }


def default_case_packs(price: np.ndarray, seed: int = 0) -> np.ndarray:
    """Assumed case pack sizes. M5 has no pack data, so these are invented.

    Cheap fast-moving goods ship in bigger cases than expensive ones, so the
    pack is tied to price rather than assigned at random - a $1 item in packs of
    12, a $20 item in singles. Stated as an assumption because it is one; its
    only real effect is to coarsen order quantities, which raises average
    inventory slightly and reduces the order count.
    """
    packs = np.ones_like(price, dtype=np.float32)
    packs[price < 2.0] = 12.0
    packs[(price >= 2.0) & (price < 5.0)] = 6.0
    packs[(price >= 5.0) & (price < 10.0)] = 4.0
    packs[(price >= 10.0) & (price < 20.0)] = 2.0
    return packs


def build(
    history: np.ndarray,
    forecast_daily_mean: np.ndarray,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    review_period: int = DEFAULT_REVIEW_PERIOD,
    case_pack: np.ndarray | None = None,
    min_order_qty: np.ndarray | None = None,
    n_sims: int = 2000,
    seed: int = 0,
) -> Policy:
    """Derive (s, S) for every series at one service level."""
    n_series = history.shape[0]

    # Two simulations: one over L alone (for s), one over T + L (for S).
    lt_only: LeadTimeDemand = simulate(
        history,
        forecast_daily_mean,
        review_period=review_period,
        n_sims=n_sims,
        seed=seed,
        cover_review_period=False,
    )
    cycle: LeadTimeDemand = simulate(
        history,
        forecast_daily_mean,
        review_period=review_period,
        n_sims=n_sims,
        seed=seed + 1,
        cover_review_period=True,
    )

    s = lt_only.quantile(service_level)
    big_s = cycle.quantile(service_level)

    e_lt = lt_only.mean()
    e_cycle = cycle.mean()

    # S must exceed s, or a review would order a negative quantity.
    big_s = np.maximum(big_s, s + 1e-6)

    case_pack = case_pack if case_pack is not None else np.ones(n_series, dtype=np.float32)
    min_order_qty = (
        min_order_qty if min_order_qty is not None else np.zeros(n_series, dtype=np.float32)
    )

    return Policy(
        reorder_point=np.ceil(s).astype(np.float32),
        order_up_to=np.ceil(big_s).astype(np.float32),
        safety_stock=np.clip(np.ceil(s) - e_lt, 0, None).astype(np.float32),
        expected_lt_demand=e_lt.astype(np.float32),
        expected_cycle_demand=e_cycle.astype(np.float32),
        service_level=service_level,
        review_period=review_period,
        case_pack=case_pack.astype(np.float32),
        min_order_qty=min_order_qty.astype(np.float32),
    )


def build_normal_approx(
    history: np.ndarray,
    forecast_daily_mean: np.ndarray,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    review_period: int = DEFAULT_REVIEW_PERIOD,
    mean_lead_time: float = 5.0,
    case_pack: np.ndarray | None = None,
    min_order_qty: np.ndarray | None = None,
) -> Policy:
    """The textbook policy: normal demand, z * sigma * sqrt(T + L) safety stock.

    Implemented as the COMPARISON ARM, never as the recommendation. Keeping it
    lets the results quantify what assuming normality costs on this data instead
    of merely asserting that it would be wrong.

    Its known failure mode on intermittent series is visible in the arithmetic:
    for a series selling 0.05 units/day with sigma near 0.3, the normal formula
    puts the 95th percentile at roughly mu*w + 1.64*sigma*sqrt(w), a smooth
    continuous quantity, when the true distribution is a spike at zero with
    occasional jumps of 1 or more. It systematically misses the shape.
    """
    from scipy.stats import norm

    n_series = history.shape[0]
    z = float(norm.ppf(service_level))
    window = review_period + mean_lead_time

    with np.errstate(invalid="ignore"):
        sigma = np.nan_to_num(np.nanstd(history, axis=1), nan=0.0)

    e_lt = forecast_daily_mean * mean_lead_time
    e_cycle = forecast_daily_mean * window

    s = np.clip(e_lt + z * sigma * np.sqrt(mean_lead_time), 0, None)
    big_s = np.clip(e_cycle + z * sigma * np.sqrt(window), 0, None)
    big_s = np.maximum(big_s, s + 1e-6)

    case_pack = case_pack if case_pack is not None else np.ones(n_series, dtype=np.float32)
    min_order_qty = (
        min_order_qty if min_order_qty is not None else np.zeros(n_series, dtype=np.float32)
    )

    return Policy(
        reorder_point=np.ceil(s).astype(np.float32),
        order_up_to=np.ceil(big_s).astype(np.float32),
        safety_stock=np.clip(np.ceil(s) - e_lt, 0, None).astype(np.float32),
        expected_lt_demand=e_lt.astype(np.float32),
        expected_cycle_demand=e_cycle.astype(np.float32),
        service_level=service_level,
        review_period=review_period,
        case_pack=case_pack.astype(np.float32),
        min_order_qty=min_order_qty.astype(np.float32),
    )


def build_frontier(
    history: np.ndarray,
    forecast_daily_mean: np.ndarray,
    service_levels: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 0.975, 0.99),
    **kwargs,
) -> dict[float, Policy]:
    """One policy per service level - the input to the service-level frontier.

    The simulation is the expensive part and it does not depend on the service
    level, so in principle it could be shared. It is not, because `build` also
    varies the seed per call and re-running keeps each policy independent; the
    cost is a few seconds and the benefit is that no two points on the frontier
    share simulation noise.
    """
    out = {}
    for sl in service_levels:
        log.info("  policy at service level %.3f", sl)
        out[sl] = build(history, forecast_daily_mean, service_level=sl, **kwargs)
    return out


def round_to_pack(qty: np.ndarray, case_pack: np.ndarray, min_order_qty: np.ndarray) -> np.ndarray:
    """Round an order quantity up to a whole case, respecting any MOQ.

    Rounding UP rather than to nearest: a supplier will not split a case, and
    rounding down would systematically undershoot the order-up-to level and
    silently degrade the service level the policy was built to hit.
    """
    q = np.maximum(qty, 0.0)
    packs = np.ceil(np.divide(q, case_pack, out=np.zeros_like(q), where=case_pack > 0))
    out = packs * case_pack
    needs_order = q > 0
    out[needs_order] = np.maximum(out[needs_order], min_order_qty[needs_order])
    return out
