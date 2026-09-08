"""The cost model. Every parameter is explicit, named, and sensitivity-tested.

    Total cost = holding + stockout + ordering

      holding  = h * average_on_hand_units * unit_cost
      stockout = p * expected_unmet_units  * unit_margin
      ordering = K * number_of_orders

NONE OF THESE PARAMETERS ARE IN THE DATA
----------------------------------------
M5 gives a selling price. It does not give cost of goods, holding cost, the true
penalty for a lost sale, or the cost of cutting a purchase order. Every one of
those is assumed here, and the assumptions drive the headline number directly:
double the stockout penalty and the optimizer buys more inventory and reports a
different saving.

So the defensible claim is never "this policy saves X". It is "this policy saves
X under these stated assumptions, and here is how X moves when they change".
`sensitivity_grid` exists to make that second half easy to produce, and
`optimize/` results are reported across it rather than at a single point.

THE DEFAULTS, AND WHY
---------------------
gross_margin = 0.25
    US grocery retail gross margin runs roughly 22-28%. Unit cost is therefore
    0.75 * price and unit margin 0.25 * price. This is the assumption the whole
    cost model is most sensitive to, because it sets both the value of held
    inventory and the value of a lost sale.

holding_rate = 0.25 / year
    The standard textbook figure for retail carrying cost: capital, warehousing,
    shrinkage, insurance, obsolescence. Converted to a daily rate.

stockout_penalty = 1.5
    A lost sale costs the lost margin (1.0) plus something for the customer who
    goes elsewhere. 1.5 is a deliberately moderate choice; grocery literature
    ranges from about 1.0 to well over 3.0 for staples where substitution is
    easy and loyalty is at stake.

order_cost = 0.50 per order line
    The marginal administrative cost of one more line on a truck that is going
    to the store anyway. Deliberately small - treating each SKU order as if it
    dispatched its own vehicle would be wrong for replenishment inside an
    existing distribution network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class CostParams:
    """All economic assumptions in one place, so they can be varied in one place."""

    gross_margin: float = 0.25
    holding_rate_annual: float = 0.25
    stockout_penalty: float = 1.5
    order_cost: float = 0.50

    @property
    def holding_rate_daily(self) -> float:
        return self.holding_rate_annual / DAYS_PER_YEAR

    def unit_cost(self, price: np.ndarray) -> np.ndarray:
        return price * (1.0 - self.gross_margin)

    def unit_margin(self, price: np.ndarray) -> np.ndarray:
        return price * self.gross_margin

    def label(self) -> str:
        return (
            f"m={self.gross_margin:.2f} h={self.holding_rate_annual:.2f} "
            f"p={self.stockout_penalty:.1f} K={self.order_cost:.2f}"
        )


@dataclass
class CostBreakdown:
    holding: float
    stockout: float
    ordering: float

    @property
    def total(self) -> float:
        return self.holding + self.stockout + self.ordering

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}holding_cost": self.holding,
            f"{prefix}stockout_cost": self.stockout,
            f"{prefix}ordering_cost": self.ordering,
            f"{prefix}total_cost": self.total,
        }


def evaluate(
    on_hand: np.ndarray,
    unmet: np.ndarray,
    n_orders: np.ndarray,
    price: np.ndarray,
    params: CostParams,
) -> CostBreakdown:
    """Cost of one simulated policy run.

    on_hand : (n_series, n_days) units on hand at end of each day
    unmet   : (n_series, n_days) demand that could not be served
    n_orders: (n_series,) count of orders placed
    price   : (n_series, n_days) selling price

    Holding is charged on END-OF-DAY inventory. Charging on the average of
    opening and closing balances would be marginally more accurate and would not
    change any comparison, since both policies are costed the same way.
    """
    with np.errstate(invalid="ignore"):
        unit_cost = params.unit_cost(price)
        unit_margin = params.unit_margin(price)

        holding = float(np.nansum(on_hand * unit_cost) * params.holding_rate_daily)
        stockout = float(np.nansum(unmet * unit_margin) * params.stockout_penalty)
        ordering = float(np.nansum(n_orders) * params.order_cost)

    return CostBreakdown(holding=holding, stockout=stockout, ordering=ordering)


def service_level(demand: np.ndarray, unmet: np.ndarray) -> dict[str, float]:
    """Two service definitions, because they answer different questions.

    fill_rate   fraction of demanded UNITS actually served. This is the one the
                cost model cares about and the one a service target normally
                means in grocery.
    cycle_service
                fraction of DAYS with no shortfall at all.

    NEITHER IS RELIABLY THE LARGER. Which one is higher depends on the SHAPE of
    the failures, not on the definitions: rare-but-large shortfalls push fill
    rate below cycle service, while frequent-but-tiny ones do the reverse. On
    this data the failures are rare and large, so cycle service comes out
    higher - but that is an empirical property of these series, not a rule, and
    treating it as a rule is how a service number gets quoted on whichever
    definition happens to flatter it.

    Both are reported everywhere for exactly that reason.
    """
    with np.errstate(invalid="ignore"):
        total_demand = float(np.nansum(demand))
        total_unmet = float(np.nansum(unmet))
        served_days = float(np.nansum((unmet <= 0) & np.isfinite(unmet)))
        total_days = float(np.nansum(np.isfinite(unmet)))

    return {
        "fill_rate": 1.0 - (total_unmet / total_demand) if total_demand > 0 else float("nan"),
        "cycle_service": served_days / total_days if total_days > 0 else float("nan"),
        "total_demand_units": total_demand,
        "total_unmet_units": total_unmet,
    }


def sensitivity_grid(
    base: CostParams | None = None,
    gross_margins: tuple[float, ...] = (0.20, 0.25, 0.30),
    holding_rates: tuple[float, ...] = (0.15, 0.25, 0.40),
    penalties: tuple[float, ...] = (1.0, 1.5, 3.0),
    order_costs: tuple[float, ...] = (0.50,),
) -> list[CostParams]:
    """The parameter sets the results are reported across.

    One-at-a-time sensitivity would understate the uncertainty, because holding
    cost and stockout penalty push the optimum in opposite directions and can
    partially cancel. This is a full factorial over the plausible ranges.
    """
    base = base or CostParams()
    out = []
    for m, h, p, k in product(gross_margins, holding_rates, penalties, order_costs):
        out.append(
            replace(
                base,
                gross_margin=m,
                holding_rate_annual=h,
                stockout_penalty=p,
                order_cost=k,
            )
        )
    return out
