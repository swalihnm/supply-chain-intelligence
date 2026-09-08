"""Week 3: policy, counterfactual, service-level frontier, shadow prices.

Usage:  uv run python scripts/run_optimizer.py

Requires scripts/run_model_ladder.py to have written forecast_test.npz.

Produces in $SCI_DATA_DIR/artifacts/:
    counterfactual.csv          the four-policy comparison on the test window
    counterfactual_ci.csv       bootstrap confidence intervals on the cost delta
    service_level_frontier.csv  cost vs achieved service, the money chart
    cost_sensitivity.csv        the result across the full cost-parameter grid
    shadow_prices.csv           duals on capacity and budget constraints
    policy_test.csv             per-series (s, S), safety stock, class
"""

from __future__ import annotations

import numpy as np
import polars as pl

from evaluation.protocol import FIRST_D, test_fold
from evaluation.simulate_counterfactual import (
    WARMUP_DAYS,
    bootstrap_cost_delta,
    calibrate_policy_to_fill,
    simulate_policy,
)
from features import panel_matrix
from optimize import constrained_solver, expected_cost, policy
from optimize.cost_model import CostParams, sensitivity_grid
from sci.config import settings
from sci.log import get_logger

log = get_logger(__name__)

# Extended to 0.999 because the cost model's arithmetic points high: over a
# ~12-day exposure window a held unit costs about $0.025 while a lost unit costs
# about $1.50, so the newsvendor critical ratio p/(p+h) sits near 0.98.
#
# MEASURED: the frontier is monotone over this whole range - total cost keeps
# falling as the uniform target rises, with no interior minimum. That is the
# reason the recommendation is a PER-SERIES allocation rather than a uniform
# target: if one number were the answer, it would just be "as high as capacity
# allows", which is not a decision worth making a project about.
SERVICE_LEVELS = (0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999)
HISTORY_DAYS = 365
N_SIMS = 1500
N_BOOT = 500


def main() -> None:
    settings.ensure_dirs()
    art = settings.artifacts_dir
    pm = panel_matrix.build()
    fold = test_fold()

    fc_path = art / "forecast_test.npz"
    if not fc_path.exists():
        raise SystemExit(
            f"{fc_path} not found. Run scripts/run_model_ladder.py first - the optimizer "
            "consumes the test-fold forecast it produces."
        )
    fc = np.load(fc_path)
    point = fc["point"]  # (S, 28)

    # ---- windows -----------------------------------------------------------
    # Simulation runs a warm-up period before the scored window so no policy
    # benefits from its own choice of starting inventory.
    sim_start_d = fold.valid_start_d - WARMUP_DAYS
    a = sim_start_d - FIRST_D
    b = fold.valid_end_d - FIRST_D + 1
    demand = pm.units[:, a:b]
    price_sim = pm.price[:, a:b]

    # Fill price gaps so the cost model never multiplies by NaN.
    med_price = np.nanmedian(pm.price, axis=1)
    price_sim = np.where(np.isnan(price_sim), med_price[:, None], price_sim)

    hist_end = fold.train_end_d - FIRST_D + 1
    history = pm.training_units()[:, max(0, hist_end - HISTORY_DAYS) : hist_end]

    forecast_mean = np.nan_to_num(np.nanmean(point, axis=1), nan=0.0)
    naive_mean = np.nan_to_num(np.nanmean(history[:, -28:], axis=1), nan=0.0)

    log.info(
        "series=%d  sim window d%d-%d (%d warm-up + %d scored)",
        pm.n_series,
        sim_start_d,
        fold.valid_end_d,
        WARMUP_DAYS,
        fold.horizon,
    )

    case_pack = policy.default_case_packs(med_price)
    params = CostParams()

    # ---- observed fill rate, from the stockout proxy ------------------------
    # The proxy says 3.99% of panel days were stockouts. Converting that to a
    # UNIT fill rate needs an assumption about how much demand was lost on those
    # days; we assume a stockout day lost that series' own average daily demand.
    stock_days = pm.stockout[:, a:b]
    avg_daily = np.nan_to_num(np.nanmean(history, axis=1), nan=0.0)
    lost_units = float((stock_days * avg_daily[:, None]).sum())
    served_units = float(np.nansum(demand[:, WARMUP_DAYS:]))
    observed_fill = served_units / (served_units + lost_units)
    log.info("observed fill rate implied by the stockout proxy: %.4f", observed_fill)

    # ---- calibrate ALL THREE policies to the same achieved fill rate --------
    #
    # This is the correction that makes the comparison mean anything. A
    # service-level PARAMETER and an achieved FILL RATE are different
    # quantities, and the normal approximation and the empirical bootstrap
    # convert one into the other very differently on intermittent demand.
    #
    # A first version ran the baseline at whatever parameter reproduced observed
    # behaviour (0.999) and the other two at a fixed 0.95, then compared cost.
    # It reported the optimized policy as 32% MORE expensive - while serving
    # 91.1% of units against the baseline's 93.9%. That comparison was measuring
    # the service gap, not the method.
    #
    # Every policy is now tuned to the same achieved fill rate, so the remaining
    # cost difference is attributable to the method: same service, different cost.
    log.info("=== calibrating every policy to fill rate %.4f ===", observed_fill)

    def _baseline_at(sl: float) -> policy.Policy:
        return policy.build_normal_approx(
            history, naive_mean, service_level=sl, case_pack=case_pack
        )

    def _forecast_only_at(sl: float) -> policy.Policy:
        return policy.build_normal_approx(
            history, forecast_mean, service_level=sl, case_pack=case_pack
        )

    def _optimized_at(sl: float) -> policy.Policy:
        return policy.build(
            history, forecast_mean, service_level=sl, case_pack=case_pack, n_sims=N_SIMS
        )

    base_sl, base_fill = calibrate_policy_to_fill(
        demand, _baseline_at, price_sim, params, observed_fill, label="baseline"
    )
    fcst_sl, fcst_fill = calibrate_policy_to_fill(
        demand, _forecast_only_at, price_sim, params, observed_fill, label="forecast_only"
    )
    opt_sl, opt_fill = calibrate_policy_to_fill(
        demand, _optimized_at, price_sim, params, observed_fill, label="optimized"
    )
    log.info(
        "matched fill -> baseline %.4f / forecast_only %.4f / optimized %.4f",
        base_fill,
        fcst_fill,
        opt_fill,
    )

    # ---- the three policies, at matched service ----------------------------
    log.info("=== simulating the three policies at matched service ===")
    pol_base = _baseline_at(base_sl)
    pol_fcst = _forecast_only_at(fcst_sl)
    pol_opt = _optimized_at(opt_sl)

    # ---- the allocation policy: a service level chosen PER SERIES -----------
    #
    # The measured frontier falls monotonically to 99.9%, so there is no
    # interesting uniform target - the real decision is which series deserve the
    # stock. Levels are chosen on EXPECTED cost, derived from the forecast's
    # lead-time demand distribution and never from realised demand. See
    # optimize/expected_cost.py for the overfitting this avoids.
    log.info("=== per-series allocation on expected cost ===")
    level_costs = expected_cost.grid(
        history,
        forecast_mean,
        price_sim[:, WARMUP_DAYS:].mean(axis=1),
        params,
        SERVICE_LEVELS,
        n_sims=N_SIMS,
    )
    chosen_levels, expected_at_choice = expected_cost.argmin_levels(level_costs)
    log.info(
        "  chosen levels: median %.4f, %d distinct; expected cost %.0f",
        float(np.median(chosen_levels)),
        len(np.unique(chosen_levels)),
        float(expected_at_choice.sum()),
    )
    pol_alloc = policy.build(
        history, forecast_mean, service_level=chosen_levels, case_pack=case_pack, n_sims=N_SIMS
    )

    runs = {
        "baseline": simulate_policy(demand, pol_base, price_sim, params, "baseline"),
        "forecast_only": simulate_policy(demand, pol_fcst, price_sim, params, "forecast_only"),
        "uniform_empirical": simulate_policy(
            demand, pol_opt, price_sim, params, "uniform_empirical"
        ),
        "cost_optimal_allocation": simulate_policy(
            demand, pol_alloc, price_sim, params, "cost_optimal_allocation"
        ),
    }
    cf = pl.DataFrame([r.summary() for r in runs.values()])
    cf.write_csv(art / "counterfactual.csv")
    print("\n=== COUNTERFACTUAL (test window, d1914-1941) ===")
    print(cf.select("policy", "fill_rate", "cycle_service", "avg_on_hand_units", "total_cost"))

    # ---- bootstrap CI on the deltas ----------------------------------------
    ci_rows = []
    for name in ("forecast_only", "uniform_empirical", "cost_optimal_allocation"):
        d = bootstrap_cost_delta(
            runs["baseline"], runs[name], price_sim[:, WARMUP_DAYS:], params, n_boot=N_BOOT
        )
        d["comparison"] = f"baseline -> {name}"
        ci_rows.append(d)
    ci = pl.DataFrame(ci_rows)
    ci.write_csv(art / "counterfactual_ci.csv")
    print("\n=== COST DELTA WITH 95% BOOTSTRAP CI ===")
    print(ci.select("comparison", "delta", "delta_pct", "delta_pct_ci_lo", "delta_pct_ci_hi"))

    # ---- service level frontier --------------------------------------------
    log.info("=== service level frontier ===")
    frontier_rows = []
    policies_by_sl: dict[float, policy.Policy] = {}
    for sl in SERVICE_LEVELS:
        p = policy.build(
            history, forecast_mean, service_level=sl, case_pack=case_pack, n_sims=N_SIMS
        )
        policies_by_sl[sl] = p
        r = simulate_policy(demand, p, price_sim, params, f"sl_{sl}")
        row = r.summary()
        row["target_service_level"] = sl
        row["avg_safety_stock"] = float(np.nanmean(p.safety_stock))
        frontier_rows.append(row)
        log.info(
            "  target %.3f -> fill %.4f, cost %.0f, avg on hand %.2f",
            sl,
            row["fill_rate"],
            row["total_cost"],
            row["avg_on_hand_units"],
        )

    base_row = runs["baseline"].summary()
    base_row["target_service_level"] = float("nan")
    base_row["avg_safety_stock"] = float(np.nanmean(pol_base.safety_stock))
    base_row["policy"] = "reconstructed_baseline"
    frontier_rows.append(base_row)

    frontier = pl.DataFrame(frontier_rows)
    frontier.write_csv(art / "service_level_frontier.csv")
    print("\n=== SERVICE LEVEL FRONTIER ===")
    print(
        frontier.select(
            "policy", "target_service_level", "fill_rate", "avg_on_hand_units", "total_cost"
        )
    )

    # ---- cost sensitivity ---------------------------------------------------
    log.info("=== cost parameter sensitivity ===")
    sens_rows = []
    for cp in sensitivity_grid():
        rb = simulate_policy(demand, pol_base, price_sim, cp, "baseline")
        ro = simulate_policy(demand, pol_alloc, price_sim, cp, "cost_optimal_allocation")
        sens_rows.append(
            {
                "gross_margin": cp.gross_margin,
                "holding_rate": cp.holding_rate_annual,
                "stockout_penalty": cp.stockout_penalty,
                "baseline_cost": rb.cost.total,
                "optimized_cost": ro.cost.total,
                "delta": ro.cost.total - rb.cost.total,
                "delta_pct": 100.0 * (ro.cost.total - rb.cost.total) / rb.cost.total,
                "baseline_fill": rb.service["fill_rate"],
                "optimized_fill": ro.service["fill_rate"],
            }
        )
    sens = pl.DataFrame(sens_rows)
    sens.write_csv(art / "cost_sensitivity.csv")
    print("\n=== COST SENSITIVITY (delta %% across the assumption grid) ===")
    print(
        sens.select("gross_margin", "holding_rate", "stockout_penalty", "delta_pct").sort(
            "delta_pct"
        )
    )
    print(
        f"\ndelta_pct range: {sens['delta_pct'].min():.2f}% to {sens['delta_pct'].max():.2f}%  "
        f"(median {sens['delta_pct'].median():.2f}%)"
    )

    # ---- constrained allocation + shadow prices -----------------------------
    log.info("=== constrained allocation ===")
    store_names = sorted(pm.series["store_id"].unique().to_list())
    store_code = np.array(
        [store_names.index(s) for s in pm.series["store_id"].to_numpy()], dtype=np.int32
    )

    # Candidates are costed on EXPECTED cost from the forecast distribution, not
    # on simulated realised cost. Using realised cost here would let the solver
    # pick each series' level with knowledge of the outcome it is then scored
    # against - it inflated the "unconstrained optimum" to 77,554 against a
    # 127,930 baseline, a 39% saving that was mostly hindsight.
    cands = [
        constrained_solver.Candidate(
            service_level=lc.service_level,
            expected_cost=lc.expected_cost,
            inventory_units=lc.inventory_units,
            capital=lc.capital,
        )
        for lc in level_costs
    ]

    unconstrained = constrained_solver.solve(cands, store_code, store_names)
    log.info("unconstrained optimum cost: %.0f", unconstrained.total_cost)

    # Capacity set to 85% of what the unconstrained optimum wants at each store,
    # so the constraint actually binds and the duals are informative.
    best_units = np.stack([c.inventory_units for c in cands], axis=1)
    k_star = np.nanargmin(np.stack([c.expected_cost for c in cands], axis=1), axis=1)
    want = best_units[np.arange(pm.n_series), k_star]
    capacity = {s: 0.85 * float(want[store_code == i].sum()) for i, s in enumerate(store_names)}
    budget = 0.90 * float(
        np.stack([c.capital for c in cands], axis=1)[np.arange(pm.n_series), k_star].sum()
    )

    constrained = constrained_solver.solve(
        cands, store_code, store_names, store_capacity=capacity, total_budget=budget
    )
    print("\n=== SHADOW PRICES ===")
    print(constrained.constraint_usage)
    print(
        f"\nconstrained cost {constrained.total_cost:,.0f} vs unconstrained "
        f"{constrained.unconstrained_cost:,.0f} "
        f"(constraints cost {constrained.cost_of_constraints():,.0f}); "
        f"{constrained.n_fractional} fractional assignments out of {pm.n_series}"
    )
    constrained.constraint_usage.write_csv(art / "shadow_prices.csv")

    # ---- per-series policy output -------------------------------------------
    pl.DataFrame(
        {
            "item_id": pm.series["item_id"],
            "store_id": pm.series["store_id"],
            "cat_id": pm.series["cat_id"],
            "abc_class": pm.series["abc_class"],
            "xyz_class": pm.series["xyz_class"],
            "sb_class": pm.series["sb_class"],
            "forecast_daily_mean": forecast_mean,
            "reorder_point": pol_alloc.reorder_point,
            "order_up_to": pol_alloc.order_up_to,
            "safety_stock": pol_alloc.safety_stock,
            "expected_lt_demand": pol_alloc.expected_lt_demand,
            "chosen_service_level": chosen_levels,
            "constrained_service_level": constrained.chosen_level,
        }
    ).write_csv(art / "policy_test.csv")

    log.info("done - artifacts in %s", art)


if __name__ == "__main__":
    main()
