"""FastAPI service over the forecast and policy artifacts.

    GET  /health
    GET  /forecast/{store_id}/{item_id}?horizon=28
    GET  /policy/{store_id}/{item_id}
    POST /simulate      {service_level, holding_cost_rate, lead_time_days, ...}
    GET  /series        search/browse
    GET  /results       headline numbers from the backtest and counterfactual

/simulate runs a REAL simulation over all 30,490 series on each call, not an
interpolation of a precomputed table. That is affordable because the serving
bundle carries the demand window and history as compact arrays - the whole
simulation is a 56-step loop over vectorised numpy and returns in a few seconds.

Run locally:
    uv run uvicorn app.api:app --reload
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.store import ArtifactsMissing, Store, get_store
from evaluation.simulate_counterfactual import simulate_policy
from optimize import policy as policy_mod
from optimize.cost_model import CostParams

app = FastAPI(
    title="Supply Chain Intelligence",
    description=(
        "Demand forecasts, replenishment policy, and live cost simulation over "
        "30,490 retail SKU-store series."
    ),
    version="0.1.0",
)


def store_dep() -> Store:
    try:
        return get_store()
    except ArtifactsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


StoreDep = Annotated[Store, Depends(store_dep)]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    service_level: float = Field(0.95, ge=0.5, le=0.999)
    holding_cost_rate: float = Field(0.25, gt=0, le=2.0, description="annual carrying rate")
    lead_time_days: int = Field(5, ge=1, le=21, description="mean lead time")
    stockout_penalty: float = Field(1.5, ge=0.0, le=10.0)
    gross_margin: float = Field(0.25, gt=0.0, lt=1.0)
    n_sims: int = Field(600, ge=100, le=3000)


class SimulateResponse(BaseModel):
    service_level_target: float
    fill_rate: float
    cycle_service: float
    avg_on_hand_units: float
    holding_cost: float
    stockout_cost: float
    ordering_cost: float
    total_cost: float
    orders_placed: float
    avg_safety_stock: float
    n_series: int


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    try:
        s = get_store()
    except ArtifactsMissing as exc:
        return {"status": "degraded", "detail": str(exc)}
    return {
        "status": "ok",
        "n_series": s.n_series,
        "has_forecast": s.has_forecast(),
        "quantiles": s.quantile_levels(),
        "test_window": [s.valid_start_d, s.valid_end_d],
        "tables_loaded": sorted(s.tables),
    }


@app.get("/series")
def series(
    s: StoreDep,
    store_id: str | None = None,
    cat_id: str | None = None,
    abc_class: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    import polars as pl

    df = s.series
    if store_id:
        df = df.filter(pl.col("store_id") == store_id)
    if cat_id:
        df = df.filter(pl.col("cat_id") == cat_id)
    if abc_class:
        df = df.filter(pl.col("abc_class") == abc_class)
    return {"total": df.height, "series": df.head(limit).to_dicts()}


@app.get("/forecast/{store_id}/{item_id}")
def forecast(store_id: str, item_id: str, s: StoreDep, horizon: int = Query(28, ge=1, le=28)):
    if not s.has_forecast():
        raise HTTPException(503, "No forecast artifacts. Run scripts/run_model_ladder.py.")
    try:
        idx = s.index_of(store_id, item_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    point = s.forecasts["point"][idx, :horizon]
    quantiles = {f"q{q}": s.forecasts[f"q{q}"][idx, :horizon].tolist() for q in s.quantile_levels()}
    actual = s.actual_test[idx, :horizon]

    return {
        "store_id": store_id,
        "item_id": item_id,
        "meta": s.meta(idx),
        "horizon": horizon,
        "start_d": s.valid_start_d,
        "point": np.nan_to_num(point).tolist(),
        "quantiles": quantiles,
        "actual": [None if np.isnan(v) else float(v) for v in actual],
    }


@app.get("/policy/{store_id}/{item_id}")
def get_policy(store_id: str, item_id: str, s: StoreDep):
    import polars as pl

    tbl = s.tables.get("policy_test")
    if tbl is None:
        raise HTTPException(503, "No policy artifacts. Run scripts/run_optimizer.py.")
    hit = tbl.filter((pl.col("store_id") == store_id) & (pl.col("item_id") == item_id))
    if hit.height == 0:
        raise HTTPException(404, f"no policy for {store_id}/{item_id}")
    return hit.row(0, named=True)


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest, s: StoreDep) -> SimulateResponse:
    """Re-derive the policy at the requested settings and simulate it."""
    if not s.has_forecast():
        raise HTTPException(503, "No forecast artifacts. Run scripts/run_model_ladder.py.")

    forecast_mean = np.nan_to_num(np.nanmean(s.forecasts["point"], axis=1), nan=0.0)
    case_pack = policy_mod.default_case_packs(s.median_price)

    lt = int(req.lead_time_days)
    lo, hi = max(1, lt - 2), lt + 2

    pol = policy_mod.build(
        s.history,
        forecast_mean,
        service_level=req.service_level,
        case_pack=case_pack,
        n_sims=req.n_sims,
    )
    params = CostParams(
        gross_margin=req.gross_margin,
        holding_rate_annual=req.holding_cost_rate,
        stockout_penalty=req.stockout_penalty,
    )
    res = simulate_policy(
        s.demand,
        pol,
        s.price,
        params,
        name="api",
        lead_time_min=lo,
        lead_time_max=hi,
        warmup=s.warmup_days,
    )
    summary = res.summary()
    return SimulateResponse(
        service_level_target=req.service_level,
        fill_rate=summary["fill_rate"],
        cycle_service=summary["cycle_service"],
        avg_on_hand_units=summary["avg_on_hand_units"],
        holding_cost=summary["holding_cost"],
        stockout_cost=summary["stockout_cost"],
        ordering_cost=summary["ordering_cost"],
        total_cost=summary["total_cost"],
        orders_placed=summary["orders_placed"],
        avg_safety_stock=float(np.nanmean(pol.safety_stock)),
        n_series=s.n_series,
    )


@app.get("/results")
def results(s: StoreDep) -> dict:
    """Headline numbers, so the README and dashboard read from one source."""
    out: dict = {}
    for name in (
        "backtest_summary",
        "counterfactual",
        "counterfactual_ci",
        "service_level_frontier",
        "shadow_prices",
        "calibration",
    ):
        if name in s.tables:
            out[name] = s.tables[name].to_dicts()
    return out
