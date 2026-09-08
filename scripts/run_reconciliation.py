"""Hierarchical reconciliation over the full 12-level M5 structure.

Usage:  uv run python scripts/run_reconciliation.py

Requires scripts/run_model_ladder.py to have written forecast_test.npz.

Produces in $SCI_DATA_DIR/artifacts/:
    reconciliation_accuracy.csv   RMSSE per level, base vs each method
    reconciliation_coherence.csv  the adding-up violation, before and after
    hierarchy_levels.csv          node counts per level

THE SETUP
---------
Base forecasts are made INDEPENDENTLY at each altitude, which is what creates
the incoherence in the first place:

    bottom (item x store)   the global LightGBM from the model ladder
    levels 1-11             AutoETS per node - aggregates are smooth and
                            weekly-seasonal, which is ETS's home ground

Then four reconciliation methods are compared against the base forecasts and
against each other, scored per level.

WHAT WOULD MAKE THIS A FAILURE
------------------------------
If reconciliation improved coherence but degraded accuracy at every level, it
would be buying a property nobody asked for at a real cost. The per-level
accuracy table is what checks that, and it is reported whichever way it comes
out.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from evaluation.protocol import FIRST_D, test_fold
from features import panel_matrix
from models import hierarchy, reconcile, statistical
from sci.config import settings
from sci.log import get_logger

log = get_logger(__name__)

HISTORY_DAYS = 730
METHODS = ("bottom_up", "ols", "wls_struct", "wls_var")


def rmsse_per_node(actual: np.ndarray, pred: np.ndarray, scale_sq: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.nanmean((actual - pred) ** 2, axis=1)
        return np.sqrt(mse / scale_sq)


def main() -> None:
    settings.ensure_dirs()
    art = settings.artifacts_dir

    fc_path = art / "forecast_test.npz"
    if not fc_path.exists():
        raise SystemExit(
            f"{fc_path} not found. Run scripts/run_model_ladder.py first — the bottom-level "
            "forecast comes from there."
        )

    pm = panel_matrix.build()
    fold = test_fold()
    horizon = fold.horizon

    log.info("=== building the hierarchy ===")
    h = hierarchy.build(pm.series)
    h.summary().write_csv(art / "hierarchy_levels.csv")

    # ---- history, aggregated up ------------------------------------------
    hist_end = fold.train_end_d - FIRST_D + 1
    hist_start = max(0, hist_end - HISTORY_DAYS)
    # Aggregates must be built from OBSERVED units (NaN outside availability
    # means "did not exist", which contributes zero to a sum), not from the
    # censored-masked training array - a store's total did not dip because one
    # SKU was out of stock, it dipped by exactly that SKU's lost sales.
    bottom_hist = np.nan_to_num(pm.units[:, hist_start:hist_end], nan=0.0).astype(np.float64)
    agg_hist = h.aggregate(bottom_hist)
    log.info("aggregated history: %s", agg_hist.shape)

    # ---- actuals over the test window ------------------------------------
    v_lo = fold.valid_start_d - FIRST_D
    bottom_actual = np.nan_to_num(pm.units[:, v_lo : v_lo + horizon], nan=0.0).astype(np.float64)
    node_actual = h.aggregate(bottom_actual)

    # ---- scaling denominators, per node ----------------------------------
    diff = np.diff(agg_hist, axis=1)
    scale_sq = np.nanmean(diff**2, axis=1)
    scale_sq = np.where(scale_sq > 0, scale_sq, np.nan)

    # ---- base forecasts ---------------------------------------------------
    log.info("=== base forecasts ===")
    bottom_base = np.nan_to_num(np.load(fc_path)["point"], nan=0.0).astype(np.float64)
    log.info("bottom level: global LightGBM, %s", bottom_base.shape)

    # Aggregate levels are split by the CHARACTER of the series, not by their
    # position in the hierarchy.
    #
    #   levels 1-9  (154 nodes)     genuinely smooth once summed over a whole
    #                               store or category -> AutoETS
    #   item, item_state (12,196)   still intermittent: one slow SKU summed over
    #                               ten stores still has zero days -> TSB
    #
    # A first version ran AutoETS on all 12,350. It took over twenty minutes and
    # pushed the machine into swap, for exactly the levels where a smoothing
    # model has least to offer.
    smooth_end = h.rows_for("item").start  # levels 1-9
    agg_rows = h.rows_for("item_store").start

    log.info("smooth aggregate levels: AutoETS over %d nodes", smooth_end)
    t0 = time.time()
    smooth_fc = statistical.forecast_aggregates(agg_hist[:smooth_end], horizon)
    log.info("  fitted in %.1fs", time.time() - t0)

    log.info("intermittent aggregate levels: TSB over %d nodes", agg_rows - smooth_end)
    t0 = time.time()
    sparse_fc = statistical.forecast_intermittent_nodes(agg_hist[smooth_end:agg_rows], horizon)
    log.info("  fitted in %.1fs", time.time() - t0)

    base = np.zeros((h.n_nodes, horizon), dtype=np.float64)
    base[:smooth_end] = smooth_fc.point
    base[smooth_end:agg_rows] = sparse_fc.point
    base[h.rows_for("item_store")] = bottom_base

    # In-sample MSE per node, for the wls_var weights. The bottom level's is
    # taken from the naive scale, which is the same quantity RMSSE uses.
    residual_mse = np.empty(h.n_nodes, dtype=np.float64)
    residual_mse[:smooth_end] = smooth_fc.in_sample_mse
    residual_mse[smooth_end:agg_rows] = sparse_fc.in_sample_mse
    bottom_scale = scale_sq[h.rows_for("item_store")]
    residual_mse[h.rows_for("item_store")] = np.nan_to_num(bottom_scale, nan=1.0)

    # ---- how incoherent are the base forecasts? ---------------------------
    coh_abs = reconcile.coherence_error(h, base)
    coh_rel = reconcile.relative_coherence_error(h, base)
    total_base = base[h.rows_for("total")].sum()
    total_implied = (h.S @ base[h.rows_for("item_store")])[h.rows_for("total")].sum()
    log.info(
        "base coherence violation: max %.1f units, %.2f%% of total; "
        "total-level forecast says %.0f, bottom sums to %.0f",
        coh_abs,
        100 * coh_rel,
        total_base,
        total_implied,
    )

    # ---- reconcile --------------------------------------------------------
    log.info("=== reconciling ===")
    results: dict[str, np.ndarray] = {"base": base}
    for method in METHODS:
        t0 = time.time()
        r = reconcile.reconcile(h, base, method=method, residual_mse=residual_mse)
        results[method] = r.full(h)
        log.info("  %-12s %.1fs", method, time.time() - t0)

    # ---- score per level --------------------------------------------------
    rows = []
    for name, node_pred in results.items():
        per_node = rmsse_per_node(node_actual, node_pred, scale_sq)
        for level in h.level_names:
            sl = h.rows_for(level)
            vals = per_node[sl]
            ok = np.isfinite(vals)
            rows.append(
                {
                    "method": name,
                    "level": level,
                    "n_nodes": sl.stop - sl.start,
                    "rmsse": float(vals[ok].mean()) if ok.any() else float("nan"),
                }
            )

    acc = pl.DataFrame(rows)
    acc.write_csv(art / "reconciliation_accuracy.csv")

    wide = acc.pivot(values="rmsse", index=["level", "n_nodes"], on="method")
    order = {n: i for i, n in enumerate(h.level_names)}
    wide = (
        wide.with_columns(pl.col("level").replace_strict(order, return_dtype=pl.Int32).alias("_o"))
        .sort("_o")
        .drop("_o")
    )

    print("\n=== RMSSE BY LEVEL (lower is better) ===")
    print(wide)

    coh_rows = []
    for name, node_pred in results.items():
        coh_rows.append(
            {
                "method": name,
                "max_abs_violation": reconcile.coherence_error(h, node_pred),
                "relative_violation": reconcile.relative_coherence_error(h, node_pred),
            }
        )
    coh = pl.DataFrame(coh_rows)
    coh.write_csv(art / "reconciliation_coherence.csv")
    print("\n=== COHERENCE (adding-up violation) ===")
    print(coh)

    # ---- headline ---------------------------------------------------------
    agg_levels = [n for n in h.level_names if n != "item_store"]
    summary = (
        acc.filter(pl.col("level").is_in(agg_levels))
        .group_by("method")
        .agg(pl.col("rmsse").mean().alias("mean_rmsse_aggregate_levels"))
        .sort("mean_rmsse_aggregate_levels")
    )
    print("\n=== MEAN RMSSE ACROSS AGGREGATE LEVELS ===")
    print(summary)

    bottom_only = acc.filter(pl.col("level") == "item_store").sort("rmsse")
    print("\n=== BOTTOM LEVEL (item x store) ===")
    print(bottom_only.select("method", "rmsse"))

    log.info("done - artifacts in %s", art)


if __name__ == "__main__":
    main()
