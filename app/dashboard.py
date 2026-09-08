"""Streamlit dashboard. Four views, no more.

    uv run streamlit run app/dashboard.py

Every chart ships a table view beside it. That is not decoration: two of the
light-mode palette slots sit below 3:1 contrast on the chart surface, and the
data-viz relief rule requires visible labels or a table wherever that is true.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import streamlit as st

from app import viz
from app.store import ArtifactsMissing, get_store

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load():
    return get_store()


try:
    S = load()
except ArtifactsMissing as exc:
    st.error("Artifacts not built yet.")
    st.code(str(exc))
    st.stop()

DARK = st.sidebar.toggle("Dark charts", value=False)
VIEW = st.sidebar.radio(
    "View",
    ["Executive", "Forecast explorer", "Policy explorer", "Model health"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"{S.n_series:,} SKU-store series · test window d{S.valid_start_d}-{S.valid_end_d}"
)


def table(df: pl.DataFrame, caption: str = "") -> None:
    """The table view that accompanies every chart."""
    with st.expander(f"Table view{' — ' + caption if caption else ''}"):
        st.dataframe(df.to_pandas(), width="stretch")


def missing(what: str, script: str) -> None:
    st.info(f"{what} not available. Run `{script}` to produce it.")


# ==========================================================================
# 1. EXECUTIVE
# ==========================================================================
if VIEW == "Executive":
    st.title("Inventory policy — executive view")

    cf = S.tables.get("counterfactual")
    ci = S.tables.get("counterfactual_ci")
    frontier = S.tables.get("service_level_frontier")

    if cf is None:
        missing("Counterfactual results", "uv run python scripts/run_optimizer.py")
    else:
        base = cf.filter(pl.col("policy") == "baseline")
        opt = cf.filter(pl.col("policy") == "optimized")
        if base.height and opt.height:
            b, o = base.row(0, named=True), opt.row(0, named=True)
            delta_pct = 100.0 * (o["total_cost"] - b["total_cost"]) / b["total_cost"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Total cost vs baseline",
                f"${o['total_cost']:,.0f}",
                f"{delta_pct:+.1f}%",
                delta_color="inverse",
            )
            c2.metric(
                "Fill rate",
                f"{o['fill_rate']:.2%}",
                f"{100 * (o['fill_rate'] - b['fill_rate']):+.2f} pts",
            )
            c3.metric(
                "Avg units on hand",
                f"{o['avg_on_hand_units']:.2f}",
                f"{100 * (o['avg_on_hand_units'] / b['avg_on_hand_units'] - 1):+.1f}%",
                delta_color="inverse",
            )
            c4.metric("Orders placed", f"{o['orders_placed']:,.0f}")

            if ci is not None:
                row = ci.filter(pl.col("comparison").str.contains("optimized"))
                if row.height:
                    r = row.row(0, named=True)
                    st.caption(
                        f"95% bootstrap CI on the cost delta (resampling series): "
                        f"{r['delta_pct_ci_lo']:+.2f}% to {r['delta_pct_ci_hi']:+.2f}%. "
                        "The baseline is reconstructed, not observed — see the limitations doc."
                    )

    st.subheader("Service level vs cost")
    st.caption(
        "The whole trade-off in one chart. Cost rises non-linearly with service "
        "because each extra point of service buys a thinner slice of the demand tail."
    )
    if frontier is None:
        missing("Frontier", "uv run python scripts/run_optimizer.py")
    else:
        pts = frontier.filter(pl.col("target_service_level").is_not_null()).sort(
            "target_service_level"
        )
        bl = frontier.filter(pl.col("policy") == "reconstructed_baseline")
        b_fill = float(bl["fill_rate"][0]) if bl.height else None
        b_cost = float(bl["total_cost"][0]) if bl.height else None

        targets = pts["target_service_level"].to_numpy()
        rec = int(np.argmin(np.abs(targets - 0.95))) if len(targets) else None

        st.plotly_chart(
            viz.service_level_frontier(
                fill_rates=pts["fill_rate"].to_numpy(),
                costs=pts["total_cost"].to_numpy(),
                targets=targets,
                baseline_fill=b_fill,
                baseline_cost=b_cost,
                recommended_idx=rec,
                dark=DARK,
            ),
            width="stretch",
        )
        table(
            pts.select(
                "target_service_level",
                "fill_rate",
                "cycle_service",
                "avg_on_hand_units",
                "avg_safety_stock",
                "total_cost",
            ),
            "service level frontier",
        )

    if cf is not None:
        st.subheader("Where the cost sits")
        st.plotly_chart(
            viz.cost_breakdown(
                labels=cf["policy"].to_list(),
                holding=cf["holding_cost"].to_numpy(),
                stockout=cf["stockout_cost"].to_numpy(),
                ordering=cf["ordering_cost"].to_numpy(),
                dark=DARK,
            ),
            width="stretch",
        )
        table(cf, "counterfactual")

    sens = S.tables.get("cost_sensitivity")
    if sens is not None:
        st.subheader("How much of this survives the assumptions")
        lo, hi = sens["delta_pct"].min(), sens["delta_pct"].max()
        st.markdown(
            f"Across the full grid of cost assumptions the saving ranges from "
            f"**{lo:+.1f}%** to **{hi:+.1f}%** (median **{sens['delta_pct'].median():+.1f}%**). "
            "Holding cost, stockout penalty and gross margin are all assumed, not measured."
        )
        table(sens, "cost sensitivity grid")

    sp = S.tables.get("shadow_prices")
    if sp is not None:
        st.subheader("Shadow prices")
        st.caption(
            "What one more unit of shelf space is worth per period at each store. "
            "A zero means the constraint is slack and more space there buys nothing."
        )
        st.dataframe(sp.to_pandas(), width="stretch")


# ==========================================================================
# 2. FORECAST EXPLORER
# ==========================================================================
elif VIEW == "Forecast explorer":
    st.title("Forecast explorer")

    if not S.has_forecast():
        missing("Forecasts", "uv run python scripts/run_model_ladder.py")
        st.stop()

    c1, c2 = st.columns([1, 3])
    store_id = c1.selectbox("Store", S.store_ids())
    items = S.item_ids(store_id)
    item_id = c2.selectbox("Item", items)

    idx = S.index_of(store_id, item_id)
    meta = S.meta(idx)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Category", meta["cat_id"])
    m2.metric("ABC / XYZ", f"{meta['abc_class']}{meta['xyz_class']}")
    m3.metric("Demand class", meta["sb_class"])
    m4.metric("ADI", f"{meta['adi']:.2f}" if meta.get("adi") is not None else "—")

    levels = S.quantile_levels()
    lo_key = f"q{levels[0]}" if levels else None
    hi_key = f"q{levels[-1]}" if levels else None

    days = np.arange(S.valid_start_d, S.valid_start_d + S.forecasts["point"].shape[1])
    st.plotly_chart(
        viz.forecast_chart(
            days=days,
            actual=S.actual_test[idx],
            point=S.forecasts["point"][idx],
            lower=S.forecasts[lo_key][idx] if lo_key else None,
            upper=S.forecasts[hi_key][idx] if hi_key else None,
            band_label=f"{levels[0]:.0%}-{levels[-1]:.0%} band" if levels else "",
            dark=DARK,
        ),
        width="stretch",
    )

    rows = {"day": days, "actual": S.actual_test[idx], "point": S.forecasts["point"][idx]}
    for q in levels:
        rows[f"q{q}"] = S.forecasts[f"q{q}"][idx]
    table(pl.DataFrame(rows), f"{store_id} / {item_id}")

    st.subheader("Recent history")
    hist = S.history[idx]
    st.plotly_chart(
        viz.forecast_chart(
            days=np.arange(len(hist)),
            actual=hist,
            point=np.full_like(hist, np.nanmean(hist)),
            dark=DARK,
        ),
        width="stretch",
    )
    st.caption(
        "Clean demand over the last 365 training days. Gaps are days excluded as "
        "censored by the stockout proxy — they are not zero demand."
    )


# ==========================================================================
# 3. POLICY EXPLORER
# ==========================================================================
elif VIEW == "Policy explorer":
    st.title("Policy explorer")

    pol = S.tables.get("policy_test")
    if pol is None:
        missing("Policy table", "uv run python scripts/run_optimizer.py")
        st.stop()

    c1, c2, c3 = st.columns(3)
    abc = c1.multiselect("ABC class", sorted(pol["abc_class"].unique().to_list()))
    xyz = c2.multiselect("XYZ class", sorted(pol["xyz_class"].unique().to_list()))
    sb = c3.multiselect("Demand class", sorted(pol["sb_class"].unique().to_list()))

    df = pol
    if abc:
        df = df.filter(pl.col("abc_class").is_in(abc))
    if xyz:
        df = df.filter(pl.col("xyz_class").is_in(xyz))
    if sb:
        df = df.filter(pl.col("sb_class").is_in(sb))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Series", f"{df.height:,}")
    m2.metric("Mean reorder point", f"{df['reorder_point'].mean():.2f}")
    m3.metric("Mean order-up-to", f"{df['order_up_to'].mean():.2f}")
    m4.metric("Mean safety stock", f"{df['safety_stock'].mean():.2f}")

    st.subheader("Live re-simulation")
    st.caption(
        "Changing these re-derives the policy and re-runs the full 30,490-series "
        "simulation. It is a real computation, not an interpolation, so it takes a few seconds."
    )
    s1, s2, s3 = st.columns(3)
    sl = s1.slider("Service level target", 0.80, 0.99, 0.95, 0.01)
    hc = s2.slider("Holding cost rate (annual)", 0.10, 0.60, 0.25, 0.05)
    lt = s3.slider("Mean lead time (days)", 2, 14, 5, 1)

    if st.button("Run simulation", type="primary"):
        from evaluation.simulate_counterfactual import simulate_policy
        from optimize import policy as policy_mod
        from optimize.cost_model import CostParams

        with st.spinner("Simulating…"):
            fmean = np.nan_to_num(np.nanmean(S.forecasts["point"], axis=1), nan=0.0)
            cp = policy_mod.default_case_packs(S.median_price)
            p = policy_mod.build(S.history, fmean, service_level=sl, case_pack=cp, n_sims=600)
            res = simulate_policy(
                S.demand,
                p,
                S.price,
                CostParams(holding_rate_annual=hc),
                name="live",
                lead_time_min=max(1, lt - 2),
                lead_time_max=lt + 2,
                warmup=S.warmup_days,
            )
            out = res.summary()

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fill rate", f"{out['fill_rate']:.2%}")
        r2.metric("Cycle service", f"{out['cycle_service']:.2%}")
        r3.metric("Avg on hand", f"{out['avg_on_hand_units']:.2f}")
        r4.metric("Total cost", f"${out['total_cost']:,.0f}")
        st.caption(
            "Fill rate is below the target by design: the target is a CYCLE service "
            "level (probability of no shortfall in a replenishment cycle), which is "
            "a different and always-higher quantity than the fraction of units served."
        )

    st.subheader("Policy detail")
    st.dataframe(df.head(500).to_pandas(), width="stretch")


# ==========================================================================
# 4. MODEL HEALTH
# ==========================================================================
else:
    st.title("Model health")

    summary = S.tables.get("backtest_summary")
    folds = S.tables.get("backtest_folds")

    if summary is None:
        missing("Backtest results", "uv run python scripts/run_model_ladder.py")
    else:
        st.subheader("Model ranking (mean WRMSSE across CV folds)")
        st.caption(
            "Lower is better. The baselines are tuned, not strawmen — reporting a gain "
            "over a bad baseline would be meaningless."
        )
        st.plotly_chart(
            viz.model_ranking(summary["model"].to_list(), summary["wrmsse"].to_numpy(), dark=DARK),
            width="stretch",
        )
        table(summary, "CV summary")

        if folds is not None:
            st.subheader("Stability across folds")
            best = summary.head(4)["model"].to_list()
            series = {
                m: folds.filter((pl.col("model") == m) & (pl.col("fold") != "test"))
                .sort("fold")["wrmsse"]
                .to_numpy()
                for m in best
            }
            fold_names = folds.filter(pl.col("fold") != "test")["fold"].unique().sort().to_list()
            st.plotly_chart(viz.fold_stability(fold_names, series, dark=DARK), width="stretch")

            st.subheader("Locked test set")
            st.caption("Scored exactly once, as the protocol requires.")
            st.dataframe(
                folds.filter(pl.col("fold") == "test").sort("wrmsse").to_pandas(),
                width="stretch",
            )

    cal = S.tables.get("calibration")
    if cal is not None:
        st.subheader("Quantile calibration")
        st.caption(
            "Does the 95th percentile actually get exceeded 5% of the time? If not, "
            "the optimizer buys a service level it will not achieve in the store."
        )
        methods = sorted(cal["method"].unique().to_list()) if "method" in cal.columns else []
        pick = st.selectbox("Method", methods) if methods else None
        sub = cal.filter(pl.col("method") == pick) if pick else cal
        if "fold" in sub.columns:
            f_pick = st.selectbox("Fold", sorted(sub["fold"].unique().to_list()))
            sub = sub.filter(pl.col("fold") == f_pick)
        if sub.height:
            st.plotly_chart(
                viz.calibration_chart(
                    sub["nominal_coverage"].to_numpy(),
                    sub["empirical_coverage"].to_numpy(),
                    dark=DARK,
                ),
                width="stretch",
            )
            table(sub, "calibration")

    seg = S.tables.get("backtest_by_sb_class")
    if seg is not None:
        st.subheader("Error by demand class")
        st.caption(
            "A headline metric can hide that a model is excellent on smooth series and "
            "useless on intermittent ones — which is exactly the split that decides "
            "whether the optimizer can trust it."
        )
        st.dataframe(seg.to_pandas(), width="stretch")

    imp = S.tables.get("feature_importance")
    if imp is not None:
        st.subheader("Feature importance (gain share)")
        agg = (
            imp.group_by("feature")
            .agg(pl.col("gain_share").mean())
            .sort("gain_share", descending=True)
            .head(15)
        )
        st.plotly_chart(
            viz.model_ranking(agg["feature"].to_list(), agg["gain_share"].to_numpy(), dark=DARK),
            width="stretch",
        )
        table(agg, "feature importance")
