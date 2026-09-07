-- ---------------------------------------------------------------------------
-- Naive log-log price elasticity per category and department.
--
-- Fits, in closed form via covariance over the pooled panel:
--
--     ln(units) = a + b * ln(price)          b = elasticity
--
-- The OLS slope is cov(x,y)/var(x), which DuckDB computes directly with
-- regr_slope, so no modelling library is needed for a baseline.
--
-- THIS IS A SANITY BASELINE, NOT A CAUSAL ESTIMATE. Stating why, because the
-- number will look authoritative and is not:
--
--   * Endogeneity. Prices are set in response to expected demand. A retailer
--     discounts what is not moving and holds price on what is. That pushes the
--     estimate toward zero or even positive, understating true elasticity.
--   * No controls. Seasonality, promotions, competitor pricing and assortment
--     changes all move with price and none are held fixed here.
--   * Selection. Only days with positive sales enter the fit, since ln(0) is
--     undefined. That drops exactly the low-demand days a high price would
--     cause, biasing the slope upward.
--   * Pooling. One elasticity per category averages over items with genuinely
--     different price responses.
--
-- A defensible elasticity needs an instrument or a natural experiment. The
-- promo-lift analysis in promo_lift_analysis.sql is a step closer by using
-- matched control periods; this file is the crude first look that says whether
-- the sign is even sensible.
--
-- Censored days are excluded: a stockout zero is not a price response.
-- ---------------------------------------------------------------------------

WITH clean AS (
    SELECT p.cat_id, p.dept_id, p.units, p.sell_price
    FROM core.panel p
    LEFT JOIN core.stockout_flags f
      ON  f.item_id  = p.item_id
      AND f.store_id = p.store_id
      AND p.d BETWEEN f.run_start_d AND f.run_end_d
    WHERE f.item_id IS NULL
      AND p.units > 0            -- ln(0) undefined; see the selection caveat above
      AND p.sell_price > 0
),
logged AS (
    SELECT
        cat_id,
        dept_id,
        ln(units)      AS ln_units,
        ln(sell_price) AS ln_price
    FROM clean
),
by_dept AS (
    SELECT
        cat_id,
        dept_id,
        count(*)                                  AS n_obs,
        regr_slope(ln_units, ln_price)            AS elasticity,
        regr_intercept(ln_units, ln_price)        AS intercept,
        regr_r2(ln_units, ln_price)               AS r2,
        corr(ln_units, ln_price)                  AS correlation,
        stddev_samp(ln_price)                     AS sd_ln_price
    FROM logged
    GROUP BY 1, 2
),
by_cat AS (
    SELECT
        cat_id,
        'ALL'                                     AS dept_id,
        count(*)                                  AS n_obs,
        regr_slope(ln_units, ln_price)            AS elasticity,
        regr_intercept(ln_units, ln_price)        AS intercept,
        regr_r2(ln_units, ln_price)               AS r2,
        corr(ln_units, ln_price)                  AS correlation,
        stddev_samp(ln_price)                     AS sd_ln_price
    FROM logged
    GROUP BY 1, 2
)
SELECT
    cat_id,
    dept_id,
    n_obs,
    round(elasticity, 4)     AS elasticity,
    round(r2, 4)             AS r2,
    round(correlation, 4)    AS correlation,
    round(sd_ln_price, 4)    AS sd_ln_price,
    CASE
        WHEN elasticity < -1  THEN 'elastic (demand moves more than price, in %)'
        WHEN elasticity < 0   THEN 'inelastic (correct sign, weak response)'
        ELSE 'POSITIVE - endogeneity or premium effect, do not read as causal'
    END                      AS reading,
    -- Rough Giffen/quality-signal warning: a positive slope on a staple is
    -- almost always the retailer's pricing decision showing through, not
    -- consumers wanting more of a thing because it costs more.
    (elasticity > 0)         AS suspicious_sign
FROM (
    SELECT * FROM by_cat
    UNION ALL
    SELECT * FROM by_dept
)
ORDER BY cat_id, (dept_id = 'ALL') DESC, dept_id;
