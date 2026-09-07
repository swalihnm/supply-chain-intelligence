-- ---------------------------------------------------------------------------
-- Promotion lift: demand during price-drop windows vs MATCHED control periods.
--
-- M5 has no promotion flag. Promotions are inferred from price: a week is
-- treated as promotional when its price falls meaningfully below that series'
-- own recent reference price.
--
-- WHY MATCHED CONTROLS RATHER THAN A RAW BEFORE/AFTER
-- ---------------------------------------------------
-- Comparing promo days against all other days confounds three things at once:
-- promotions cluster in particular seasons, they are chosen for particular
-- items, and demand has strong day-of-week structure. A raw comparison would
-- attribute all of that to the discount.
--
-- Each promo day is therefore compared against non-promo days of THE SAME
-- SERIES, on THE SAME DAY OF WEEK, in THE SAME CALENDAR QUARTER. That holds
-- item, store, weekly seasonality and coarse season fixed, leaving the price
-- change as the main thing that differs.
--
-- Residual confounding remains, and it is not small: retailers time promotions
-- to demand they already expect, and often pair them with display space and
-- advertising that this data cannot see. The estimate below is an upper bound
-- on the causal effect of price alone.
--
-- The reference price is a trailing 8-week MAXIMUM rather than a mean, so a
-- long promotion does not gradually redefine its own baseline into
-- non-existence.
--
-- PROMOTIONS vs PERMANENT MARKDOWNS
-- ---------------------------------
-- A trailing-only definition cannot tell a temporary promotion from a permanent
-- price cut: after a list-price reduction, the next 8 weeks all sit below the
-- trailing maximum and every one of them looks promotional. That matters,
-- because permanent markdowns are disproportionately applied to products being
-- cleared or already declining - so mixing them in drags measured "promo lift"
-- below 1.0 and makes discounting look harmful.
--
-- A week is therefore classed as a genuine PROMOTION only when the price also
-- RECOVERS - it returns to at least 97% of the reference within the following 8
-- weeks. Weeks that never recover are classed as permanent markdowns and
-- reported on a separate line rather than dropped.
--
-- That recovery test looks FORWARD, which is legitimate here and would not be
-- in the feature layer. This file is a descriptive diagnostic run once over
-- history; nothing it computes is fed to a model. The promo flags that DO reach
-- features/ must be built from trailing information only.
-- ---------------------------------------------------------------------------

WITH weekly_price AS (
    -- One row per series-week. sell_price is constant within a Walmart week.
    SELECT
        item_id,
        store_id,
        wm_yr_wk,
        any_value(cat_id)          AS cat_id,
        any_value(dept_id)         AS dept_id,
        avg(sell_price)            AS price
    FROM core.panel
    GROUP BY 1, 2, 3
),
referenced AS (
    SELECT
        *,
        -- Trailing 8 weeks, strictly before the current one. No lookahead.
        max(price) OVER (
            PARTITION BY item_id, store_id ORDER BY wm_yr_wk
            ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
        ) AS reference_price,
        count(*) OVER (
            PARTITION BY item_id, store_id ORDER BY wm_yr_wk
            ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
        ) AS reference_weeks
    FROM weekly_price
),
recovery AS (
    -- Highest price over the FOLLOWING 8 weeks. Forward-looking on purpose; see
    -- the header for why that is acceptable in a descriptive report only.
    SELECT
        *,
        max(price) OVER (
            PARTITION BY item_id, store_id ORDER BY wm_yr_wk
            ROWS BETWEEN 1 FOLLOWING AND 8 FOLLOWING
        ) AS max_price_next_8w
    FROM referenced
),
promo_weeks AS (
    SELECT
        item_id,
        store_id,
        wm_yr_wk,
        cat_id,
        price,
        reference_price,
        1.0 - price / reference_price                     AS discount_depth,
        -- A 5% floor keeps rounding noise and tiny list-price drift out.
        (price < 0.95 * reference_price)                  AS is_discounted,
        -- Recovers to within 3% of the old reference => temporary promotion.
        (price < 0.95 * reference_price
         AND max_price_next_8w >= 0.97 * reference_price) AS is_promo,
        (price < 0.95 * reference_price
         AND max_price_next_8w <  0.97 * reference_price) AS is_permanent_markdown
    FROM recovery
    WHERE reference_weeks >= 8      -- need a full reference window to judge
      AND reference_price > 0
      AND max_price_next_8w IS NOT NULL
),
-- Attach the weekly promo verdict back to daily rows, dropping censored days.
daily AS (
    SELECT
        p.item_id,
        p.store_id,
        p.d,
        p.units,
        p.cat_id,
        p.wday,
        p.year,
        ((p.month - 1) / 3)::INTEGER                      AS quarter,
        CASE
            WHEN pw.is_promo              THEN 'promo'
            WHEN pw.is_permanent_markdown THEN 'markdown'
            ELSE                               'base'
        END                                               AS price_state,
        pw.discount_depth
    FROM core.panel p
    JOIN promo_weeks pw
      ON  pw.item_id  = p.item_id
      AND pw.store_id = p.store_id
      AND pw.wm_yr_wk = p.wm_yr_wk
    LEFT JOIN core.stockout_flags f
      ON  f.item_id  = p.item_id
      AND f.store_id = p.store_id
      AND p.d BETWEEN f.run_start_d AND f.run_end_d
    WHERE f.item_id IS NULL
),
-- The matched control: same series, same weekday, same quarter, at BASE price.
-- Both discounted states are excluded from the control so a markdown cannot
-- quietly depress the baseline it is being compared against.
control AS (
    SELECT
        item_id, store_id, wday, year, quarter,
        avg(units)  AS control_mean_units,
        count(*)    AS control_n
    FROM daily
    WHERE price_state = 'base'
    GROUP BY 1, 2, 3, 4, 5
),
treated AS (
    SELECT
        d.cat_id,
        d.price_state,
        d.item_id, d.store_id, d.wday, d.year, d.quarter,
        d.discount_depth,
        avg(d.units) AS promo_mean_units,
        count(*)     AS promo_n
    FROM daily d
    WHERE d.price_state <> 'base'
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
),
matched AS (
    SELECT
        t.cat_id,
        t.price_state,
        t.discount_depth,
        t.promo_mean_units,
        c.control_mean_units,
        t.promo_n,
        c.control_n
    FROM treated t
    JOIN control c
      ON  c.item_id  = t.item_id
      AND c.store_id = t.store_id
      AND c.wday     = t.wday
      AND c.year     = t.year
      AND c.quarter  = t.quarter
    -- Both arms need enough observations to be worth comparing. The requirement
    -- is symmetric on purpose: an earlier version demanded 3+ control days but
    -- accepted a single promo day, which let one-observation cells - almost all
    -- of them zero, since 91% of series are intermittent or lumpy - dominate the
    -- estimate.
    WHERE c.control_n >= 5
      AND t.promo_n   >= 5
      AND c.control_mean_units > 0
)
SELECT
    cat_id,
    price_state,
    CASE
        WHEN discount_depth < 0.10 THEN 'a. 5-10%'
        WHEN discount_depth < 0.20 THEN 'b. 10-20%'
        WHEN discount_depth < 0.35 THEN 'c. 20-35%'
        WHEN discount_depth < 0.50 THEN 'd. 35-50%'
        ELSE                            'e. 50%+'
    END                                                          AS discount_band,
    count(*)                                                     AS matched_cells,
    sum(promo_n)                                                 AS promo_days,
    sum(control_n)                                               AS control_days,
    -- Observed promo units vs units EXPECTED had those same promo days run at
    -- their own cell's base rate. Weighting by promo_n is what makes this a
    -- valid aggregate: an unweighted avg(promo)/avg(control) across cells is a
    -- ratio of means, which Simpson's paradox can send the wrong way entirely
    -- when the promo and base arms have different cell-size distributions.
    round(sum(promo_mean_units * promo_n), 1)                    AS observed_units,
    round(sum(control_mean_units * promo_n), 1)                  AS expected_units_at_base,
    round(sum(promo_mean_units * promo_n) / sum(control_mean_units * promo_n), 3) AS lift,
    round(100.0 * (sum(promo_mean_units * promo_n)
                   / sum(control_mean_units * promo_n) - 1), 1)  AS pct_uplift,
    -- Robustness check: the typical CELL's lift, unweighted. If this disagrees
    -- sharply with the weighted figure, a few large cells are driving the result.
    round(median(promo_mean_units / control_mean_units), 3)      AS median_cell_lift,
    round(avg(discount_depth), 3)                                AS avg_discount
FROM matched
GROUP BY 1, 2, 3
-- Drop bands too thin to say anything about. This is a group-level filter on an
-- aggregate, so it is HAVING, not QUALIFY - QUALIFY filters on window results.
HAVING sum(promo_n) >= 500
ORDER BY cat_id, price_state, discount_band;
