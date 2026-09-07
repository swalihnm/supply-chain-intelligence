-- ---------------------------------------------------------------------------
-- Seasonality profile: day-of-week, month, SNAP and holiday lift by category.
--
-- "Lift" throughout is the ratio of a segment's mean daily units to that
-- category's own overall mean, computed with a window function so the baseline
-- travels with each row rather than being hardcoded. A lift of 1.00 is average;
-- 1.20 means 20% above the category's own norm.
--
-- Rows on flagged stockout days are EXCLUDED. Leaving them in would drag the
-- seasonal profile toward whichever days happened to have supply failures,
-- which is precisely the censoring bias the whole pipeline exists to avoid.
-- ---------------------------------------------------------------------------

WITH clean AS (
    -- The panel minus censored days. An anti-join against the flagged runs.
    SELECT p.*
    FROM core.panel p
    LEFT JOIN core.stockout_flags f
      ON  f.item_id  = p.item_id
      AND f.store_id = p.store_id
      AND p.d BETWEEN f.run_start_d AND f.run_end_d
    WHERE f.item_id IS NULL
),

-- 1. Day of week --------------------------------------------------------------
dow AS (
    SELECT
        'day_of_week'                                        AS dimension,
        cat_id,
        weekday                                              AS segment,
        wday                                                 AS sort_key,
        avg(units)                                           AS mean_units,
        count(*)                                             AS n_obs
    FROM clean
    GROUP BY 1, 2, 3, 4
),

-- 2. Month --------------------------------------------------------------------
mth AS (
    SELECT
        'month'                                              AS dimension,
        cat_id,
        lpad(month::VARCHAR, 2, '0')                         AS segment,
        month                                                AS sort_key,
        avg(units)                                           AS mean_units,
        count(*)                                             AS n_obs
    FROM clean
    GROUP BY 1, 2, 3, 4
),

-- 3. SNAP benefit days --------------------------------------------------------
snap AS (
    SELECT
        'snap'                                               AS dimension,
        cat_id,
        CASE snap WHEN 1 THEN 'snap_day' ELSE 'non_snap_day' END AS segment,
        snap                                                 AS sort_key,
        avg(units)                                           AS mean_units,
        count(*)                                             AS n_obs
    FROM clean
    GROUP BY 1, 2, 3, 4
),

-- 4. Named events -------------------------------------------------------------
evt AS (
    SELECT
        'event'                                              AS dimension,
        cat_id,
        coalesce(event_name_1, 'no_event')                   AS segment,
        0                                                    AS sort_key,
        avg(units)                                           AS mean_units,
        count(*)                                             AS n_obs
    FROM clean
    GROUP BY 1, 2, 3, 4
),

stacked AS (
    SELECT * FROM dow
    UNION ALL SELECT * FROM mth
    UNION ALL SELECT * FROM snap
    UNION ALL SELECT * FROM evt
)

SELECT
    dimension,
    cat_id,
    segment,
    n_obs,
    round(mean_units, 4)                                                AS mean_units,
    -- Baseline is the category's own mean across the whole dimension.
    round(avg(mean_units) OVER (PARTITION BY dimension, cat_id), 4)     AS category_baseline,
    round(mean_units / avg(mean_units) OVER (PARTITION BY dimension, cat_id), 3) AS lift,
    round(100.0 * (mean_units / avg(mean_units) OVER (PARTITION BY dimension, cat_id) - 1), 1)
                                                                        AS pct_vs_baseline,
    rank() OVER (PARTITION BY dimension, cat_id ORDER BY mean_units DESC) AS rank_in_dimension
FROM stacked
-- There are ~30 named events and most are rare, so keep only the strongest few
-- per category. This is a filter on a WINDOW result (the rank computed in the
-- SELECT), which is exactly what QUALIFY is for - a HAVING cannot see it and a
-- subquery would be needed otherwise.
QUALIFY dimension <> 'event'
     OR rank() OVER (PARTITION BY dimension, cat_id ORDER BY mean_units DESC) <= 12
ORDER BY dimension, cat_id, sort_key, mean_units DESC;
