-- ---------------------------------------------------------------------------
-- Stockout proxy: diagnostic report and parameter sensitivity.
--
-- The detection itself lives in sql/transform/020_stockout_proxy.sql, which
-- materialises core.stockout_flags. This file is the report over it, plus the
-- sensitivity analysis that keeps the three tuning choices honest:
--
--     alpha = 0.005          significance level for "this run is too long"
--     window = 56 days       history used to estimate the local sale rate
--     MAX_STOCKOUT_DAYS = 28 above which a gap is dormancy, not a stockout
--
-- Section 4 recomputes the flag rate across a grid of alpha so the reader can
-- see how much the headline censoring rate depends on a number that was chosen
-- rather than measured.
-- ---------------------------------------------------------------------------

-- 1. Headline -----------------------------------------------------------------
WITH panel_size AS (
    SELECT count(*) AS panel_days FROM core.panel
),
headline AS (
    SELECT
        censor_type,
        count(*)                                       AS runs,
        sum(run_len)                                   AS censored_days,
        count(DISTINCT (item_id, store_id))            AS series_affected,
        avg(run_len)                                   AS avg_run_len,
        median(run_len)                                AS median_run_len,
        max(run_len)                                   AS max_run_len,
        count(*) FILTER (WHERE touches_series_end)     AS runs_at_series_end,
        count(*) FILTER (WHERE rate_basis = 'lifetime_fallback') AS runs_on_fallback_rate
    FROM core.stockout_flags
    GROUP BY 1
),

-- 2. Per-series stockout burden, joined to segmentation ------------------------
per_series AS (
    SELECT
        item_id,
        store_id,
        sum(run_len) FILTER (WHERE censor_type = 'stockout') AS stockout_days,
        sum(run_len) FILTER (WHERE censor_type = 'dormant')  AS dormant_days
    FROM core.stockout_flags
    GROUP BY 1, 2
),
by_segment AS (
    SELECT
        c.abc_class,
        c.sb_class,
        count(*)                                                     AS series,
        sum(c.available_days)                                        AS available_days,
        sum(coalesce(s.stockout_days, 0))                            AS stockout_days,
        sum(coalesce(s.dormant_days, 0))                             AS dormant_days,
        sum(c.total_revenue)                                         AS revenue
    FROM core.series_class c
    LEFT JOIN per_series s
      ON s.item_id = c.item_id AND s.store_id = c.store_id
    GROUP BY 1, 2
),

-- 3. Sensitivity to alpha ------------------------------------------------------
-- Recomputed against core.zero_runs - EVERY zero run, not the already-filtered
-- core.stockout_flags. Running this over the filtered table would let alpha only
-- ever remove runs, so every value looser than the build-time 0.005 would return
-- an identical answer and the parameter would look far more inert than it is.
alpha_grid AS (
    SELECT unnest([0.05, 0.01, 0.005, 0.001]) AS alpha
),
sensitivity AS (
    SELECT
        g.alpha,
        count(*) FILTER (
            WHERE z.run_len >= ceil(ln(g.alpha) / ln(nullif(1.0 - z.p_used, 0.0)))
              AND z.run_len <= 28
        )                                                            AS stockout_runs,
        sum(z.run_len) FILTER (
            WHERE z.run_len >= ceil(ln(g.alpha) / ln(nullif(1.0 - z.p_used, 0.0)))
              AND z.run_len <= 28
        )                                                            AS stockout_days
    FROM core.zero_runs z
    CROSS JOIN alpha_grid g
    GROUP BY 1
)

-- Report -----------------------------------------------------------------------
SELECT
    '1_headline'                                        AS section,
    h.censor_type                                       AS label,
    h.runs                                              AS metric_a,
    h.censored_days                                     AS metric_b,
    round(100.0 * h.censored_days / p.panel_days, 3)    AS pct_of_panel,
    h.series_affected                                   AS metric_c,
    round(h.avg_run_len, 1)                             AS metric_d,
    round(h.median_run_len, 1)                          AS metric_e,
    h.max_run_len                                       AS metric_f,
    'runs | censored_days | series | avg_len | median_len | max_len' AS legend
FROM headline h CROSS JOIN panel_size p

UNION ALL

SELECT
    '2_by_segment',
    s.abc_class || ' / ' || s.sb_class,
    s.series,
    s.stockout_days,
    round(100.0 * s.stockout_days / nullif(s.available_days, 0), 3),
    s.dormant_days,
    round(100.0 * s.dormant_days / nullif(s.available_days, 0), 1),
    NULL,
    NULL,
    'series | stockout_days | pct_days_stockout | dormant_days | pct_dormant'
FROM by_segment s

UNION ALL

SELECT
    '3_alpha_sensitivity',
    'alpha=' || sv.alpha::VARCHAR,
    sv.stockout_runs,
    sv.stockout_days,
    round(100.0 * sv.stockout_days / (SELECT panel_days FROM panel_size), 3),
    NULL, NULL, NULL, NULL,
    'runs | days | pct_of_panel at this alpha'
FROM sensitivity sv

ORDER BY section, label;
