-- ---------------------------------------------------------------------------
-- ABC x XYZ 9-box segmentation.
--
-- ABC splits the catalogue by how much REVENUE is at stake (A = the series
-- carrying the first 80% of cumulative revenue, B = to 95%, C = the tail).
-- XYZ splits it by how PREDICTABLE demand is (coefficient of variation of daily
-- units: X < 0.5, Y <= 1.0, Z above).
--
-- The two together drive the inventory policy, which is why this is not a
-- decorative chart:
--
--   AX  high value, predictable    -> tight safety stock, high service target
--   AZ  high value, erratic        -> the expensive box; needs the widest
--                                     forecast quantiles and the most stock
--   CZ  low value, erratic         -> do not spend optimizer effort here;
--                                     a simple reorder rule is enough
--
-- Computed over availability windows only (core.series_class reads core.panel).
-- ---------------------------------------------------------------------------

WITH grid AS (
    SELECT
        abc_class,
        xyz_class,
        count(*)                                    AS series,
        sum(total_units)                            AS total_units,
        sum(total_revenue)                          AS revenue,
        avg(cv_units)                               AS avg_cv,
        avg(adi)                                    AS avg_adi,
        count(*) FILTER (WHERE sb_class IN ('intermittent', 'lumpy')) AS n_intermittent_or_lumpy
    FROM core.series_class
    GROUP BY 1, 2
)
SELECT
    abc_class,
    xyz_class,
    abc_class || xyz_class                                          AS box,
    series,
    round(100.0 * series / sum(series) OVER (), 2)                  AS pct_series,
    round(revenue, 0)                                               AS revenue,
    round(100.0 * revenue / sum(revenue) OVER (), 2)                AS pct_revenue,
    -- Revenue concentration: how much more revenue per series than average.
    round((revenue / series) / (sum(revenue) OVER () / sum(series) OVER ()), 2)
                                                                    AS revenue_intensity,
    round(avg_cv, 2)                                                AS avg_cv,
    round(avg_adi, 2)                                               AS avg_adi,
    round(100.0 * n_intermittent_or_lumpy / series, 1)              AS pct_intermittent_or_lumpy,
    CASE
        WHEN abc_class = 'A' AND xyz_class = 'X' THEN 'tight safety stock, high service'
        WHEN abc_class = 'A' AND xyz_class = 'Z' THEN 'widest quantiles, most stock'
        WHEN abc_class = 'A'                     THEN 'core optimizer focus'
        WHEN abc_class = 'C' AND xyz_class = 'Z' THEN 'simple rule, do not over-engineer'
        ELSE 'standard policy'
    END                                                             AS policy_hint
FROM grid
ORDER BY abc_class, xyz_class;
