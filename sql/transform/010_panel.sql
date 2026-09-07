-- ---------------------------------------------------------------------------
-- core.panel - the analyst-facing daily panel.
--
-- One row per (item_id, store_id, day) RESTRICTED TO THE AVAILABILITY WINDOW,
-- joined to calendar, item, store and price attributes.
--
-- WHY THE AVAILABILITY FILTER IS THE MOST IMPORTANT LINE IN THIS FILE
-- -------------------------------------------------------------------
-- 20.78% of the raw 59.2M rows have no sell_price. Profiling established that
-- this is not missing data - it is a clean availability signal:
--
--   * the price-NULL block is contiguous and leading for 100.00% of all
--     30,490 series (checked, not assumed)
--   * ZERO rows exist where units > 0 but sell_price IS NULL
--   * median gap between first priced day and first sale is 2 days (p95 = 6)
--
-- So a NULL price means "this SKU was not carried by this store yet", not
-- "we lost the price". Those rows are structural zeros. Including them would
-- teach every model that ~21% of demand is zero when the product simply did
-- not exist, dragging forecasts - and therefore safety stock - down.
--
-- Median first-priced day is d=160 and 15,752 of 30,490 series (51.7%) do not
-- start until after d=100, so this is not a rounding-error correction.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE core.panel AS
WITH priced AS (
    SELECT
        s.item_id,
        s.store_id,
        s.d,
        s.units,
        p.sell_price
    FROM core.fact_sales s
    JOIN core.dim_date dt
      ON dt.d = s.d
    -- INNER JOIN is the availability filter. See the header.
    JOIN core.fact_price p
      ON  p.item_id  = s.item_id
      AND p.store_id = s.store_id
      AND p.wm_yr_wk = dt.wm_yr_wk
),
launch AS (
    SELECT item_id, store_id, min(d) AS launch_d, max(d) AS last_available_d
    FROM priced
    GROUP BY 1, 2
)
SELECT
    pr.item_id,
    pr.store_id,
    pr.d,
    dt.date,
    dt.wm_yr_wk,

    pr.units,
    pr.sell_price,
    pr.units * pr.sell_price          AS revenue,

    it.dept_id,
    it.cat_id,
    st.state_id,

    dt.wday,
    dt.weekday,
    dt.month,
    dt.year,
    dt.event_name_1,
    dt.event_type_1,

    -- SNAP is published as three per-state columns; resolve to the one that
    -- actually applies to this store.
    CASE st.state_id
        WHEN 'CA' THEN dt.snap_CA
        WHEN 'TX' THEN dt.snap_TX
        WHEN 'WI' THEN dt.snap_WI
    END::TINYINT                      AS snap,

    lc.launch_d,
    pr.d - lc.launch_d                AS days_since_launch,
    lc.last_available_d

FROM priced pr
JOIN core.dim_date  dt ON dt.d        = pr.d
JOIN core.dim_item  it ON it.item_id  = pr.item_id
JOIN core.dim_store st ON st.store_id = pr.store_id
JOIN launch         lc ON lc.item_id  = pr.item_id AND lc.store_id = pr.store_id;
