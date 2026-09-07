-- ---------------------------------------------------------------------------
-- core.series_class - one row per SKU-store series with the segmentation that
-- every later layer routes on.
--
-- Three orthogonal classifications, because they answer different questions:
--
--   ABC  (revenue)      how much does getting this series wrong COST?
--   XYZ  (variability)  how predictable is it?
--   SB   (Syntetos-Boylan, ADI x CV^2)  which MODEL FAMILY does it need?
--
-- The SB class is the one with teeth. Profiling the M5 panel over availability
-- windows gives 72.5% intermittent, 18.4% lumpy, 6.2% smooth, 2.9% erratic -
-- so roughly 91% of series are intermittent or lumpy. Fitting ETS or ARIMA to
-- those is a category error; they need Croston / TSB. Routing by this column is
-- what stops the model ladder from averaging a good method over a bad one.
--
-- Everything is computed over the AVAILABILITY WINDOW only (core.panel), never
-- over the raw fact table. See sql/transform/010_panel.sql for why.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE core.series_class AS
WITH base AS (
    SELECT
        item_id,
        store_id,
        any_value(dept_id)                      AS dept_id,
        any_value(cat_id)                       AS cat_id,
        any_value(state_id)                     AS state_id,

        count(*)                                AS available_days,
        count(*) FILTER (WHERE units > 0)       AS selling_days,
        sum(units)                              AS total_units,
        sum(revenue)                            AS total_revenue,
        avg(units)                              AS mean_units,
        stddev_samp(units)                      AS sd_units,

        -- Croston-style decomposition: interval between demands, and the size
        -- of a demand WHEN it occurs. CV^2 is on demand SIZE, not on the raw
        -- series - that is the Syntetos-Boylan definition and mixing them up
        -- silently misclassifies most of the catalogue.
        avg(units) FILTER (WHERE units > 0)     AS mean_demand_size,
        stddev_samp(units) FILTER (WHERE units > 0) AS sd_demand_size,

        avg(sell_price)                         AS mean_price,
        min(d)                                  AS launch_d,
        max(d)                                  AS last_available_d
    FROM core.panel
    GROUP BY 1, 2
),
derived AS (
    SELECT
        *,
        available_days::DOUBLE / nullif(selling_days, 0)          AS adi,
        pow(sd_demand_size / nullif(mean_demand_size, 0), 2)      AS cv2_size,
        sd_units / nullif(mean_units, 0)                          AS cv_units
    FROM base
),
ranked AS (
    SELECT
        *,
        sum(total_revenue) OVER (
            ORDER BY total_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / sum(total_revenue) OVER ()                            AS cum_revenue_share,
        row_number() OVER (ORDER BY total_revenue DESC)           AS revenue_rank
    FROM derived
)
SELECT
    item_id,
    store_id,
    dept_id,
    cat_id,
    state_id,

    available_days,
    selling_days,
    total_units,
    total_revenue,
    mean_units,
    mean_price,
    launch_d,
    last_available_d,

    adi,
    cv2_size,
    cv_units,
    revenue_rank,
    cum_revenue_share,

    -- ABC on cumulative revenue contribution.
    CASE
        WHEN cum_revenue_share <= 0.80 THEN 'A'
        WHEN cum_revenue_share <= 0.95 THEN 'B'
        ELSE 'C'
    END AS abc_class,

    -- XYZ on coefficient of variation of the daily series.
    CASE
        WHEN cv_units IS NULL     THEN 'Z'
        WHEN cv_units <  0.5      THEN 'X'
        WHEN cv_units <= 1.0      THEN 'Y'
        ELSE 'Z'
    END AS xyz_class,

    -- Syntetos-Boylan cutoffs: ADI = 1.32, CV^2 = 0.49.
    CASE
        WHEN adi IS NULL                        THEN 'never_sold'
        WHEN adi <  1.32 AND cv2_size <  0.49   THEN 'smooth'
        WHEN adi <  1.32 AND cv2_size >= 0.49   THEN 'erratic'
        WHEN adi >= 1.32 AND cv2_size <  0.49   THEN 'intermittent'
        ELSE 'lumpy'
    END AS sb_class

FROM ranked;
