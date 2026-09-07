-- ---------------------------------------------------------------------------
-- Star schema for the supply chain warehouse (DuckDB).
--
-- Grain decisions, stated up front because everything downstream depends on
-- them:
--   fact_sales    one row per (item_id, store_id, day)      ~59.2M rows
--   fact_price    one row per (item_id, store_id, week)     ~6.8M rows
--   fact_weather  one row per (state_id, date)              ~6.2K rows
--   fact_macro    one row per (series_id, date), native freq
--
-- Price is weekly in the source and we deliberately do NOT upsample it here.
-- Broadcasting a weekly price to daily is a modelling choice with leakage
-- implications, so it happens in the feature layer where the point-in-time
-- rules live.
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS core;

-- ---------------------------------------------------------------------------
-- Dimensions
-- ---------------------------------------------------------------------------

-- The M5 day index `d` (1..1941) is the natural key everywhere in the source,
-- so we keep it as the PK and carry the real calendar date alongside it.
CREATE OR REPLACE TABLE core.dim_date (
    d              SMALLINT   NOT NULL,
    date           DATE       NOT NULL,
    wm_yr_wk       INTEGER    NOT NULL,   -- Walmart fiscal week, joins to price
    weekday        VARCHAR    NOT NULL,
    wday           TINYINT    NOT NULL,   -- 1=Saturday in the M5 convention
    month          TINYINT    NOT NULL,
    year           SMALLINT   NOT NULL,
    event_name_1   VARCHAR,
    event_type_1   VARCHAR,
    event_name_2   VARCHAR,
    event_type_2   VARCHAR,
    snap_CA        TINYINT    NOT NULL,   -- SNAP benefit day flags, per state
    snap_TX        TINYINT    NOT NULL,
    snap_WI        TINYINT    NOT NULL,
    PRIMARY KEY (d)
);

CREATE OR REPLACE TABLE core.dim_item (
    item_id   VARCHAR NOT NULL,
    dept_id   VARCHAR NOT NULL,
    cat_id    VARCHAR NOT NULL,
    PRIMARY KEY (item_id)
);

CREATE OR REPLACE TABLE core.dim_store (
    store_id  VARCHAR NOT NULL,
    state_id  VARCHAR NOT NULL,
    PRIMARY KEY (store_id)
);

-- ---------------------------------------------------------------------------
-- Facts
-- ---------------------------------------------------------------------------

-- No PRIMARY KEY on fact_sales on purpose. DuckDB materialises an ART index
-- for a PK, and a 3-column index over 59.2M rows costs real memory on a laptop
-- for no query benefit here (every access path is a full or filtered scan).
-- Uniqueness of the grain is asserted by tests/test_warehouse.py instead, which
-- is the honest place for it: the constraint is still checked, just not paid
-- for on every insert.
CREATE OR REPLACE TABLE core.fact_sales (
    item_id   VARCHAR  NOT NULL,
    store_id  VARCHAR  NOT NULL,
    d         SMALLINT NOT NULL,
    units     SMALLINT NOT NULL   -- observed SALES, not demand; censored during stockouts
);

CREATE OR REPLACE TABLE core.fact_price (
    item_id     VARCHAR NOT NULL,
    store_id    VARCHAR NOT NULL,
    wm_yr_wk    INTEGER NOT NULL,
    sell_price  REAL    NOT NULL
);

CREATE OR REPLACE TABLE core.fact_weather (
    state_id                 VARCHAR NOT NULL,
    date                     DATE    NOT NULL,
    weather_city             VARCHAR NOT NULL,  -- representative metro, not the true store city
    temperature_max          REAL,
    temperature_min          REAL,
    temperature_mean         REAL,
    apparent_temperature_max REAL,
    precipitation_sum        REAL,
    snowfall_sum             REAL,
    windspeed_max            REAL,
    PRIMARY KEY (state_id, date)
);

-- Long/EAV shape because the series have different frequencies (monthly CPI,
-- weekly gas price) and different scopes (national vs per-state). Pivoting to
-- wide happens in the feature layer after forward-fill.
CREATE OR REPLACE TABLE core.fact_macro (
    series_id  VARCHAR NOT NULL,
    label      VARCHAR NOT NULL,
    state_id   VARCHAR,               -- NULL for national series
    freq       VARCHAR NOT NULL,
    date       DATE    NOT NULL,
    value      DOUBLE  NOT NULL,
    PRIMARY KEY (series_id, date)
);
