-- ---------------------------------------------------------------------------
-- core.stockout_flags - likely stockout days, so we do not train on censored
-- demand.
--
-- THE PROBLEM
-- -----------
-- fact_sales records what was SOLD, not what was DEMANDED. On a day a store ran
-- out, observed sales are zero while true demand was positive. Train on that
-- and the model learns to predict stockouts instead of demand: forecasts fall,
-- safety stock falls, and the optimizer systematically UNDER-orders. This is
-- assumption A2 in docs/03-assumptions-and-limitations.md and it is the single
-- most consequential data-quality issue in the project.
--
-- WHY NOT A FIXED THRESHOLD
-- -------------------------
-- The obvious rule - "zero for 7+ consecutive days = stockout" - is wrong in
-- both directions. Inside availability windows there are 6.6M zero runs with a
-- median length of 2 days, but the demand rate varies enormously across series:
-- 9.0% of series sell on most days (ADI < 1.32) while 7.7% sell less than once
-- in ten days (ADI > 10). For a fast mover, 7 straight zeros is a screaming
-- anomaly. For a slow mover, 7 straight zeros is a Tuesday.
--
-- THE RULE
-- --------
-- Estimate the daily probability of a sale, p. Under a working assumption of
-- day-to-day independence, a run of k consecutive zeros has probability
-- (1-p)^k. Flag the run when that falls below a significance level alpha:
--
--     (1 - p)^k < alpha   <=>   k > ln(alpha) / ln(1 - p)
--
-- The threshold is derived per series rather than imposed.
--
-- WHY p IS LOCAL, NOT LIFETIME  <-- this is the correction that matters
-- ----------------------------
-- The first version of this script used each series' LIFETIME sale rate. It
-- flagged 18.3% of the whole panel across 97.3% of series, with a mean flagged
-- run of 40.3 days. A 40-day stockout is not a stockout. Replenishment cycles
-- are days to weeks; a gap that long is a dead SKU or a seasonal product out of
-- season. The rule was measuring dormancy and calling it censoring.
--
-- The cause was the constant-p assumption. A series that sold daily in 2012 and
-- faded in 2015 carries a high lifetime p, so every genuine 2015 quiet stretch
-- looked impossible. Demand rate drifts; a lifetime average cannot see that.
--
-- So p is now estimated from the 56 days IMMEDIATELY PRECEDING each run. The
-- question becomes the one actually worth asking: "this series was selling at
-- rate p right up to this gap - how surprising is the gap?" Series with less
-- than 28 days of prior history fall back to the lifetime rate, flagged in
-- `rate_basis` so the fallback is never invisible.
--
-- WHY LONG RUNS ARE SEPARATED RATHER THAN DROPPED
-- -----------------------------------------------
-- Even with a local rate, a run can be long. Runs over MAX_STOCKOUT_DAYS (28)
-- are labelled 'dormant' instead of 'stockout'. Both are censored and neither
-- should be trained on as observed demand, but they need different treatment
-- downstream: a stockout should be imputed (demand existed, we could not see
-- it), while a dormant stretch should be excluded (the product was effectively
-- not on sale). Collapsing them would impute phantom demand into dead SKUs and
-- inflate every inventory recommendation that follows.
--
-- WHAT THIS STILL CANNOT DO
-- -------------------------
-- Independence remains false within the 56-day window - demand is autocorrelated
-- and day-of-week seasonal - so the rule stays somewhat over-eager. That is the
-- safer direction: excluding a genuine zero costs a little information, while
-- training on a censored zero biases every downstream cost number. It also
-- cannot detect a PARTIAL stockout, where the shelf emptied at noon and sales
-- were merely truncated. Those survive as understated positives and are not
-- recoverable from this data.
--
-- alpha = 0.005, window = 56 days and MAX_STOCKOUT_DAYS = 28 are choices, not
-- facts. sql/analytics/stockout_proxy_detection.sql reports sensitivity to all
-- three.
-- ---------------------------------------------------------------------------

-- Built in two steps on purpose. core.zero_runs holds EVERY zero run with its
-- per-run baseline attached; core.stockout_flags is the alpha-filtered subset.
-- Keeping the unfiltered table is what makes the sensitivity analysis in
-- sql/analytics/stockout_proxy_detection.sql actually mean something: a report
-- that recomputes thresholds over the already-filtered table can only ever
-- remove runs, so loosening alpha would silently return the same answer and
-- look like insensitivity to a parameter that in fact matters.
CREATE OR REPLACE TABLE core.zero_runs AS
WITH trail_stats AS (
    -- core.panel is contiguous in `d` within each series' availability window
    -- (verified: priced days == span between first and last priced day, for
    -- 100.00% of all 30,490 series), so a ROWS frame is a true 56-day window.
    SELECT
        item_id,
        store_id,
        d,
        units,
        lag(units) OVER (PARTITION BY item_id, store_id ORDER BY d) AS prev_units,
        sum(CASE WHEN units > 0 THEN 1 ELSE 0 END) OVER trail       AS trail_sell_days,
        count(*) OVER trail                                          AS trail_obs
    FROM core.panel
    WINDOW trail AS (
        PARTITION BY item_id, store_id ORDER BY d
        ROWS BETWEEN 56 PRECEDING AND 1 PRECEDING
    )
),
run_ids AS (
    -- A new run starts whenever the zero / non-zero state flips.
    SELECT
        item_id,
        store_id,
        d,
        units,
        trail_sell_days,
        trail_obs,
        sum(CASE WHEN (units = 0) IS DISTINCT FROM (prev_units = 0) THEN 1 ELSE 0 END)
            OVER (PARTITION BY item_id, store_id ORDER BY d
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run_id
    FROM trail_stats
),
zero_runs AS (
    SELECT
        item_id,
        store_id,
        run_id,
        count(*)                                  AS run_len,
        min(d)                                    AS run_start_d,
        max(d)                                    AS run_end_d,
        -- The trailing stats as they stood on the FIRST day of the run.
        arg_min(trail_sell_days, d)               AS trail_sell_days,
        arg_min(trail_obs, d)                     AS trail_obs
    FROM run_ids
    WHERE units = 0
    GROUP BY 1, 2, 3
),
lifetime AS (
    SELECT
        item_id,
        store_id,
        count(*)                                              AS available_days,
        count(*) FILTER (WHERE units > 0)                     AS selling_days,
        count(*) FILTER (WHERE units > 0)::DOUBLE / count(*)  AS p_lifetime,
        max(d)                                                AS last_available_d
    FROM core.panel
    GROUP BY 1, 2
),
scored AS (
    SELECT
        zr.item_id,
        zr.store_id,
        zr.run_start_d,
        zr.run_end_d,
        zr.run_len,
        lt.available_days,
        lt.selling_days,
        lt.p_lifetime,
        zr.trail_obs,
        -- Local rate when there is enough prior history, else lifetime.
        CASE WHEN zr.trail_obs >= 28
             THEN zr.trail_sell_days::DOUBLE / zr.trail_obs
             ELSE lt.p_lifetime
        END                                                   AS p_used,
        CASE WHEN zr.trail_obs >= 28 THEN 'local_56d' ELSE 'lifetime_fallback' END
                                                              AS rate_basis,
        (zr.run_end_d = lt.last_available_d)                  AS touches_series_end
    FROM zero_runs zr
    JOIN lifetime lt
      ON  lt.item_id  = zr.item_id
      AND lt.store_id = zr.store_id
)
SELECT
    item_id,
    store_id,
    run_start_d,
    run_end_d,
    run_len,
    p_used,
    rate_basis,
    p_lifetime,
    trail_obs,
    available_days,
    selling_days,
    min_run_len,
    pow(1.0 - p_used, run_len)                                AS run_probability,
    touches_series_end,
    CASE
        WHEN run_len <= 28 THEN 'stockout'   -- plausible replenishment failure: impute
        ELSE 'dormant'                       -- out of season or discontinued: exclude
    END AS censor_type
FROM (
    SELECT
        *,
        ceil(ln(0.005) / ln(nullif(1.0 - p_used, 0.0)))       AS min_run_len
    FROM scored
    WHERE p_used > 0    -- no baseline to test a run against
);

-- The flagged subset: runs long enough to be improbable at alpha = 0.005.
CREATE OR REPLACE TABLE core.stockout_flags AS
SELECT *
FROM core.zero_runs
WHERE run_len >= min_run_len;
