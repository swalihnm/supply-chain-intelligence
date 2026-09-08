"""The predictive distribution of demand over a random lead time.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
Safety stock is a statement about the tail of demand over the lead time. The
forecast layer produces DAILY quantiles, and daily quantiles do not add:

    q95(D1 + D2 + ... + DL)  !=  q95(D1) + q95(D2) + ... + q95(DL)

Summing them overstates the tail badly (it assumes every day is simultaneously
at its 95th percentile), which would inflate safety stock across the entire
catalogue. So the lead-time distribution has to be constructed, not assembled.

WHY NOT THE TEXTBOOK NORMAL FORMULA
-----------------------------------
Every inventory textbook gives

    safety stock = z * sigma_demand * sqrt(lead_time)

which assumes demand is normal, i.i.d. across days, and that lead time is
constant. On this data all three are false:

  * 72.6% of series are intermittent - demand is zero-inflated and heavily
    right-skewed, nothing like a normal
  * demand is autocorrelated and strongly day-of-week seasonal, so sqrt(L)
    understates the variance of a run of days
  * lead time is not a constant

Applying the normal formula to an intermittent series produces negative
reorder points at low service levels and wildly wrong ones in the tail. This is
the "why did you not assume normality" question, and the answer is that the
assumption is not close enough to survive contact with 91% intermittent data.

THE METHOD: SCALED BLOCK BOOTSTRAP
----------------------------------
For each series, resample CONTIGUOUS blocks of its own recent observed demand:

  1. draw a lead time L from its distribution
  2. draw a random start point in the series' clean recent history
  3. take the next L days and sum them
  4. scale by (forecast level / historical level) so the draw is centred on what
     the model expects rather than on the past
  5. repeat n_sims times

Contiguous blocks are the point. Sampling individual days independently would
destroy the autocorrelation and day-of-week structure and understate the
variance of the sum - exactly the error the sqrt(L) formula makes. Blocks carry
that structure through for free, with no distributional assumption anywhere.

Censored days are excluded from the history being resampled, so a stockout
cannot be resampled as if it were genuine zero demand.

LIMITS
------
The scaling step assumes the SHAPE of the demand distribution is stable and only
its level moves. For a series whose variability is changing, not just its mean,
this understates the change. Series with too little clean history fall back to a
Poisson draw at the forecast rate, which is flagged in the output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sci.log import get_logger

log = get_logger(__name__)

# Lead time is ASSUMED, not measured - M5 has no supplier data. A 3-7 day
# uniform spread is typical for grocery DC-to-store replenishment.
LEAD_TIME_MIN = 3
LEAD_TIME_MAX = 7

MIN_HISTORY_DAYS = 90
DEFAULT_SIMS = 2000


@dataclass
class LeadTimeDemand:
    """Simulated lead-time demand, shaped (n_series, n_sims)."""

    samples: np.ndarray
    used_bootstrap: np.ndarray  # (n_series,) bool - False where Poisson fallback ran
    review_period: int
    lead_time_min: int
    lead_time_max: int

    def quantile(self, q: float) -> np.ndarray:
        """Per-series quantile of lead-time demand at one shared level."""
        return np.quantile(self.samples, q, axis=1)

    def quantile_per_series(self, qs: np.ndarray) -> np.ndarray:
        """A DIFFERENT quantile level for each series.

        Needed once the optimizer stops applying one uniform service level to
        the whole catalogue and starts choosing per series - which is the entire
        point of the allocation step.
        """
        srt = np.sort(self.samples, axis=1)
        n = srt.shape[1]
        idx = np.clip(np.round(np.asarray(qs) * (n - 1)).astype(np.int64), 0, n - 1)
        return srt[np.arange(srt.shape[0]), idx]

    def expected_shortfall(self, level: np.ndarray) -> np.ndarray:
        """E[(D - level)+] per series - expected unmet units in one cycle.

        Computed from the simulated distribution rather than a formula, so it
        inherits the skew and zero-inflation of the bootstrap instead of
        assuming a shape.
        """
        excess = self.samples - np.asarray(level)[:, None]
        return np.maximum(excess, 0.0).mean(axis=1)

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=1)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=1)

    def quantiles(self, qs: tuple[float, ...]) -> dict[float, np.ndarray]:
        return {q: self.quantile(q) for q in qs}


def simulate(
    history: np.ndarray,
    forecast_daily_mean: np.ndarray,
    review_period: int = 7,
    lead_time_min: int = LEAD_TIME_MIN,
    lead_time_max: int = LEAD_TIME_MAX,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 0,
    cover_review_period: bool = True,
) -> LeadTimeDemand:
    """Simulate demand over (review period + lead time) per series.

    history : (n_series, n_hist_days) clean demand, NaN where unusable
    forecast_daily_mean : (n_series,) expected daily demand over the horizon

    With `cover_review_period` the window is T + L, which is what a periodic
    review policy must protect against: after placing an order you wait L days
    for it AND another T days before you can order again. Protecting only L is
    the classic under-stocking mistake in periodic review systems.
    """
    rng = np.random.default_rng(seed)
    n_series, n_hist = history.shape

    valid = np.isfinite(history)
    n_valid = valid.sum(axis=1)
    hist_mean = np.divide(
        np.nansum(np.where(valid, history, 0.0), axis=1),
        np.maximum(n_valid, 1),
        out=np.zeros(n_series),
        where=n_valid > 0,
    )

    # Scale factor: move the historical shape onto the forecast level.
    # Clipped because a near-zero historical mean would otherwise produce an
    # explosive multiplier on a series the model expects to wake up.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(hist_mean > 1e-6, forecast_daily_mean / hist_mean, 1.0)
    scale = np.clip(np.nan_to_num(scale, nan=1.0), 0.1, 10.0)

    base = review_period if cover_review_period else 0
    max_window = base + lead_time_max

    can_bootstrap = n_valid >= max(MIN_HISTORY_DAYS, max_window + 1)
    samples = np.zeros((n_series, n_sims), dtype=np.float32)

    # Censored days inside a block are IMPUTED with the series' own clean mean,
    # not zero-filled. A block must stay contiguous to preserve autocorrelation,
    # so a censored day cannot simply be dropped - and filling it with zero
    # would resample a stockout as genuine zero demand, biasing lead-time demand
    # down and safety stock with it. That is precisely the censoring bias the
    # stockout proxy exists to remove, reintroduced at the last step.
    #
    # Pre-launch padding is never sampled, because block starts are bounded
    # below by each series' first valid day.
    hist0 = np.where(valid, np.nan_to_num(history, nan=0.0), hist_mean[:, None]).astype(np.float32)
    cs = np.zeros((n_series, n_hist + 1), dtype=np.float32)
    np.cumsum(hist0, axis=1, out=cs[:, 1:])

    lead_times = rng.integers(lead_time_min, lead_time_max + 1, size=(n_series, n_sims))
    windows = base + lead_times

    # Block start positions, one per simulation, leaving room for the longest
    # window. Starts are bounded BELOW by each series' first valid day: the
    # history array is zero-filled before a series launched, and sampling blocks
    # from that padding would resample structural zeros as if they were demand
    # and quietly drag every safety stock down.
    first_valid = np.argmax(np.isfinite(history), axis=1).astype(np.int32)
    span = np.maximum(n_hist - max_window - first_valid, 1)
    starts = first_valid[:, None] + (rng.random((n_series, n_sims)) * span[:, None]).astype(
        np.int32
    )
    # Hard bound so the prefix-sum lookup can never run past the array. Series
    # whose history starts too late to fit a window are clipped here and then
    # discarded anyway by `can_bootstrap` below - but the indexing happens for
    # every row, so the clip has to come first.
    starts = np.minimum(starts, max(n_hist - max_window, 0))

    rows = np.arange(n_series)[:, None]
    block_sum = cs[rows, starts + windows] - cs[rows, starts]
    samples[can_bootstrap] = (block_sum * scale[:, None])[can_bootstrap]

    # Fallback: Poisson at the forecast rate over the same window.
    fallback = ~can_bootstrap
    if fallback.any():
        lam = np.clip(forecast_daily_mean[fallback], 0, None)[:, None] * windows[fallback]
        samples[fallback] = rng.poisson(lam).astype(np.float32)
        log.info(
            "  lead-time demand: %d/%d series used the Poisson fallback (thin history)",
            int(fallback.sum()),
            n_series,
        )

    return LeadTimeDemand(
        samples=np.clip(samples, 0, None),
        used_bootstrap=can_bootstrap,
        review_period=review_period,
        lead_time_min=lead_time_min,
        lead_time_max=lead_time_max,
    )


def normal_approx_quantile(
    history: np.ndarray,
    forecast_daily_mean: np.ndarray,
    q: float,
    review_period: int = 7,
    mean_lead_time: float = (LEAD_TIME_MIN + LEAD_TIME_MAX) / 2,
) -> np.ndarray:
    """The textbook normal formula, implemented ONLY as a comparison baseline.

        SS = z * sigma_daily * sqrt(T + L)

    Kept so the results can quantify what assuming normality would have cost,
    rather than just asserting that it would have been wrong. Not used to set
    any policy.
    """
    from scipy.stats import norm

    z = norm.ppf(q)
    with np.errstate(invalid="ignore"):
        sigma = np.nanstd(history, axis=1)
    sigma = np.nan_to_num(sigma, nan=0.0)
    window = review_period + mean_lead_time
    return np.clip(forecast_daily_mean * window + z * sigma * np.sqrt(window), 0, None)
