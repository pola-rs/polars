use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_utils::cache::LruCache;

use crate::prelude::*;

pub trait PolarsTruncate {
    fn truncate(&self, tz: Option<&Tz>, every: &StringChunked) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[inline(always)]
pub(crate) fn fast_truncate(t: i64, every: i64) -> i64 {
    let remainder = t % every;
    t - (remainder + every * (remainder < 0) as i64)
}

/// Whether truncating by `every` in `time_zone` maps sorted input to sorted output.
///
/// Truncation floors each timestamp on the *local* clock, whereas the sorted flag describes the
/// underlying instants. Those only stay in step if:
/// - the local clock is monotone in the instants, which holds for tz-naive and UTC input but not
///   for zones that change offset (`Asia/Magadan` moved +12 -> +10 on 2014-10-26, so two ascending
///   instants can truncate to two descending ones), and
/// - every row is truncated to the same window, as a per-row `every` can floor a later timestamp
///   into an earlier bucket.
///
/// See <https://github.com/pola-rs/polars/issues/28560>.
fn preserves_order(time_zone: Option<&TimeZone>, every: &StringChunked) -> bool {
    let single_window = every.len() == 1 && every.get(0).is_some();
    single_window && (time_zone.is_none() || time_zone == Some(&TimeZone::UTC))
}

impl PolarsTruncate for DatetimeChunked {
    fn truncate(&self, tz: Option<&Tz>, every: &StringChunked) -> PolarsResult<Self> {
        let mut out = truncate_datetime(self, tz, every)?;
        if preserves_order(self.time_zone().as_ref(), every) {
            out.physical_mut()
                .set_sorted_flag(self.physical().is_sorted_flag());
        }
        Ok(out)
    }
}

impl PolarsTruncate for DateChunked {
    fn truncate(&self, _tz: Option<&Tz>, every: &StringChunked) -> PolarsResult<Self> {
        let mut out = truncate_date(self, every)?;
        // A `Date` has no time zone, so only a per-row `every` can break ordering.
        if preserves_order(None, every) {
            out.physical_mut()
                .set_sorted_flag(self.physical().is_sorted_flag());
        }
        Ok(out)
    }
}

fn truncate_datetime(
    ca: &DatetimeChunked,
    tz: Option<&Tz>,
    every: &StringChunked,
) -> PolarsResult<DatetimeChunked> {
    polars_ensure!(
        ca.len() == every.len() || ca.len() == 1 || every.len() == 1,
        length_mismatch = "dt.truncate",
        ca.len(),
        every.len()
    );

    let time_zone = ca.time_zone();
    let offset = Duration::new(0);

    // Let's check if we can use a fastpath...
    if every.len() == 1 {
        if let Some(every) = every.get(0) {
            let every_parsed = Duration::try_parse(every)?;
            if every_parsed.negative {
                polars_bail!(ComputeError: "cannot truncate a Datetime to a negative duration")
            }
            if (time_zone.is_none() || time_zone.as_ref() == Some(&TimeZone::UTC))
                && (every_parsed.months() == 0 && every_parsed.weeks() == 0)
            {
                // ... yes we can! Weeks, months, and time zones require extra logic.
                // But in this simple case, it's just simple integer arithmetic.
                let every = match ca.time_unit() {
                    TimeUnit::Milliseconds => every_parsed.duration_ms(),
                    TimeUnit::Microseconds => every_parsed.duration_us(),
                    TimeUnit::Nanoseconds => every_parsed.duration_ns(),
                };
                if every == 0 {
                    return Ok(ca.clone());
                }
                return Ok(ca
                    .physical()
                    .apply_values(|t| fast_truncate(t, every))
                    .into_datetime(ca.time_unit(), time_zone.clone()));
            } else {
                let w = Window::new(every_parsed, every_parsed, offset);
                let out = match ca.time_unit() {
                    TimeUnit::Milliseconds => ca
                        .physical()
                        .try_apply_nonnull_values_generic(|t| w.truncate_ms(t, tz)),
                    TimeUnit::Microseconds => ca
                        .physical()
                        .try_apply_nonnull_values_generic(|t| w.truncate_us(t, tz)),
                    TimeUnit::Nanoseconds => ca
                        .physical()
                        .try_apply_nonnull_values_generic(|t| w.truncate_ns(t, tz)),
                };
                return Ok(out?.into_datetime(ca.time_unit(), ca.time_zone().clone()));
            }
        } else {
            return Ok(Int64Chunked::full_null(ca.name().clone(), ca.len())
                .into_datetime(ca.time_unit(), ca.time_zone().clone()));
        }
    }

    // A sqrt(n) cache is not too small, not too large.
    let mut duration_cache = LruCache::with_capacity((every.len() as f64).sqrt() as usize);

    let func = match ca.time_unit() {
        TimeUnit::Nanoseconds => Window::truncate_ns,
        TimeUnit::Microseconds => Window::truncate_us,
        TimeUnit::Milliseconds => Window::truncate_ms,
    };

    let out = broadcast_try_binary_elementwise(ca.physical(), every, |opt_timestamp, opt_every| {
        match (opt_timestamp, opt_every) {
            (Some(timestamp), Some(every)) => {
                let every = *duration_cache.try_get_or_insert_with(every, Duration::try_parse)?;

                if every.negative {
                    polars_bail!(ComputeError: "cannot truncate a Datetime to a negative duration")
                }

                let w = Window::new(every, every, offset);
                func(&w, timestamp, tz).map(Some)
            },
            _ => Ok(None),
        }
    });
    Ok(out?.into_datetime(ca.time_unit(), ca.time_zone().clone()))
}

fn truncate_date(ca: &DateChunked, every: &StringChunked) -> PolarsResult<DateChunked> {
    polars_ensure!(
        ca.len() == every.len() || ca.len() == 1 || every.len() == 1,
        length_mismatch = "dt.truncate",
        ca.len(),
        every.len()
    );

    let offset = Duration::new(0);
    let out = match every.len() {
        1 => {
            if let Some(every) = every.get(0) {
                let every = Duration::try_parse(every)?;
                if every.negative {
                    polars_bail!(ComputeError: "cannot truncate a Date to a negative duration")
                }
                let w = Window::new(every, every, offset);
                ca.physical().try_apply_nonnull_values_generic(|t| {
                    Ok(
                        (w.truncate_ms(MILLISECONDS_IN_DAY * t as i64, None)? / MILLISECONDS_IN_DAY)
                            as i32,
                    )
                })
            } else {
                Ok(Int32Chunked::full_null(ca.name().clone(), ca.len()))
            }
        },
        _ => broadcast_try_binary_elementwise(ca.physical(), every, |opt_t, opt_every| {
            // A sqrt(n) cache is not too small, not too large.
            let mut duration_cache = LruCache::with_capacity((every.len() as f64).sqrt() as usize);
            match (opt_t, opt_every) {
                (Some(t), Some(every)) => {
                    let every =
                        *duration_cache.try_get_or_insert_with(every, Duration::try_parse)?;

                    if every.negative {
                        polars_bail!(ComputeError: "cannot truncate a Date to a negative duration")
                    }

                    let w = Window::new(every, every, offset);
                    Ok(Some(
                        (w.truncate_ms(MILLISECONDS_IN_DAY * t as i64, None)? / MILLISECONDS_IN_DAY)
                            as i32,
                    ))
                },
                _ => Ok(None),
            }
        }),
    };
    Ok(out?.into_date())
}
