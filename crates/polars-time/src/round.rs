use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_utils::cache::FastFixedCache;

use crate::prelude::*;
use crate::truncate::fast_truncate;

#[inline(always)]
fn fast_round(t: i64, every: i64) -> i64 {
    fast_truncate(t + every / 2, every)
}

pub trait PolarsRound {
    fn round(&self, every: &StringChunked, tz: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsRound for DatetimeChunked {
    fn round(&self, every: &StringChunked, tz: Option<&Tz>) -> PolarsResult<Self> {
        let time_zone = self.time_zone();
        let offset = Duration::new(0);

        // Let's check if we can use a fastpath...
        if every.len() == 1 {
            if let Some(every) = every.get(0) {
                let every_parsed = Duration::parse(every);
                if every_parsed.negative {
                    polars_bail!(ComputeError: "cannot round a Datetime to a negative duration")
                }
                if (time_zone.is_none() || time_zone.as_deref() == Some("UTC"))
                    && (every_parsed.months() == 0 && every_parsed.weeks() == 0)
                {
                    // ... yes we can! Weeks, months, and time zones require extra logic.
                    // But in this simple case, it's just simple integer arithmetic.
                    let every = match self.time_unit() {
                        TimeUnit::Milliseconds => every_parsed.duration_ms(),
                        TimeUnit::Microseconds => every_parsed.duration_us(),
                        TimeUnit::Nanoseconds => every_parsed.duration_ns(),
                    };
                    return Ok(self
                        .apply_values(|t| fast_round(t, every))
                        .into_datetime(self.time_unit(), time_zone.clone()));
                } else {
                    let w = Window::new(every_parsed, every_parsed, offset);
                    let out = match self.time_unit() {
                        TimeUnit::Milliseconds => {
                            self.try_apply_nonnull_values_generic(|t| w.round_ms(t, tz))
                        },
                        TimeUnit::Microseconds => {
                            self.try_apply_nonnull_values_generic(|t| w.round_us(t, tz))
                        },
                        TimeUnit::Nanoseconds => {
                            self.try_apply_nonnull_values_generic(|t| w.round_ns(t, tz))
                        },
                    };
                    return Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()));
                }
            } else {
                return Ok(Int64Chunked::full_null(self.name(), self.len())
                    .into_datetime(self.time_unit(), self.time_zone().clone()));
            }
        }

        // A sqrt(n) cache is not too small, not too large.
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::round_ns,
            TimeUnit::Microseconds => Window::round_us,
            TimeUnit::Milliseconds => Window::round_ms,
        };

        let out = broadcast_try_binary_elementwise(self, every, |opt_timestamp, opt_every| match (
            opt_timestamp,
            opt_every,
        ) {
            (Some(timestamp), Some(every)) => {
                let every =
                    *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));

                if every.negative {
                    polars_bail!(ComputeError: "cannot round a Datetime to a negative duration")
                }

                let w = Window::new(every, every, offset);
                func(&w, timestamp, tz).map(Some)
            },
            _ => Ok(None),
        });
        Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsRound for DateChunked {
    fn round(&self, every: &StringChunked, _tz: Option<&Tz>) -> PolarsResult<Self> {
        let offset = Duration::new(0);
        let out = match every.len() {
            1 => {
                if let Some(every) = every.get(0) {
                    let every = Duration::parse(every);
                    if every.negative {
                        polars_bail!(ComputeError: "cannot round a Date to a negative duration")
                    }
                    let w = Window::new(every, every, offset);
                    self.try_apply_nonnull_values_generic(|t| {
                        Ok(
                            (w.round_ms(MILLISECONDS_IN_DAY * t as i64, None)?
                                / MILLISECONDS_IN_DAY) as i32,
                        )
                    })
                } else {
                    Ok(Int32Chunked::full_null(self.name(), self.len()))
                }
            },
            _ => broadcast_try_binary_elementwise(self, every, |opt_t, opt_every| {
                // A sqrt(n) cache is not too small, not too large.
                let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);
                match (opt_t, opt_every) {
                    (Some(t), Some(every)) => {
                        let every = *duration_cache
                            .get_or_insert_with(every, |every| Duration::parse(every));

                        if every.negative {
                            polars_bail!(ComputeError: "cannot round a Date to a negative duration")
                        }

                        let w = Window::new(every, every, offset);
                        Ok(Some(
                            (w.round_ms(MILLISECONDS_IN_DAY * t as i64, None)?
                                / MILLISECONDS_IN_DAY) as i32,
                        ))
                    },
                    _ => Ok(None),
                }
            }),
        };
        Ok(out?.into_date())
    }
}
