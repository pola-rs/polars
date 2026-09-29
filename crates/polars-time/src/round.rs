use polars_arrow::legacy::time_zone::Tz;
use polars_arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_defs::time::duration::Duration;
use polars_utils::cache::LruCache;

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
                let every_parsed = Duration::try_parse(every)?;
                if every_parsed.negative {
                    polars_bail!(ComputeError: "cannot round a Datetime to a negative duration")
                }
                if (time_zone.is_none() || time_zone == &Some(TimeZone::UTC))
                    && (every_parsed.months() == 0 && every_parsed.weeks() == 0)
                {
                    // ... yes we can! Weeks, months, and time zones require extra logic.
                    // But in this simple case, it's just simple integer arithmetic.
                    let every = every_parsed.duration(self.time_unit());
                    return Ok(self
                        .physical()
                        .apply_values(|t| fast_round(t, every))
                        .into_datetime(self.time_unit(), time_zone.clone()));
                } else {
                    let w = Window::new(every_parsed, every_parsed, offset);
                    let out = self
                        .physical()
                        .try_apply_nonnull_values_generic(|t| w.round(self.time_unit(), t, tz));
                    return Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()));
                }
            } else {
                return Ok(Int64Chunked::full_null(self.name().clone(), self.len())
                    .into_datetime(self.time_unit(), self.time_zone().clone()));
            }
        }

        polars_ensure!(
            self.len() == every.len() || self.len() == 1,
            length_mismatch = "dt.round",
            self.len(),
            every.len()
        );

        // A sqrt(n) cache is not too small, not too large.
        let mut duration_cache = LruCache::with_capacity((every.len() as f64).sqrt() as usize);

        let out = broadcast_try_binary_elementwise(
            self.physical(),
            every,
            |opt_timestamp, opt_every| match (opt_timestamp, opt_every) {
                (Some(timestamp), Some(every)) => {
                    let every = *duration_cache.get_or_insert_with(every, Duration::parse);

                    if every.negative {
                        polars_bail!(ComputeError: "cannot round a Datetime to a negative duration")
                    }

                    let w = Window::new(every, every, offset);
                    w.round(self.time_unit(), timestamp, tz).map(Some)
                },
                _ => Ok(None),
            },
        );
        Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsRound for DateChunked {
    fn round(&self, every: &StringChunked, _tz: Option<&Tz>) -> PolarsResult<Self> {
        let offset = Duration::new(0);
        let out = match every.len() {
            1 => {
                if let Some(every) = every.get(0) {
                    let every = Duration::try_parse(every)?;
                    if every.negative {
                        polars_bail!(ComputeError: "cannot round a Date to a negative duration")
                    }
                    let w = Window::new(every, every, offset);
                    self.physical().try_apply_nonnull_values_generic(|t| {
                        Ok((w.round(
                            TimeUnit::Milliseconds,
                            MILLISECONDS_IN_DAY * t as i64,
                            None,
                        )? / MILLISECONDS_IN_DAY) as i32)
                    })
                } else {
                    Ok(Int32Chunked::full_null(self.name().clone(), self.len()))
                }
            },
            _ => {
                polars_ensure!(
                    self.len() == every.len() || self.len() == 1,
                    length_mismatch = "dt.round",
                    self.len(),
                    every.len()
                );
                broadcast_try_binary_elementwise(self.physical(), every, |opt_t, opt_every| {
                    // A sqrt(n) cache is not too small, not too large.
                    let mut duration_cache =
                        LruCache::with_capacity((every.len() as f64).sqrt() as usize);
                    match (opt_t, opt_every) {
                        (Some(t), Some(every)) => {
                            let every = *duration_cache.get_or_insert_with(every, Duration::parse);

                            if every.negative {
                                polars_bail!(ComputeError: "cannot round a Date to a negative duration")
                            }

                            let w = Window::new(every, every, offset);
                            Ok(Some(
                                (w.round(
                                    TimeUnit::Milliseconds,
                                    MILLISECONDS_IN_DAY * t as i64,
                                    None,
                                )? / MILLISECONDS_IN_DAY) as i32,
                            ))
                        },
                        _ => Ok(None),
                    }
                })
            },
        };
        Ok(out?.into_date())
    }
}
