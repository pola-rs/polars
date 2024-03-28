use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::chunked_array::ops::arity::try_binary_elementwise;
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsTruncate {
    fn truncate(&self, tz: Option<&Tz>, every: &StringChunked, offset: &str) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsTruncate for DatetimeChunked {
    fn truncate(&self, tz: Option<&Tz>, every: &StringChunked, offset: &str) -> PolarsResult<Self> {
        let offset = Duration::parse(offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Microseconds => Window::truncate_us,
            TimeUnit::Milliseconds => Window::truncate_ms,
        };

        let out = match every.len() {
            1 => {
                if let Some(every) = every.get(0) {
                    let every = Duration::parse(every);
                    if every.negative {
                        polars_bail!(ComputeError: "cannot truncate a Datetime to a negative duration")
                    }

                    let w = Window::new(every, every, offset);
                    self.0
                        .try_apply_nonnull_values_generic(|timestamp| func(&w, timestamp, tz))
                } else {
                    Ok(Int64Chunked::full_null(self.name(), self.len()))
                }
            },
            _ => try_binary_elementwise(self, every, |opt_timestamp, opt_every| {
                match (opt_timestamp, opt_every) {
                    (Some(timestamp), Some(every)) => {
                        let every = Duration::parse(every);
                        if every.negative {
                            polars_bail!(ComputeError: "cannot truncate a Datetime to a negative duration")
                        }

                        let w = Window::new(every, every, offset);
                        func(&w, timestamp, tz).map(Some)
                    },
                    _ => Ok(None),
                }
            }),
        };
        Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsTruncate for DateChunked {
    fn truncate(
        &self,
        _tz: Option<&Tz>,
        every: &StringChunked,
        offset: &str,
    ) -> PolarsResult<Self> {
        let offset = Duration::parse(offset);
        let out = match every.len() {
            1 => {
                if let Some(every) = every.get(0) {
                    let every = Duration::parse(every);
                    if every.negative {
                        polars_bail!(ComputeError: "cannot truncate a Date to a negative duration")
                    }

                    let w = Window::new(every, every, offset);
                    self.try_apply_nonnull_values_generic(|t| {
                        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                        Ok((w.truncate_ms(MSECS_IN_DAY * t as i64, None)? / MSECS_IN_DAY) as i32)
                    })
                } else {
                    Ok(Int32Chunked::full_null(self.name(), self.len()))
                }
            },
            _ => try_binary_elementwise(&self.0, every, |opt_t, opt_every| {
                match (opt_t, opt_every) {
                    (Some(t), Some(every)) => {
                        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                        let every = Duration::parse(every);
                        if every.negative {
                            polars_bail!(ComputeError: "cannot truncate a Date to a negative duration")
                        }

                        let w = Window::new(every, every, offset);
                        Ok(Some(
                            (w.truncate_ms(MSECS_IN_DAY * t as i64, None)? / MSECS_IN_DAY) as i32,
                        ))
                    },
                    _ => Ok(None),
                }
            }),
        };
        Ok(out?.into_date())
    }
}

#[cfg(feature = "dtype-duration")]
impl PolarsTruncate for DurationChunked {
    fn truncate(
        &self,
        _tz: Option<&Tz>,
        every: &StringChunked,
        offset: &str,
    ) -> PolarsResult<Self> {
        let to_time_unit = match self.time_unit() {
            TimeUnit::Nanoseconds => Duration::duration_ns,
            TimeUnit::Microseconds => Duration::duration_us,
            TimeUnit::Milliseconds => Duration::duration_ms,
        };

        let offset_duration = Duration::parse(offset);
        if !offset_duration.is_constant_duration() {
            polars_bail!(InvalidOperation:
                "Cannot offset a Duration series by a non-constant duration."
            );
        }
        let offset_units = if offset_duration.negative {
            -to_time_unit(&offset_duration)
        } else {
            to_time_unit(&offset_duration)
        };

        let out = if every.len() == 1 {
            if let Some(every) = every.get(0) {
                let every_duration = Duration::parse(every);
                if every_duration.negative {
                    polars_bail!(ComputeError: "cannot truncate a Duration to a negative duration")
                }
                if every_duration.is_constant_duration() {
                    let every_units = to_time_unit(&every_duration);

                    if every_units == 0 {
                        polars_bail!(InvalidOperation: "duration cannot be zero.")
                    }

                    Ok(self
                        .0
                        .apply_values(|duration| duration - duration % every_units + offset_units))
                } else {
                    polars_bail!(InvalidOperation:
                        "Cannot truncate a Duration series to a non-constant duration."
                    )
                }
            } else {
                Ok(Int64Chunked::full_null(self.name(), self.len()))
            }
        } else {
            try_binary_elementwise(self, every, |opt_duration, opt_every| {
                if let (Some(duration), Some(every)) = (opt_duration, opt_every) {
                    let every_duration = Duration::parse(every);
                    if every_duration.negative {
                        polars_bail!(ComputeError: "cannot truncate a Duration to a negative duration")
                    }
                    if every_duration.is_constant_duration() {
                        let every_units = to_time_unit(&every_duration);

                        Ok(Some(duration - duration % every_units + offset_units))
                    } else {
                        polars_bail!(InvalidOperation:
                            "Cannot truncate a Duration series to a non-constant duration."
                        )
                    }
                } else {
                    Ok(None)
                }
            })
        };
        out.map(|s| s.into_duration(self.time_unit()))
    }
}
