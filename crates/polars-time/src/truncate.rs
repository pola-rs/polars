use arrow::legacy::time_zone::Tz;
#[cfg(feature = "dtype-date")]
use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::chunked_array::ops::arity::try_binary_elementwise;
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsTruncate {
    fn truncate(&self, tz: Option<&Tz>, every: &StringChunked, offset: &str) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-datetime")]
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
                    let w = Window::new(every, every, offset);
                    self.0.try_apply(|timestamp| func(&w, timestamp, tz))
                } else {
                    Ok(Int64Chunked::full_null(self.name(), self.len()))
                }
            },
            _ => try_binary_elementwise(self, every, |opt_timestamp, opt_every| {
                match (opt_timestamp, opt_every) {
                    (Some(timestamp), Some(every)) => {
                        let every = Duration::parse(every);
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

#[cfg(feature = "dtype-date")]
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
                    let w = Window::new(every, every, offset);
                    self.try_apply(|t| {
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
