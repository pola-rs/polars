use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_utils::cache::FastFixedCache;

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

        // A sqrt(n) cache is not too small, not too large.
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);
        let out = broadcast_try_binary_elementwise(self, every, |opt_timestamp, opt_every| match (
            opt_timestamp,
            opt_every,
        ) {
            (Some(timestamp), Some(every)) => {
                let every =
                    *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));

                if every.negative {
                    polars_bail!(ComputeError: "cannot truncate a Datetime to a negative duration")
                }

                let w = Window::new(every, every, offset);
                func(&w, timestamp, tz).map(Some)
            },
            _ => Ok(None),
        });
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
        // A sqrt(n) cache is not too small, not too large.
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);
        let out = broadcast_try_binary_elementwise(&self.0, every, |opt_t, opt_every| {
            match (opt_t, opt_every) {
                (Some(t), Some(every)) => {
                    const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                    let every =
                        *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));
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
        });
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
        let to_i64 = match self.time_unit() {
            TimeUnit::Nanoseconds => Duration::duration_ns,
            TimeUnit::Microseconds => Duration::duration_us,
            TimeUnit::Milliseconds => Duration::duration_ms,
        };
        polars_ensure!(
            Duration::parse(offset).is_zero(),
            InvalidOperation: "`offset` is not supported for truncating Durations."
        );

        let out = if every.len() == 1 {
            if let Some(every) = every.get(0) {
                let every_duration = Duration::parse(every);

                polars_ensure!(
                    !every_duration.negative,
                    ComputeError: "cannot truncate a Duration to a negative duration"
                );
                ensure_is_constant_duration(every_duration, None, "every")?;
                let every_units = to_i64(&every_duration);
                polars_ensure!(
                    every_units != 0,
                    InvalidOperation: "`every` duration cannot be zero"
                );

                Ok(self
                    .0
                    .apply_values(|duration| duration - duration % every_units))
            } else {
                Ok(Int64Chunked::full_null(self.name(), self.len()))
            }
        } else {
            try_binary_elementwise(self, every, |opt_duration, opt_every| {
                if let (Some(duration), Some(every)) = (opt_duration, opt_every) {
                    let every_duration = Duration::parse(every);

                    polars_ensure!(
                        !every_duration.negative,
                        ComputeError: "cannot truncate a Duration to a negative duration"
                    );
                    ensure_is_constant_duration(every_duration, None, "every")?;
                    let every_units = to_i64(&every_duration);
                    polars_ensure!(
                        every_units != 0,
                        InvalidOperation: "`every` duration cannot be zero"
                    );

                    Ok(Some(duration - duration % every_units))
                } else {
                    Ok(None)
                }
            })
        };
        out.map(|s| s.into_duration(self.time_unit()))
    }
}
