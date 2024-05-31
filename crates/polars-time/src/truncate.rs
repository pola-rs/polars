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
        let offset: Duration = Duration::parse(offset);
        let time_zone = self.time_zone();
        let mut duration_cache_opt: Option<FastFixedCache<String, Duration>> = None;

        // Let's check if we can use a fastpath...
        if every.len() == 1 {
            if let Some(every) = every.get(0) {
                let every_parsed = Duration::parse(every);
                if every_parsed.negative {
                    polars_bail!(ComputeError: "cannot truncate a Datetime to a negative duration")
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
                        .apply_values(|t| {
                            let remainder = t % every;
                            t - remainder + every * (remainder < 0) as i64
                        })
                        .into_datetime(self.time_unit(), time_zone.clone()));
                } else {
                    // A sqrt(n) cache is not too small, not too large.
                    duration_cache_opt =
                        Some(FastFixedCache::new((every.len() as f64).sqrt() as usize));
                    duration_cache_opt
                        .as_mut()
                        .map(|cache| *cache.insert(every.to_string(), every_parsed));
                }
            }
        }
        let mut duration_cache = match duration_cache_opt {
            Some(cache) => cache,
            None => FastFixedCache::new((every.len() as f64).sqrt() as usize),
        };

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Microseconds => Window::truncate_us,
            TimeUnit::Milliseconds => Window::truncate_ms,
        };

        // TODO: optimize the code below, so it does the following:
        //       - convert to naive
        //       - truncate all naively
        //       - localize, preserving the fold of the original datetime.
        //       The last step is the non-trivial one. But it should be worth it,
        //       and faster than the current approach of truncating everything
        //       as tz-aware.

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
