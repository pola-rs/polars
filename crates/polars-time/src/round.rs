use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_utils::cache::FastFixedCache;

use crate::prelude::*;

pub trait PolarsRound {
    fn round(&self, every: &StringChunked, tz: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsRound for DatetimeChunked {
    fn round(&self, every: &StringChunked, tz: Option<&Tz>) -> PolarsResult<Self> {
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);
        let offset = Duration::new(0);
        let out = broadcast_try_binary_elementwise(self, every, |opt_t, opt_every| {
            match (opt_t, opt_every) {
                (Some(timestamp), Some(every)) => {
                    let every =
                        *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));

                    if every.negative {
                        polars_bail!(ComputeError: "Cannot round a Datetime to a negative duration")
                    }

                    let w = Window::new(every, every, offset);

                    let func = match self.time_unit() {
                        TimeUnit::Nanoseconds => Window::round_ns,
                        TimeUnit::Microseconds => Window::round_us,
                        TimeUnit::Milliseconds => Window::round_ms,
                    };
                    func(&w, timestamp, tz).map(Some)
                },
                _ => Ok(None),
            }
        });
        Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsRound for DateChunked {
    fn round(&self, every: &StringChunked, _tz: Option<&Tz>) -> PolarsResult<Self> {
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);
        let offset = Duration::new(0);
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let out = broadcast_try_binary_elementwise(&self.0, every, |opt_t, opt_every| {
            match (opt_t, opt_every) {
                (Some(t), Some(every)) => {
                    let every =
                        *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));
                    if every.negative {
                        polars_bail!(ComputeError: "Cannot round a Date to a negative duration")
                    }

                    let w = Window::new(every, every, offset);
                    Ok(Some(
                        (w.round_ms(MSECS_IN_DAY * t as i64, None)? / MSECS_IN_DAY) as i32,
                    ))
                },
                _ => Ok(None),
            }
        });
        Ok(out?.into_date())
    }
}
