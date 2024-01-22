use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsRound {
    fn round(&self, every: Duration, offset: Duration, tz: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsRound for DatetimeChunked {
    fn round(&self, every: Duration, offset: Duration, tz: Option<&Tz>) -> PolarsResult<Self> {
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::round_ns,
            TimeUnit::Microseconds => Window::round_us,
            TimeUnit::Milliseconds => Window::round_ms,
        };

        let out = { self.try_apply(|t| func(&w, t, tz)) };
        out.map(|ok| ok.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsRound for DateChunked {
    fn round(&self, every: Duration, offset: Duration, _tz: Option<&Tz>) -> PolarsResult<Self> {
        let w = Window::new(every, every, offset);
        Ok(self
            .try_apply(|t| {
                const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                Ok((w.round_ms(MSECS_IN_DAY * t as i64, None)? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }
}
