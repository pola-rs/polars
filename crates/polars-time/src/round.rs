use polars_arrow::export::arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_arrow::time_zone::Tz;
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsRound {
    fn round(&self, every: Duration, offset: Duration, tz: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsRound for DatetimeChunked {
    fn round(&self, every: Duration, offset: Duration, tz: Option<&Tz>) -> PolarsResult<Self> {
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::round_ns,
            TimeUnit::Microseconds => Window::round_us,
            TimeUnit::Milliseconds => Window::round_ms,
        };
        Ok(self
            .try_apply(|t| func(&w, t, tz))?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

#[cfg(feature = "dtype-date")]
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
