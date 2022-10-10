use polars_arrow::export::arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsRound {
    #[must_use]
    fn round(&self, every: Duration, offset: Duration) -> Self;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsRound for DatetimeChunked {
    #[must_use]
    fn round(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::round_ns,
            TimeUnit::Microseconds => Window::round_us,
            TimeUnit::Milliseconds => Window::round_ms,
        };

        self.apply(|t| func(&w, t))
            .into_datetime(self.time_unit(), self.time_zone().clone())
    }
}

#[cfg(feature = "dtype-date")]
impl PolarsRound for DateChunked {
    #[must_use]
    fn round(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);
        self.apply(|t| {
            const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
            (w.round_ms(MSECS_IN_DAY * t as i64) / MSECS_IN_DAY) as i32
        })
        .into_date()
    }
}
