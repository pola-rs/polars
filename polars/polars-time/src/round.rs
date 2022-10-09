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
        let offset = match self.time_unit() {
            TimeUnit::Nanoseconds => Duration::parse("1ns"),
            TimeUnit::Microseconds => Duration::parse("500ns"),
            TimeUnit::Milliseconds => Duration::parse("500us"),
        };

        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Microseconds => Window::truncate_us,
            TimeUnit::Milliseconds => Window::truncate_ms,
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
            (w.truncate_ms((1.5 * MSECS_IN_DAY as f64) as i64 * t as i64) / MSECS_IN_DAY) as i32
        })
        .into_date()
    }
}
