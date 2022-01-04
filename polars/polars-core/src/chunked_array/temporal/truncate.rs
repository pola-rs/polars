use crate::prelude::*;
use arrow::temporal_conversions::MILLISECONDS;
use polars_time::{Duration, Window};

#[cfg(feature = "dtype-datetime")]
impl DatetimeChunked {
    #[must_use]
    pub fn truncate(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Milliseconds => Window::truncate_ms,
        };

        self.apply(|t| func(&w, t))
            .into_datetime(self.time_unit(), self.time_zone().clone())
    }
}

#[cfg(feature = "dtype-date")]
impl DateChunked {
    #[must_use]
    pub fn truncate(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);
        self.apply(|t| {
            const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
            (w.truncate_ms(MSECS_IN_DAY * t as i64) / MSECS_IN_DAY) as i32
        })
        .into_date()
    }
}
