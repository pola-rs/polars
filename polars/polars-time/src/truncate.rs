use polars_arrow::export::arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsTruncate {
    #[must_use]
    fn truncate(&self, every: Duration, offset: Duration) -> Self;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsTruncate for DatetimeChunked {
    #[must_use]
    fn truncate(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Microseconds => Window::truncate_us,
            TimeUnit::Milliseconds => Window::truncate_ms,
        };

        self.apply_on_tz_corrected(|ca| {
            Ok(ca
                .apply(|t| func(&w, t))
                .into_datetime(self.time_unit(), self.time_zone().clone()))
        })
        .unwrap()
    }
}

#[cfg(feature = "dtype-date")]
impl PolarsTruncate for DateChunked {
    #[must_use]
    fn truncate(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);
        self.apply(|t| {
            const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
            (w.truncate_ms(MSECS_IN_DAY * t as i64) / MSECS_IN_DAY) as i32
        })
        .into_date()
    }
}
