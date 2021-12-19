use crate::prelude::*;
use arrow::temporal_conversions::NANOSECONDS;
use polars_time::{Duration, Window};

#[cfg(feature = "dtype-datetime")]
impl DatetimeChunked {
    pub fn truncate(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);
        self.apply(|t| w.truncate(t)).into_date()
    }
}

#[cfg(feature = "dtype-date")]
impl DateChunked {
    pub fn truncate(&self, every: Duration, offset: Duration) -> Self {
        let w = Window::new(every, every, offset);
        self.apply(|t| {
            const NSECS_IN_DAY: i64 = NANOSECONDS * SECONDS_IN_DAY;
            (w.truncate(NSECS_IN_DAY * t as i64) / NSECS_IN_DAY) as i32
        })
        .into_date()
    }
}
