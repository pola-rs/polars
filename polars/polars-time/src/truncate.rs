#[cfg(feature = "dtype-date")]
use polars_arrow::export::arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_arrow::time_zone::Tz;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TruncateOptions {
    /// Period length
    pub every: String,
    /// Offset of the window
    pub offset: String,
    /// How to deal with ambiguous datetimes
    pub use_earliest: Option<bool>,
}

pub trait PolarsTruncate {
    fn truncate(&self, options: &TruncateOptions, tz: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsTruncate for DatetimeChunked {
    fn truncate(&self, options: &TruncateOptions, tz: Option<&Tz>) -> PolarsResult<Self> {
        let every = Duration::parse(&options.every);
        let offset = Duration::parse(&options.offset);
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Microseconds => Window::truncate_us,
            TimeUnit::Milliseconds => Window::truncate_ms,
        };

        Ok(self
            .try_apply(|t| func(&w, t, tz, options.use_earliest))?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

#[cfg(feature = "dtype-date")]
impl PolarsTruncate for DateChunked {
    fn truncate(&self, options: &TruncateOptions, _tz: Option<&Tz>) -> PolarsResult<Self> {
        let every = Duration::parse(&options.every);
        let offset = Duration::parse(&options.offset);
        let w = Window::new(every, every, offset);
        Ok(self
            .try_apply(|t| {
                const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                Ok((w.truncate_ms(MSECS_IN_DAY * t as i64, None, None)? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }
}
