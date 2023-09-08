#[cfg(feature = "dtype-date")]
use polars_arrow::export::arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_arrow::time_zone::Tz;
use polars_core::chunked_array::ops::arity::try_binary_elementwise_values;
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
}

pub trait PolarsTruncate {
    fn truncate(
        &self,
        options: &TruncateOptions,
        tz: Option<&Tz>,
        ambiguous: &Utf8Chunked,
    ) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsTruncate for DatetimeChunked {
    fn truncate(
        &self,
        options: &TruncateOptions,
        tz: Option<&Tz>,
        ambiguous: &Utf8Chunked,
    ) -> PolarsResult<Self> {
        let every = Duration::parse(&options.every);
        let offset = Duration::parse(&options.offset);
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::truncate_ns,
            TimeUnit::Microseconds => Window::truncate_us,
            TimeUnit::Milliseconds => Window::truncate_ms,
        };

        let out = match ambiguous.len() {
            1 => match ambiguous.get(0) {
                Some(ambiguous) => self
                    .0
                    .try_apply(|timestamp| func(&w, timestamp, tz, ambiguous)),
                _ => Ok(self.0.apply(|_| None)),
            },
            _ => {
                try_binary_elementwise_values(self, ambiguous, |timestamp: i64, ambiguous: &str| {
                    func(&w, timestamp, tz, ambiguous)
                })
            },
        };
        Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

#[cfg(feature = "dtype-date")]
impl PolarsTruncate for DateChunked {
    fn truncate(
        &self,
        options: &TruncateOptions,
        _tz: Option<&Tz>,
        _ambiguous: &Utf8Chunked,
    ) -> PolarsResult<Self> {
        let every = Duration::parse(&options.every);
        let offset = Duration::parse(&options.offset);
        let w = Window::new(every, every, offset);
        Ok(self
            .try_apply(|t| {
                const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                Ok((w.truncate_ms(MSECS_IN_DAY * t as i64, None, "raise")? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }
}
