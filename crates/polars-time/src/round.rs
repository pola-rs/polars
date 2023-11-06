use std::str::FromStr;

use arrow::legacy::kernels::Ambiguous;
use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_core::prelude::arity::try_binary_elementwise;
use polars_core::prelude::*;

use crate::prelude::*;

pub trait PolarsRound {
    fn round(
        &self,
        every: Duration,
        offset: Duration,
        tz: Option<&Tz>,
        ambiguous: &Utf8Chunked,
    ) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsRound for DatetimeChunked {
    fn round(
        &self,
        every: Duration,
        offset: Duration,
        tz: Option<&Tz>,
        ambiguous: &Utf8Chunked,
    ) -> PolarsResult<Self> {
        let w = Window::new(every, every, offset);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::round_ns,
            TimeUnit::Microseconds => Window::round_us,
            TimeUnit::Milliseconds => Window::round_ms,
        };

        let out = match ambiguous.len() {
            1 => match ambiguous.get(0) {
                Some(ambiguous) => {
                    self.try_apply(|t| func(&w, t, tz, Ambiguous::from_str(ambiguous)?))
                },
                None => Ok(Int64Chunked::full_null(self.name(), self.len())),
            },
            _ => try_binary_elementwise(self, ambiguous, |opt_t, opt_aambiguous| {
                match (opt_t, opt_aambiguous) {
                    (Some(t), Some(ambiguous)) => {
                        func(&w, t, tz, Ambiguous::from_str(ambiguous)?).map(Some)
                    },
                    _ => Ok(None),
                }
            }),
        };
        out.map(|ok| ok.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

#[cfg(feature = "dtype-date")]
impl PolarsRound for DateChunked {
    fn round(
        &self,
        every: Duration,
        offset: Duration,
        _tz: Option<&Tz>,
        _ambiguous: &Utf8Chunked,
    ) -> PolarsResult<Self> {
        let w = Window::new(every, every, offset);
        Ok(self
            .try_apply(|t| {
                const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
                Ok(
                    (w.round_ms(MSECS_IN_DAY * t as i64, None, Ambiguous::Raise)? / MSECS_IN_DAY)
                        as i32,
                )
            })?
            .into_date())
    }
}
