use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_utils::cache::FastFixedCache;

use crate::prelude::*;

pub trait PolarsRound {
    fn round(&self, every: &StringChunked, tz: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsRound for DatetimeChunked {
    fn round(&self, every: &StringChunked, tz: Option<&Tz>) -> PolarsResult<Self> {
        let time_zone = self.time_zone();
        let offset = Duration::new(0);

        // Let's check if we can use a fastpath...
        if every.len() == 1 {
            if let Some(every) = every.get(0) {
                let every_parsed = Duration::parse(every);
                polars_ensure!(!every_parsed.negative & !every_parsed.is_zero(), InvalidOperation: "cannot round a Datetime to a non-positive duration");
                if (time_zone.is_none() || time_zone.as_deref() == Some("UTC"))
                    && (every_parsed.months() == 0 && every_parsed.weeks() == 0)
                {
                    // ... yes we can! Weeks, months, and time zones require extra logic.
                    // But in this simple case, it's just simple integer arithmetic.
                    let every = match self.time_unit() {
                        TimeUnit::Milliseconds => every_parsed.duration_ms(),
                        TimeUnit::Microseconds => every_parsed.duration_us(),
                        TimeUnit::Nanoseconds => every_parsed.duration_ns(),
                    };
                    return Ok(self
                        .apply_values(|t| {
                            // Round half-way values away from zero
                            let half_away = t.signum() * every / 2;
                            t + half_away - (t + half_away) % every
                        })
                        .into_datetime(self.time_unit(), time_zone.clone()));
                } else {
                    let w = Window::new(every_parsed, every_parsed, offset);
                    let out = match self.time_unit() {
                        TimeUnit::Milliseconds => {
                            self.try_apply_nonnull_values_generic(|t| w.round_ms(t, tz))
                        },
                        TimeUnit::Microseconds => {
                            self.try_apply_nonnull_values_generic(|t| w.round_us(t, tz))
                        },
                        TimeUnit::Nanoseconds => {
                            self.try_apply_nonnull_values_generic(|t| w.round_ns(t, tz))
                        },
                    };
                    return Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()));
                }
            } else {
                return Ok(Int64Chunked::full_null(self.name(), self.len())
                    .into_datetime(self.time_unit(), self.time_zone().clone()));
            }
        }

        // A sqrt(n) cache is not too small, not too large.
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);

        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => Window::round_ns,
            TimeUnit::Microseconds => Window::round_us,
            TimeUnit::Milliseconds => Window::round_ms,
        };

        let out = broadcast_try_binary_elementwise(self, every, |opt_timestamp, opt_every| match (
            opt_timestamp,
            opt_every,
        ) {
            (Some(timestamp), Some(every)) => {
                let every_parsed =
                    *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));
                polars_ensure!(!every_parsed.negative & !every_parsed.is_zero(), InvalidOperation: "cannot round a Date to a non-positive duration");

                let w = Window::new(every_parsed, every_parsed, offset);
                func(&w, timestamp, tz).map(Some)
            },
            _ => Ok(None),
        });
        Ok(out?.into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsRound for DateChunked {
    fn round(&self, every: &StringChunked, _tz: Option<&Tz>) -> PolarsResult<Self> {
        let offset = Duration::new(0);
        let out = match every.len() {
            1 => {
                if let Some(every) = every.get(0) {
                    let every_parsed = Duration::parse(every);
                    polars_ensure!(!every_parsed.negative & !every_parsed.is_zero(), InvalidOperation: "cannot round a Date to a non-positive duration");
                    let w = Window::new(every_parsed, every_parsed, offset);
                    self.try_apply_nonnull_values_generic(|t| {
                        Ok(
                            (w.round_ms(MILLISECONDS_IN_DAY * t as i64, None)?
                                / MILLISECONDS_IN_DAY) as i32,
                        )
                    })
                } else {
                    Ok(Int32Chunked::full_null(self.name(), self.len()))
                }
            },
            _ => broadcast_try_binary_elementwise(self, every, |opt_t, opt_every| {
                // A sqrt(n) cache is not too small, not too large.
                let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);
                match (opt_t, opt_every) {
                    (Some(t), Some(every)) => {
                        let every_parsed = *duration_cache
                            .get_or_insert_with(every, |every| Duration::parse(every));
                        polars_ensure!(!every_parsed.negative & !every_parsed.is_zero(), InvalidOperation: "cannot round a Date to a non-positive duration");

                        let w = Window::new(every_parsed, every_parsed, offset);
                        Ok(Some(
                            (w.round_ms(MILLISECONDS_IN_DAY * t as i64, None)?
                                / MILLISECONDS_IN_DAY) as i32,
                        ))
                    },
                    _ => Ok(None),
                }
            }),
        };
        Ok(out?.into_date())
    }
}

#[cfg(feature = "dtype-duration")]
impl PolarsRound for DurationChunked {
    fn round(&self, every: &StringChunked, _tz: Option<&Tz>) -> PolarsResult<Self> {
        if every.len() == 1 {
            if let Some(every) = every.get(0) {
                let every_parsed = Duration::parse(every);
                polars_ensure!(!every_parsed.negative & !every_parsed.is_zero(), InvalidOperation: "cannot round a Duration to a non-positive duration");
                polars_ensure!(every_parsed.is_constant_duration(None), InvalidOperation:"cannot round a Duration to a non-constant duration (i.e. one that involves weeks / months)");
                let every = match self.time_unit() {
                    TimeUnit::Milliseconds => every_parsed.duration_ms(),
                    TimeUnit::Microseconds => every_parsed.duration_us(),
                    TimeUnit::Nanoseconds => every_parsed.duration_ns(),
                };
                return Ok(self
                    .apply_values(|t| {
                        // Round half-way values away from zero
                        let half_away = t.signum() * every / 2;
                        t + half_away - (t + half_away) % every
                    })
                    .into_duration(self.time_unit()));
            } else {
                return Ok(Int64Chunked::full_null(self.name(), self.len())
                    .into_duration(self.time_unit()));
            }
        }

        // A sqrt(n) cache is not too small, not too large.
        let mut duration_cache = FastFixedCache::new((every.len() as f64).sqrt() as usize);

        let out = broadcast_try_binary_elementwise(self, every, |opt_timestamp, opt_every| match (
            opt_timestamp,
            opt_every,
        ) {
            (Some(t), Some(every)) => {
                let every_parsed =
                    *duration_cache.get_or_insert_with(every, |every| Duration::parse(every));
                polars_ensure!(!every_parsed.negative, InvalidOperation: "cannot round a Duration to a negative duration");
                polars_ensure!(every_parsed.is_constant_duration(None), InvalidOperation:"cannot round a Duration to a non-constant duration (i.e. one that involves weeks / months)");
                let every = match self.time_unit() {
                    TimeUnit::Milliseconds => every_parsed.duration_ms(),
                    TimeUnit::Microseconds => every_parsed.duration_us(),
                    TimeUnit::Nanoseconds => every_parsed.duration_ns(),
                };
                // Round half-way values away from zero
                let half_away = t.signum() * every / 2;
                Ok(Some(t + half_away - (t + half_away) % every))
            },
            _ => Ok(None),
        });
        Ok(out?.into_duration(self.time_unit()))
    }
}
