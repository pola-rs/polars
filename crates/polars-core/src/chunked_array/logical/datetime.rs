use super::*;
use crate::prelude::*;

pub type DatetimeChunked = Logical<DatetimeType, Int64Type>;

impl Int64Chunked {
    pub fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> DatetimeChunked {
        let mut dt = DatetimeChunked::new_logical(self);
        dt.2 = Some(DataType::Datetime(timeunit, tz));
        dt
    }
}

impl LogicalType for DatetimeChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0
            .get_any_value(i)
            .map(|av| av.into_datetime(self.time_unit(), self.time_zone()))
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0
            .get_any_value_unchecked(i)
            .into_datetime(self.time_unit(), self.time_zone())
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match (self.dtype(), dtype) {
            (Datetime(from_unit, _), Datetime(to_unit, tz)) => {
                let (multiplier, divisor) = match (from_unit, to_unit) {
                    // scaling from lower precision to higher precision
                    (TimeUnit::Milliseconds, TimeUnit::Nanoseconds) => (Some(1_000_000i64), None),
                    (TimeUnit::Milliseconds, TimeUnit::Microseconds) => (Some(1_000i64), None),
                    (TimeUnit::Microseconds, TimeUnit::Nanoseconds) => (Some(1_000i64), None),
                    // scaling from higher precision to lower precision
                    (TimeUnit::Nanoseconds, TimeUnit::Milliseconds) => (None, Some(1_000_000i64)),
                    (TimeUnit::Nanoseconds, TimeUnit::Microseconds) => (None, Some(1_000i64)),
                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => (None, Some(1_000i64)),
                    _ => return self.0.cast(dtype),
                };
                let result = match multiplier {
                    // scale to higher precision (eg: ms → us, ms → ns, us → ns)
                    Some(m) => Ok((self.0.as_ref() * m)
                        .into_datetime(*to_unit, tz.clone())
                        .into_series()),
                    // scale to lower precision (eg: ns → us, ns → ms, us → ms)
                    None => match divisor {
                        Some(d) => Ok(self
                            .0
                            .apply_values(|v| v.div_euclid(d))
                            .into_datetime(*to_unit, tz.clone())
                            .into_series()),
                        None => unreachable!("must always have a time unit divisor here"),
                    },
                };
                result
            },
            #[cfg(feature = "dtype-date")]
            (Datetime(tu, _), Date) => {
                let cast_to_date = |tu_in_day: i64| {
                    Ok(self
                        .0
                        .apply_values(|v| v.div_euclid(tu_in_day))
                        .cast(&Int32)
                        .unwrap()
                        .into_date()
                        .into_series())
                };
                match tu {
                    TimeUnit::Nanoseconds => cast_to_date(NS_IN_DAY),
                    TimeUnit::Microseconds => cast_to_date(US_IN_DAY),
                    TimeUnit::Milliseconds => cast_to_date(MS_IN_DAY),
                }
            },
            #[cfg(feature = "dtype-time")]
            (Datetime(tu, _), Time) => Ok({
                let (scaled_mod, multiplier) = match tu {
                    TimeUnit::Nanoseconds => (NS_IN_DAY, 1i64),
                    TimeUnit::Microseconds => (US_IN_DAY, 1_000i64),
                    TimeUnit::Milliseconds => (MS_IN_DAY, 1_000_000i64),
                };
                self.0
                    .apply_values(|v| {
                        let t = v % scaled_mod * multiplier;
                        t + (NS_IN_DAY * (v < 0 && t != 0) as i64)
                    })
                    .into_time()
                    .into_series()
            }),
            _ => self.0.cast(dtype),
        }
    }
}
