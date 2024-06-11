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
            .map(|av| av.as_datetime(self.time_unit(), self.time_zone()))
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0
            .get_any_value_unchecked(i)
            .as_datetime(self.time_unit(), self.time_zone())
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;
        use TimeUnit::*;
        let out = match dtype {
            Datetime(to_unit, tz) => {
                let from_unit = self.time_unit();
                let (multiplier, divisor) = match (from_unit, to_unit) {
                    // scaling from lower precision to higher precision
                    (Milliseconds, Nanoseconds) => (Some(1_000_000i64), None),
                    (Milliseconds, Microseconds) => (Some(1_000i64), None),
                    (Microseconds, Nanoseconds) => (Some(1_000i64), None),
                    // scaling from higher precision to lower precision
                    (Nanoseconds, Milliseconds) => (None, Some(1_000_000i64)),
                    (Nanoseconds, Microseconds) => (None, Some(1_000i64)),
                    (Microseconds, Milliseconds) => (None, Some(1_000i64)),
                    _ => return self.0.cast_with_options(dtype, cast_options),
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
            Date => {
                let cast_to_date = |tu_in_day: i64| {
                    let mut dt = self
                        .0
                        .apply_values(|v| v.div_euclid(tu_in_day))
                        .cast_with_options(&Int32, cast_options)
                        .unwrap()
                        .into_date()
                        .into_series();
                    dt.set_sorted_flag(self.is_sorted_flag());
                    Ok(dt)
                };
                match self.time_unit() {
                    Nanoseconds => cast_to_date(NS_IN_DAY),
                    Microseconds => cast_to_date(US_IN_DAY),
                    Milliseconds => cast_to_date(MS_IN_DAY),
                }
            },
            #[cfg(feature = "dtype-time")]
            Time => {
                let (scaled_mod, multiplier) = match self.time_unit() {
                    Nanoseconds => (NS_IN_DAY, 1i64),
                    Microseconds => (US_IN_DAY, 1_000i64),
                    Milliseconds => (MS_IN_DAY, 1_000_000i64),
                };
                return Ok(self
                    .0
                    .apply_values(|v| {
                        let t = v % scaled_mod * multiplier;
                        t + (NS_IN_DAY * (t < 0) as i64)
                    })
                    .into_time()
                    .into_series());
            },
            dt if dt.is_numeric() => return self.0.cast_with_options(dtype, cast_options),
            dt => {
                polars_bail!(
                    InvalidOperation:
                    "casting from {:?} to {:?} not supported",
                    self.dtype(), dt
                )
            },
        };
        out.map(|mut s| {
            // TODO!; implement the divisions/multipliers above
            // in a checked manner so that we raise on overflow
            s.set_sorted_flag(self.is_sorted_flag());
            s
        })
    }
}
