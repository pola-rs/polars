use std::str::FromStr;

#[cfg(feature = "timezones")]
use arrow::legacy::kernels::convert_to_naive_local;
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono_tz::Tz;

use super::*;
use crate::datatypes::time_unit::TimeUnit;
use crate::prelude::*;

pub type DatetimeChunked = Logical<DatetimeType, Int64Type>;

impl Int64Chunked {
    pub fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> DatetimeChunked {
        let mut dt = DatetimeChunked::new_logical(self);
        dt.dtype = Some(DataType::Datetime(timeunit, tz));
        dt
    }
}

impl LogicalType for DatetimeChunked {
    fn dtype(&self) -> &DataType {
        self.dtype.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.phys
            .get_any_value(i)
            .map(|av| av.as_datetime(self.time_unit(), self.time_zone().as_ref()))
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.phys
            .get_any_value_unchecked(i)
            .as_datetime(self.time_unit(), self.time_zone().as_ref())
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;

        use crate::datatypes::time_unit::TimeUnit::*;

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
                    _ => return self.phys.cast_with_options(dtype, cast_options),
                };
                match multiplier {
                    // scale to higher precision (eg: ms → us, ms → ns, us → ns)
                    Some(m) => Ok((self.phys.as_ref() * m)
                        .into_datetime(*to_unit, tz.clone())
                        .into_series()),
                    // scale to lower precision (eg: ns → us, ns → ms, us → ms)
                    None => match divisor {
                        Some(d) => Ok(self
                            .phys
                            .apply_values(|v| v.div_euclid(d))
                            .into_datetime(*to_unit, tz.clone())
                            .into_series()),
                        None => unreachable!("must always have a time unit divisor here"),
                    },
                }
            },
            #[cfg(feature = "dtype-date")]
            Date => {
                let timestamp_to_datetime = match self.time_unit() {
                    TimeUnit::Milliseconds => timestamp_ms_to_datetime,
                    TimeUnit::Microseconds => timestamp_us_to_datetime,
                    TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
                };
                let datetime_to_timestamp = match self.time_unit() {
                    TimeUnit::Milliseconds => datetime_to_timestamp_ms,
                    TimeUnit::Microseconds => datetime_to_timestamp_us,
                    TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
                };
                let cast_to_date = |tu_in_day: i64| {
                    let values = match self.dtype() {
                        #[cfg(feature = "timezones")]
                        Datetime(_, Some(tz)) => {
                            let from_tz = tz.to_chrono()?;
                            let ambiguous = StringChunked::from_iter(std::iter::once("raise"));
                            self.phys.apply_values(|timestamp| {
                                let ndt = timestamp_to_datetime(timestamp);
                                let res = convert_to_naive_local(
                                    &from_tz,
                                    &Tz::UTC,
                                    ndt,
                                    Ambiguous::from_str(ambiguous.get(0).unwrap()).unwrap(),
                                    NonExistent::Raise,
                                )
                                .unwrap();
                                res.map(datetime_to_timestamp)
                                    .unwrap()
                                    .div_euclid(tu_in_day)
                            })
                        },
                        _ => self.phys.apply_values(|v| v.div_euclid(tu_in_day)),
                    };

                    let mut dt = values
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
                    .phys
                    .apply_values(|v| {
                        let t = v % scaled_mod * multiplier;
                        t + (NS_IN_DAY * (t < 0) as i64)
                    })
                    .into_time()
                    .into_series());
            },
            dt if dt.is_primitive_numeric() => {
                return self.phys.cast_with_options(dtype, cast_options);
            },
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
