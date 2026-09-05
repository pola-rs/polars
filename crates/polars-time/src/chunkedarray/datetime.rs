use std::borrow::Cow;

use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::*;
#[cfg(feature = "timezones")]
use polars_ops::chunked_array::datetime::replace_time_zone;

use super::*;

/// The column read as its local wall time, so that a value can be asked what calendar date and
/// time it names in its own time zone.
///
/// A column with no time zone already is its wall time and is handed back untouched. Stripping one
/// is always well-defined — it is *adding* a time zone that has to answer for hours that occur
/// twice or not at all — which is why this cannot fail.
fn local(ca: &DatetimeChunked) -> Cow<'_, DatetimeChunked> {
    match ca.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => Cow::Owned(
            polars_ops::chunked_array::replace_time_zone(
                ca,
                None,
                &StringChunked::new("".into(), ["raise"]),
                NonExistent::Raise,
            )
            .expect("Removing time zone is infallible"),
        ),
        _ => Cow::Borrowed(ca),
    }
}

/// Extracts one field of the local wall time of every element, with the timestamp unit of the
/// column picking which of the three extractions is applied.
macro_rules! extract {
    ($ca:expr, $ns:ident, $us:ident, $ms:ident) => {{
        let ca = $ca;
        let f = match ca.time_unit() {
            TimeUnit::Nanoseconds => $ns,
            TimeUnit::Microseconds => $us,
            TimeUnit::Milliseconds => $ms,
        };
        let ca_local = local(ca);
        unary_elementwise(ca_local.physical(), |opt| opt.and_then(f))
    }};
}

pub trait DatetimeMethods: AsDatetime {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_year_ns,
            datetime_to_year_us,
            datetime_to_year_ms
        )
    }

    /// Extract year from underlying NaiveDate representation.
    /// Returns whether the year is a leap year.
    fn is_leap_year(&self) -> BooleanChunked {
        extract!(
            self.as_datetime(),
            datetime_to_is_leap_year_ns,
            datetime_to_is_leap_year_us,
            datetime_to_is_leap_year_ms
        )
    }

    fn iso_year(&self) -> Int32Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_iso_year_ns,
            datetime_to_iso_year_us,
            datetime_to_iso_year_ms
        )
    }

    /// Extract quarter from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    fn quarter(&self) -> Int8Chunked {
        let months = self.month();
        months_to_quarters(months)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_month_ns,
            datetime_to_month_us,
            datetime_to_month_ms
        )
    }

    /// Returns the number of days in the month of the underlying NaiveDateTime
    /// representation.
    fn days_in_month(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_days_in_month_ns,
            datetime_to_days_in_month_us,
            datetime_to_days_in_month_ms
        )
    }

    /// Extract ISO weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 1 and sunday = 7
    fn weekday(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_weekday_ns,
            datetime_to_weekday_us,
            datetime_to_weekday_ms
        )
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_iso_week_ns,
            datetime_to_iso_week_us,
            datetime_to_iso_week_ms
        )
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_day_ns,
            datetime_to_day_us,
            datetime_to_day_ms
        )
    }

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_hour_ns,
            datetime_to_hour_us,
            datetime_to_hour_ms
        )
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_minute_ns,
            datetime_to_minute_us,
            datetime_to_minute_ms
        )
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> Int8Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_second_ns,
            datetime_to_second_us,
            datetime_to_second_ms
        )
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> Int32Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_nanosecond_ns,
            datetime_to_nanosecond_us,
            datetime_to_nanosecond_ms
        )
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> Int16Chunked {
        extract!(
            self.as_datetime(),
            datetime_to_ordinal_ns,
            datetime_to_ordinal_us,
            datetime_to_ordinal_ms
        )
    }

    fn parse_from_str_slice(
        name: PlSmallStr,
        v: &[&str],
        fmt: &str,
        tu: TimeUnit,
    ) -> DatetimeChunked {
        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        Int64Chunked::from_iter_options(
            name,
            v.iter()
                .map(|s| NaiveDateTime::parse_from_str(s, fmt).ok().map(func)),
        )
        .into_datetime(tu, None)
    }

    /// Construct a datetime ChunkedArray from individual time components.
    #[allow(clippy::too_many_arguments)]
    fn new_from_parts(
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        hour: &Int8Chunked,
        minute: &Int8Chunked,
        second: &Int8Chunked,
        nanosecond: &Int32Chunked,
        ambiguous: &StringChunked,
        time_unit: &TimeUnit,
        time_zone: Option<TimeZone>,
        name: PlSmallStr,
    ) -> PolarsResult<DatetimeChunked> {
        let ca: Int64Chunked = year
            .iter()
            .zip(month.iter())
            .zip(day.iter())
            .zip(hour.iter())
            .zip(minute.iter())
            .zip(second.iter())
            .zip(nanosecond.iter())
            .map(|((((((y, m), d), h), mnt), s), ns)| {
                if let (Some(y), Some(m), Some(d), Some(h), Some(mnt), Some(s), Some(ns)) =
                    (y, m, d, h, mnt, s, ns)
                {
                    NaiveDate::from_ymd_opt(y, m as u32, d as u32).map_or_else(
                        // We have an invalid date.
                        || polars_bail!(ComputeError: "Invalid date components ({y}, {m}, {d}) supplied"),
                        // We have a valid date.
                        |date| {
                            date.and_hms_nano_opt(h as u32, mnt as u32, s as u32, ns as u32)
                                .map_or_else(
                                    // We have invalid time components for the specified date.
                                    || polars_bail!(ComputeError: "Invalid time components ({h}, {mnt}, {s}, {ns}) supplied"),
                                    // We have a valid time.
                                    |ndt| {
                                        let t = ndt.and_utc();
                                        Ok(Some(match time_unit {
                                            TimeUnit::Milliseconds => t.timestamp_millis(),
                                            TimeUnit::Microseconds => t.timestamp_micros(),
                                            TimeUnit::Nanoseconds => {
                                                t.timestamp_nanos_opt().unwrap()
                                            },
                                        }))
                                    },
                                )
                        },
                    )
                } else {
                    Ok(None)
                }
            })
            .try_collect_ca_with_dtype(name, DataType::Int64)?;

        let ca = match time_zone {
            #[cfg(feature = "timezones")]
            Some(_) => {
                let mut ca = ca.into_datetime(*time_unit, None);
                ca = replace_time_zone(&ca, time_zone.as_ref(), ambiguous, NonExistent::Raise)?;
                ca
            },
            _ => {
                polars_ensure!(
                    time_zone.is_none(),
                    ComputeError: "cannot make use of the `time_zone` argument without the 'timezones' feature enabled."
                );
                ca.into_datetime(*time_unit, None)
            },
        };
        Ok(ca)
    }
}

pub trait AsDatetime {
    fn as_datetime(&self) -> &DatetimeChunked;
}

impl AsDatetime for DatetimeChunked {
    fn as_datetime(&self) -> &DatetimeChunked {
        self
    }
}

impl DatetimeMethods for DatetimeChunked {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_datetime() {
        let datetimes: Vec<_> = [
            "1988-08-25 00:00:16",
            "2015-09-05 23:56:04",
            "2012-12-21 00:00:00",
        ]
        .iter()
        .map(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap())
        .collect();

        // NOTE: the values are checked and correct.
        let dt = DatetimeChunked::from_naive_datetime(
            "name".into(),
            datetimes.iter().copied(),
            TimeUnit::Nanoseconds,
        );
        assert_eq!(
            [
                588_470_416_000_000_000,
                1_441_497_364_000_000_000,
                1_356_048_000_000_000_000
            ],
            dt.physical().to_flat().cont_slice().unwrap()
        );
    }
}
