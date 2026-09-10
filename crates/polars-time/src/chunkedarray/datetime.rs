use arrow::temporal_conversions::{
    timestamp_ms_to_datetime_opt, timestamp_ns_to_datetime_opt, timestamp_us_to_datetime_opt,
};
#[cfg(feature = "timezones")]
use chrono::TimeZone as _;
use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::*;
#[cfg(feature = "timezones")]
use polars_ops::chunked_array::datetime::replace_time_zone;

use super::*;

/// Extracts one field of the local wall time of every element, reading the instants with
/// `$to_datetime` — the conversion the column's timestamp unit asks for.
///
/// A column that names a time zone has that zone's offset applied as each instant is read, in the
/// same pass the field is taken in — rather than the wall times being written out as a column of
/// their own first and then read back.
macro_rules! extract_with {
    ($ca:expr, $field:expr, $to_datetime:path) => {{
        let ca = $ca;

        #[cfg(feature = "timezones")]
        if let DataType::Datetime(_, Some(time_zone)) = ca.dtype() {
            let tz = time_zone
                .to_chrono()
                .expect("a column's time zone is validated when it is set");

            return unary_elementwise(ca.physical(), move |opt| {
                opt.and_then($to_datetime)
                    .map(|instant| $field(tz.from_utc_datetime(&instant).naive_local()))
            });
        }

        unary_elementwise(ca.physical(), move |opt| {
            opt.and_then($to_datetime).map($field)
        })
    }};
}

/// [`extract_with`], over whichever conversion the column's timestamp unit asks for.
///
/// The unit is dispatched on here, once per column, rather than its conversion being picked as a
/// `fn` pointer that the loop then calls indirectly once per element: the conversion is cheap
/// enough that a call it cannot inline costs about a third as much again as the work it does.
macro_rules! extract {
    ($ca:expr, $field:expr) => {{
        let ca = $ca;
        match ca.time_unit() {
            TimeUnit::Nanoseconds => extract_with!(ca, $field, timestamp_ns_to_datetime_opt),
            TimeUnit::Microseconds => extract_with!(ca, $field, timestamp_us_to_datetime_opt),
            TimeUnit::Milliseconds => extract_with!(ca, $field, timestamp_ms_to_datetime_opt),
        }
    }};
}

pub trait DatetimeMethods: AsDatetime {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked {
        extract!(self.as_datetime(), datetime_year)
    }

    /// Extract year from underlying NaiveDate representation.
    /// Returns whether the year is a leap year.
    fn is_leap_year(&self) -> BooleanChunked {
        extract!(self.as_datetime(), datetime_is_leap_year)
    }

    fn iso_year(&self) -> Int32Chunked {
        extract!(self.as_datetime(), datetime_iso_year)
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
        extract!(self.as_datetime(), datetime_month)
    }

    /// Returns the number of days in the month of the underlying NaiveDateTime
    /// representation.
    fn days_in_month(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_days_in_month)
    }

    /// Extract ISO weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 1 and sunday = 7
    fn weekday(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_iso_week)
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_day)
    }

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> Int8Chunked {
        extract!(self.as_datetime(), datetime_second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> Int32Chunked {
        extract!(self.as_datetime(), datetime_nanosecond)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> Int16Chunked {
        extract!(self.as_datetime(), datetime_ordinal)
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
            dt.physical().to_cont_slice().unwrap().as_slice()
        );
    }
}
