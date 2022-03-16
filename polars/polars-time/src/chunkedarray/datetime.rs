use super::*;
use polars_arrow::export::arrow::array::{MutableArray, MutableUtf8Array, Utf8Array};
use polars_arrow::export::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
use std::fmt::Write;

pub trait DatetimeMethods {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked;

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> UInt32Chunked;

    /// Extract weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> UInt32Chunked;

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> UInt32Chunked;

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> UInt32Chunked;

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> UInt32Chunked;

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> UInt32Chunked;

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> UInt32Chunked;

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> UInt32Chunked;

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> UInt32Chunked;

    /// Format Datetime with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn strftime(&self, fmt: &str) -> Utf8Chunked;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str, tu: TimeUnit) -> DatetimeChunked;
}

impl DatetimeMethods for DatetimeChunked {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_year_ns,
            TimeUnit::Microseconds => datetime_to_year_us,
            TimeUnit::Milliseconds => datetime_to_year_ms,
        };
        self.apply_kernel_cast::<Int32Type>(&f)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_month_ns,
            TimeUnit::Microseconds => datetime_to_month_us,
            TimeUnit::Milliseconds => datetime_to_month_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Extract weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_weekday_ns,
            TimeUnit::Microseconds => datetime_to_weekday_us,
            TimeUnit::Milliseconds => datetime_to_weekday_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_week_ns,
            TimeUnit::Microseconds => datetime_to_week_us,
            TimeUnit::Milliseconds => datetime_to_week_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_day_ns,
            TimeUnit::Microseconds => datetime_to_day_us,
            TimeUnit::Milliseconds => datetime_to_day_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_hour_ns,
            TimeUnit::Microseconds => datetime_to_hour_us,
            TimeUnit::Milliseconds => datetime_to_hour_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_minute_ns,
            TimeUnit::Microseconds => datetime_to_minute_us,
            TimeUnit::Milliseconds => datetime_to_minute_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_second_ns,
            TimeUnit::Microseconds => datetime_to_second_us,
            TimeUnit::Milliseconds => datetime_to_second_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_nanosecond_ns,
            TimeUnit::Microseconds => datetime_to_nanosecond_us,
            TimeUnit::Milliseconds => datetime_to_nanosecond_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> UInt32Chunked {
        let f = match self.time_unit() {
            TimeUnit::Nanoseconds => datetime_to_ordinal_ns,
            TimeUnit::Microseconds => datetime_to_ordinal_us,
            TimeUnit::Milliseconds => datetime_to_ordinal_ms,
        };
        self.apply_kernel_cast::<UInt32Type>(&f)
    }

    /// Format Datetime with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let conversion_f = match self.time_unit() {
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
            TimeUnit::Microseconds => timestamp_us_to_datetime,
            TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        };

        let dt = NaiveDate::from_ymd(2001, 1, 1).and_hms(0, 0, 0);
        let fmted = format!("{}", dt.format(fmt));

        let mut ca: Utf8Chunked = self.apply_kernel_cast(&|arr| {
            let mut buf = String::new();
            let mut mutarr =
                MutableUtf8Array::with_capacities(arr.len(), arr.len() * fmted.len() + 1);

            for opt in arr.into_iter() {
                match opt {
                    None => mutarr.push_null(),
                    Some(v) => {
                        buf.clear();
                        let datefmt = conversion_f(*v).format(fmt);
                        write!(buf, "{}", datefmt).unwrap();
                        mutarr.push(Some(&buf))
                    }
                }
            }

            let arr: Utf8Array<i64> = mutarr.into();
            Arc::new(arr)
        });
        ca.rename(self.name());
        ca
    }

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str, tu: TimeUnit) -> DatetimeChunked {
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
}

#[cfg(test)]
mod test {
    use super::*;
    use chrono::NaiveDateTime;

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
            "name",
            datetimes.iter().copied(),
            TimeUnit::Nanoseconds,
        );
        assert_eq!(
            [
                588470416000_000_000,
                1441497364000_000_000,
                1356048000000_000_000
            ],
            dt.cont_slice().unwrap()
        );
    }
}
