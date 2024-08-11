use std::ops::{Deref, Div};

use arrow::temporal_conversions::{MICROSECONDS_IN_DAY, MILLISECONDS_IN_DAY, NANOSECONDS_IN_DAY};
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::*;

use crate::chunkedarray::*;

pub trait AsSeries {
    fn as_series(&self) -> &Series;
}

impl AsSeries for Series {
    fn as_series(&self) -> &Series {
        self
    }
}

pub trait TemporalMethods: AsSeries {
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.hour()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.hour()),
            dt => polars_bail!(opq = hour, dt),
        }
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.minute()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.minute()),
            dt => polars_bail!(opq = minute, dt),
        }
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.second()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.second()),
            dt => polars_bail!(opq = second, dt),
        }
    }

    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.nanosecond()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.nanosecond()),
            dt => polars_bail!(opq = nanosecond, dt),
        }
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.day()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.day()),
            dt => polars_bail!(opq = day, dt),
        }
    }
    /// Returns the ISO weekday number where monday = 1 and sunday = 7
    fn weekday(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| {
                // Closed formula to find weekday, no need to go via Chrono.
                // The 4 comes from the fact that 1970-01-01 was a Thursday.
                // We do an extra `+ 7` then `% 7` to ensure the result is non-negative.
                unary_elementwise_values(ca, |t| (((t - 4) % 7 + 7) % 7 + 1) as i8)
            }),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(time_unit, time_zone) => s.datetime().map(|ca| {
                match time_zone.as_deref() {
                    Some("UTC") | None => {
                        // fastpath!
                        // Same idea as above, but we need to subtract 1 for dates
                        // before 1970-01-01 with non-zero sub-daily components.
                        let divisor = match time_unit {
                            TimeUnit::Milliseconds => MILLISECONDS_IN_DAY,
                            TimeUnit::Microseconds => MICROSECONDS_IN_DAY,
                            TimeUnit::Nanoseconds => NANOSECONDS_IN_DAY,
                        };
                        unary_elementwise_values(ca, |t| {
                            let t = t / divisor - ((t < 0 && t % divisor != 0) as i64);
                            (((t - 4) % 7 + 7) % 7 + 1) as i8
                        })
                    },
                    _ => ca.weekday(),
                }
            }),
            dt => polars_bail!(opq = weekday, dt),
        }
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.week()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.week()),
            dt => polars_bail!(opq = week, dt),
        }
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal_day(&self) -> PolarsResult<Int16Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.ordinal()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.ordinal()),
            dt => polars_bail!(opq = ordinal_day, dt),
        }
    }

    /// Calculate the millennium from the underlying NaiveDateTime representation.
    fn millennium(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            // note: adjust by one for the years on the <n>000 boundaries.
            // (2000 is the end of the 2nd millennium; 2001 is the beginning of the 3rd).
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| (ca.year() - 1i32).div(1000f64) + 1),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| (ca.year() - 1i32).div(1000f64) + 1),
            dt => polars_bail!(opq = century, dt),
        }
    }

    /// Calculate the millennium from the underlying NaiveDateTime representation.
    fn century(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            // note: adjust by one for years on the <nn>00 boundaries.
            // (1900 is the end of the 19th century; 1901 is the beginning of the 20th).
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| (ca.year() - 1i32).div(100f64) + 1),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| (ca.year() - 1i32).div(100f64) + 1),
            dt => polars_bail!(opq = century, dt),
        }
    }

    /// Extract year from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.year()),
            dt => polars_bail!(opq = year, dt),
        }
    }

    fn iso_year(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.iso_year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.iso_year()),
            dt => polars_bail!(opq = iso_year, dt),
        }
    }

    /// Extract ordinal year from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn ordinal_year(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.year()),
            dt => polars_bail!(opq = ordinal_year, dt),
        }
    }

    /// Extract year from underlying NaiveDateTime representation.
    /// Returns whether the year is a leap year.
    fn is_leap_year(&self) -> PolarsResult<BooleanChunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.is_leap_year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.is_leap_year()),
            dt => polars_bail!(opq = is_leap_year, dt),
        }
    }

    /// Extract quarter from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    fn quarter(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.quarter()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.quarter()),
            dt => polars_bail!(opq = quarter, dt),
        }
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> PolarsResult<Int8Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.month()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.month()),
            dt => polars_bail!(opq = month, dt),
        }
    }

    /// Convert Time into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn to_string(&self, format: &str) -> PolarsResult<Series> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| Ok(ca.to_string(format)?.into_series()))?,
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s
                .datetime()
                .map(|ca| Ok(ca.to_string(format)?.into_series()))?,
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.to_string(format).into_series()),
            dt => polars_bail!(opq = to_string, dt),
        }
    }

    /// Convert from Time into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    ///
    /// Alias for `to_string`.
    fn strftime(&self, format: &str) -> PolarsResult<Series> {
        self.to_string(format)
    }

    #[cfg(feature = "temporal")]
    /// Convert date(time) object to timestamp in [`TimeUnit`].
    fn timestamp(&self, tu: TimeUnit) -> PolarsResult<Int64Chunked> {
        let s = self.as_series();
        if matches!(s.dtype(), DataType::Time | DataType::Duration(_)) {
            polars_bail!(opq = timestamp, s.dtype());
        } else {
            s.cast(&DataType::Datetime(tu, None))
                .map(|s| s.datetime().unwrap().deref().clone())
        }
    }
}

impl<T: ?Sized + AsSeries> TemporalMethods for T {}
