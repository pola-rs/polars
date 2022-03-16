use crate::chunkedarray::*;
use polars_core::prelude::*;
use std::ops::Deref;

pub trait TemporalMethods: SeriesTrait {
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.hour()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.hour()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.minute()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.minute()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.second()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.second()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.nanosecond()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.nanosecond()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.day()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.day()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.weekday()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.weekday()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.week()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.week()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal_day(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.ordinal()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.ordinal()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> Result<UInt32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.month()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.month()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Result<Int32Chunked> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.year()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    /// Format Date/Datetimewith a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn strftime(&self, fmt: &str) -> Result<Series> {
        match self.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().map(|ca| ca.strftime(fmt).into_series()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().map(|ca| ca.strftime(fmt).into_series()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().map(|ca| ca.strftime(fmt).into_series()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", self.dtype()).into(),
            )),
        }
    }

    #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
    /// Convert date(time) object to timestamp in [`TimeUnit`].
    fn timestamp(&self, tu: TimeUnit) -> Result<Int64Chunked> {
        self.cast(&DataType::Datetime(tu, None))
            .map(|s| s.datetime().unwrap().deref().clone())
    }
}

impl<T: ?Sized + SeriesTrait> TemporalMethods for T {}
