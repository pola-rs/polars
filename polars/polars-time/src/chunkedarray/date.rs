use super::*;
use polars_arrow::export::arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};

pub(crate) fn naive_date_to_date(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms(0, 0, 0);
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date(ndt)
}

pub(crate) fn naive_datetime_to_date(v: NaiveDateTime) -> i32 {
    (datetime_to_timestamp_ms(v) / (MILLISECONDS * SECONDS_IN_DAY)) as i32
}

pub trait DateMethods {
    /// Extract month from underlying NaiveDate representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked;

    /// Extract month from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    fn quarter(&self) -> UInt32Chunked {
        let months = self.month();
        months_to_quarters(months)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> UInt32Chunked;

    /// Extract weekday from underlying NaiveDate representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> UInt32Chunked;

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> UInt32Chunked;

    /// Extract day from underlying NaiveDate representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> UInt32Chunked;

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> UInt32Chunked;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> DateChunked;
}

impl DateMethods for DateChunked {
    /// Extract month from underlying NaiveDate representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked {
        self.apply_kernel_cast::<Int32Type>(&date_to_year)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_month)
    }

    /// Extract weekday from underlying NaiveDate representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_week)
    }

    /// Extract day from underlying NaiveDate representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_day)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&date_to_ordinal)
    }

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> DateChunked {
        Int32Chunked::from_iter_options(
            name,
            v.iter().map(|s| {
                NaiveDate::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(|v| naive_date_to_date(*v))
            }),
        )
        .into()
    }
}
