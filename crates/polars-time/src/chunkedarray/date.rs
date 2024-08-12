use arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};

use super::*;

pub(crate) fn naive_date_to_date(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms_opt(0, 0, 0).unwrap();
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date(ndt)
}

pub(crate) fn naive_datetime_to_date(v: NaiveDateTime) -> i32 {
    (datetime_to_timestamp_ms(v) / (MILLISECONDS * SECONDS_IN_DAY)) as i32
}

pub trait DateMethods: AsDate {
    /// Extract month from underlying NaiveDate representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> Int32Chunked {
        let ca = self.as_date();
        ca.apply_kernel_cast::<Int32Type>(&date_to_year)
    }

    /// Extract year from underlying NaiveDate representation.
    /// Returns whether the year is a leap year.
    fn is_leap_year(&self) -> BooleanChunked {
        let ca = self.as_date();
        ca.apply_kernel_cast::<BooleanType>(&date_to_is_leap_year)
    }

    /// This year number might not match the calendar year number.
    fn iso_year(&self) -> Int32Chunked {
        let ca = self.as_date();
        ca.apply_kernel_cast::<Int32Type>(&date_to_iso_year)
    }

    /// Extract month from underlying NaiveDateTime representation.
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
        let ca = self.as_date();
        ca.apply_kernel_cast::<Int8Type>(&date_to_month)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> Int8Chunked {
        let ca = self.as_date();
        ca.apply_kernel_cast::<Int8Type>(&date_to_iso_week)
    }

    /// Extract day from underlying NaiveDate representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> Int8Chunked {
        let ca = self.as_date();
        ca.apply_kernel_cast::<Int8Type>(&date_to_day)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal(&self) -> Int16Chunked {
        let ca = self.as_date();
        ca.apply_kernel_cast::<Int16Type>(&date_to_ordinal)
    }

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> DateChunked;
}

impl DateMethods for DateChunked {
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

pub trait AsDate {
    fn as_date(&self) -> &DateChunked;
}

impl AsDate for DateChunked {
    fn as_date(&self) -> &DateChunked {
        self
    }
}
