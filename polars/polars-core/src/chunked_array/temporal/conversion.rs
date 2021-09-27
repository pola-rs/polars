use super::*;
use crate::chunked_array::kernels::temporal::{
    date32_to_day, date32_to_month, date32_to_ordinal, date32_to_week, date32_to_weekday,
    date32_to_year, date64_to_day, date64_to_hour, date64_to_minute, date64_to_month,
    date64_to_nanosecond, date64_to_ordinal, date64_to_second, date64_to_week, date64_to_weekday,
    date64_to_year,
};
use crate::prelude::*;
use crate::utils::CustomIterTools;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime};
pub use conversions_utils::*;

pub fn naive_date_to_date32(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms(0, 0, 0);
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date32(&ndt)
}

pub trait AsNaiveDateTime {
    fn as_naive_datetime_iter<'a>(&'a self)
        -> Box<dyn Iterator<Item = Option<NaiveDateTime>> + 'a>;
}

macro_rules! impl_as_naive_datetime {
    ($ca:ty, $fun:ident) => {
        impl AsNaiveDateTime for $ca {
            fn as_naive_datetime_iter<'a>(
                &'a self,
            ) -> Box<dyn Iterator<Item = Option<NaiveDateTime>> + 'a> {
                Box::new(self.into_iter().map(|opt_t| opt_t.map($fun)))
            }
        }
    };
}

impl_as_naive_datetime!(Date32Chunked, date32_as_datetime);
impl_as_naive_datetime!(Date64Chunked, date64_as_datetime);

pub trait AsNaiveDate {
    fn as_naive_date_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<NaiveDate>> + 'a>;
}

impl AsNaiveDate for Date32Chunked {
    fn as_naive_date_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<NaiveDate>> + 'a> {
        Box::new(self.into_iter().map(|opt_t| {
            opt_t.map(|v| {
                let dt = date32_as_datetime(v);
                NaiveDate::from_ymd(dt.year(), dt.month(), dt.day())
            })
        }))
    }
}

fn date_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> result
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    for fmt in [
        // 2021-12-31
        "%Y-%m-%d",
        // 31-12-2021
        "%d-%m-%Y",
        // 2021/12/31 12:54:98
        "%Y/%m/%d %H:%M:%S",
        // 2021-12-31 24:58:01
        "%Y-%m-%d %H:%M:%S",
        // 2021/12/31 24:58:01
        "%Y/%m/%d %H:%M:%S",
        // 20210319 23:58:50
        "%Y%m%d %H:%M:%S",
        // 2021319 (2021-03-19)
        "%Y%m%d",
        // 2019-04-18T02:45:55
        "%FT%H:%M:%S",
        // 2019-04-18T02:45:55.555000000
        "%FT%H:%M:%S.%6f",
    ] {
        if convert(val, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

impl Utf8Chunked {
    fn get_first_val(&self) -> Result<&str> {
        let idx = match self.first_non_null() {
            Some(idx) => idx,
            None => {
                return Err(PolarsError::HasNullValues(
                    "Cannot determine date parsing format, all values are null".into(),
                ))
            }
        };
        let val = self.get(idx).expect("should not be null");
        Ok(val)
    }

    fn sniff_fmt_date64(&self) -> Result<&'static str> {
        let val = self.get_first_val()?;
        if let Some(pattern) = date_pattern(val, NaiveDateTime::parse_from_str) {
            return Ok(pattern);
        }
        Err(PolarsError::ComputeError(
            "Could not find an appropriate format to parse dates, please define a fmt".into(),
        ))
    }

    fn sniff_fmt_date32(&self) -> Result<&'static str> {
        let val = self.get_first_val()?;
        if let Some(pattern) = date_pattern(val, NaiveDate::parse_from_str) {
            return Ok(pattern);
        }
        Err(PolarsError::ComputeError(
            "Could not find an appropriate format to parse dates, please define a fmt".into(),
        ))
    }

    pub fn as_date32(&self, fmt: Option<&str>) -> Result<Date32Chunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => self.sniff_fmt_date32()?,
        };

        let mut ca: Date32Chunked = match self.null_count() {
            0 => self
                .into_no_null_iter()
                .map(|s| {
                    NaiveDate::parse_from_str(s, fmt)
                        .ok()
                        .map(naive_date_to_date32)
                })
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        NaiveDate::parse_from_str(s, fmt)
                            .ok()
                            .map(naive_date_to_date32)
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect_trusted(),
        };
        ca.rename(self.name());
        Ok(ca)
    }

    pub fn as_date64(&self, fmt: Option<&str>) -> Result<Date64Chunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => self.sniff_fmt_date64()?,
        };

        let mut ca: Date64Chunked = match self.null_count() {
            0 => self
                .into_no_null_iter()
                .map(|s| {
                    NaiveDateTime::parse_from_str(s, fmt)
                        .ok()
                        .map(|dt| naive_datetime_to_date64(&dt))
                })
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        NaiveDateTime::parse_from_str(s, fmt)
                            .ok()
                            .map(|dt| naive_datetime_to_date64(&dt))
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect_trusted(),
        };
        ca.rename(self.name());
        Ok(ca)
    }
}

impl Date64Chunked {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    pub fn year(&self) -> Int32Chunked {
        self.apply_kernel_cast::<_, Int32Type>(date64_to_year)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    pub fn month(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_month)
    }

    /// Extract weekday from underlying NaiveDateTime representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    pub fn weekday(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    pub fn week(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_week)
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    pub fn day(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_day)
    }
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    pub fn hour(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    pub fn minute(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    pub fn second(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    pub fn nanosecond(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_nanosecond)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    pub fn ordinal(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_ordinal)
    }

    /// Format Date64 with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let mut ca: Utf8Chunked = self
            .as_naive_datetime_iter()
            .map(|opt_dt| opt_dt.map(|dt| format!("{}", dt.format(fmt))))
            .collect();
        ca.rename(self.name());
        ca
    }

    pub fn new_from_naive_datetime(name: &str, v: &[NaiveDateTime]) -> Self {
        let vals = v
            .iter()
            .map(naive_datetime_to_date64)
            .collect_trusted::<AlignedVec<_>>();
        ChunkedArray::new_from_aligned_vec(name, vals)
    }

    pub fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
        ChunkedArray::new_from_opt_iter(
            name,
            v.iter().map(|s| {
                NaiveDateTime::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(|v| naive_datetime_to_date64(v))
            }),
        )
    }
}

impl Date32Chunked {
    /// Extract month from underlying NaiveDate representation.
    /// Returns the year number in the calendar date.
    pub fn year(&self) -> Int32Chunked {
        self.apply_kernel_cast::<_, Int32Type>(date32_to_year)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    pub fn month(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date32_to_month)
    }

    /// Extract weekday from underlying NaiveDate representation.
    /// Returns the weekday number where monday = 0 and sunday = 6
    pub fn weekday(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date32_to_weekday)
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    pub fn week(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date32_to_week)
    }

    /// Extract day from underlying NaiveDate representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    pub fn day(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date32_to_day)
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    pub fn ordinal(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date32_to_ordinal)
    }

    /// Format Date32 with a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(&self, fmt: &str) -> Utf8Chunked {
        let mut ca: Utf8Chunked = self
            .as_naive_datetime_iter()
            .map(|opt_dt| opt_dt.map(|dt| format!("{}", dt.format(fmt))))
            .collect();
        ca.rename(self.name());
        ca
    }

    pub fn new_from_naive_date(name: &str, v: &[NaiveDate]) -> Self {
        let unit = v
            .iter()
            .map(|v| naive_date_to_date32(*v))
            .collect::<AlignedVec<_>>();
        ChunkedArray::new_from_aligned_vec(name, unit)
    }

    pub fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
        ChunkedArray::new_from_opt_iter(
            name,
            v.iter().map(|s| {
                NaiveDate::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(|v| naive_date_to_date32(*v))
            }),
        )
    }
}
