use super::*;
use crate::chunked_array::kernels::temporal::{
    date32_as_duration, date32_to_day, date32_to_month, date32_to_ordinal, date32_to_year,
    date64_as_duration, date64_to_day, date64_to_hour, date64_to_minute, date64_to_month,
    date64_to_nanosecond, date64_to_ordinal, date64_to_second,
};
use crate::prelude::*;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime};
use regex::Regex;

pub trait FromNaiveTime<T, N> {
    fn new_from_naive_time(name: &str, v: &[N]) -> Self;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self;
}

fn parse_naive_time_from_str(s: &str, fmt: &str) -> Option<NaiveTime> {
    NaiveTime::parse_from_str(s, fmt).ok()
}

macro_rules! impl_from_naive_time {
    ($arrowtype:ident, $chunkedtype:ident, $func:ident) => {
        impl FromNaiveTime<$arrowtype, NaiveTime> for $chunkedtype {
            fn new_from_naive_time(name: &str, v: &[NaiveTime]) -> Self {
                let unit = v.iter().map($func).collect::<AlignedVec<_>>();
                ChunkedArray::new_from_aligned_vec(name, unit)
            }

            fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
                ChunkedArray::new_from_opt_iter(
                    name,
                    v.iter()
                        .map(|s| parse_naive_time_from_str(s, fmt).as_ref().map($func)),
                )
            }
        }
    };
}

impl_from_naive_time!(
    Time64NanosecondType,
    Time64NanosecondChunked,
    naive_time_to_time64_nanoseconds
);
impl_from_naive_time!(
    Time64MicrosecondType,
    Time64MicrosecondChunked,
    naive_time_to_time64_microseconds
);
impl_from_naive_time!(
    Time32MillisecondType,
    Time32MillisecondChunked,
    naive_time_to_time32_milliseconds
);
impl_from_naive_time!(
    Time32SecondType,
    Time32SecondChunked,
    naive_time_to_time32_seconds
);

pub trait AsNaiveTime {
    fn as_naive_time(&self) -> Vec<Option<NaiveTime>>;
}

macro_rules! impl_as_naivetime {
    ($ca:ty, $fun:ident) => {
        impl AsNaiveTime for $ca {
            fn as_naive_time(&self) -> Vec<Option<NaiveTime>> {
                self.into_iter().map(|opt_t| opt_t.map($fun)).collect()
            }
        }
    };
}

impl_as_naivetime!(Time32SecondChunked, time32_second_as_time);
impl_as_naivetime!(Time32MillisecondChunked, time32_millisecond_as_time);
impl_as_naivetime!(Time64NanosecondChunked, time64_nanosecond_as_time);
impl_as_naivetime!(Time64MicrosecondChunked, time64_microsecond_as_time);

pub fn parse_naive_datetime_from_str(s: &str, fmt: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s, fmt).ok()
}

pub trait FromNaiveDateTime<T, N> {
    fn new_from_naive_datetime(name: &str, v: &[N]) -> Self;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self;
}

macro_rules! impl_from_naive_datetime {
    ($arrowtype:ident, $chunkedtype:ident, $func:ident) => {
        impl FromNaiveDateTime<$arrowtype, NaiveDateTime> for $chunkedtype {
            fn new_from_naive_datetime(name: &str, v: &[NaiveDateTime]) -> Self {
                let unit = v.iter().map($func).collect::<AlignedVec<_>>();
                ChunkedArray::new_from_aligned_vec(name, unit)
            }

            fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
                ChunkedArray::new_from_opt_iter(
                    name,
                    v.iter()
                        .map(|s| parse_naive_datetime_from_str(s, fmt).as_ref().map($func)),
                )
            }
        }
    };
}

impl_from_naive_datetime!(Date64Type, Date64Chunked, naive_datetime_to_date64);
impl_from_naive_datetime!(
    TimestampNanosecondType,
    TimestampNanosecondChunked,
    naive_datetime_to_timestamp_nanoseconds
);
impl_from_naive_datetime!(
    TimestampMicrosecondType,
    TimestampMicrosecondChunked,
    naive_datetime_to_timestamp_microseconds
);
impl_from_naive_datetime!(
    TimestampMillisecondType,
    TimestampMillisecondChunked,
    naive_datetime_to_timestamp_milliseconds
);
impl_from_naive_datetime!(
    TimestampSecondType,
    TimestampSecondChunked,
    naive_datetime_to_timestamp_seconds
);

pub trait FromNaiveDate<T, N> {
    fn new_from_naive_date(name: &str, v: &[N]) -> Self;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self;
}

pub fn naive_date_to_date32(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms(0, 0, 0);
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date32(&ndt)
}

pub fn parse_naive_date_from_str(s: &str, fmt: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s, fmt).ok()
}

fn unix_time_naive_date() -> NaiveDate {
    NaiveDate::from_ymd(1970, 1, 1)
}

impl FromNaiveDate<Date32Type, NaiveDate> for Date32Chunked {
    fn new_from_naive_date(name: &str, v: &[NaiveDate]) -> Self {
        let unit = v
            .iter()
            .map(|v| naive_date_to_date32(*v))
            .collect::<AlignedVec<_>>();
        ChunkedArray::new_from_aligned_vec(name, unit)
    }

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
        ChunkedArray::new_from_opt_iter(
            name,
            v.iter().map(|s| {
                parse_naive_date_from_str(s, fmt)
                    .as_ref()
                    .map(|v| naive_date_to_date32(*v))
            }),
        )
    }
}

pub trait AsNaiveDateTime {
    fn as_naive_datetime(&self) -> Vec<Option<NaiveDateTime>>;
}

macro_rules! impl_as_naive_datetime {
    ($ca:ty, $fun:ident) => {
        impl AsNaiveDateTime for $ca {
            fn as_naive_datetime(&self) -> Vec<Option<NaiveDateTime>> {
                self.into_iter().map(|opt_t| opt_t.map($fun)).collect()
            }
        }
    };
}

impl_as_naive_datetime!(Date32Chunked, date32_as_datetime);
impl_as_naive_datetime!(Date64Chunked, date64_as_datetime);
impl_as_naive_datetime!(
    TimestampNanosecondChunked,
    timestamp_nanoseconds_as_datetime
);
impl_as_naive_datetime!(
    TimestampMicrosecondChunked,
    timestamp_microseconds_as_datetime
);
impl_as_naive_datetime!(
    TimestampMillisecondChunked,
    timestamp_milliseconds_as_datetime
);
impl_as_naive_datetime!(TimestampSecondChunked, timestamp_seconds_as_datetime);

pub trait AsNaiveDate {
    fn as_naive_date(&self) -> Vec<Option<NaiveDate>>;
}

impl AsNaiveDate for Date32Chunked {
    fn as_naive_date(&self) -> Vec<Option<NaiveDate>> {
        self.into_iter()
            .map(|opt_t| {
                opt_t.map(|v| {
                    let dt = date32_as_datetime(v);
                    NaiveDate::from_ymd(dt.year(), dt.month(), dt.day())
                })
            })
            .collect()
    }
}

pub trait AsDuration<T> {
    fn as_duration(&self) -> ChunkedArray<T>;
}

impl AsDuration<DurationSecondType> for Date32Chunked {
    fn as_duration(&self) -> DurationSecondChunked {
        self.apply_kernel_cast(date32_as_duration)
    }
}

impl AsDuration<DurationMillisecondType> for Date64Chunked {
    fn as_duration(&self) -> DurationMillisecondChunked {
        self.apply_kernel_cast(date64_as_duration)
    }
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
        let pat = r"^\d{4}-\d{1,2}-\d{1,2} \d{2}:\d{2}:\d{2}\s*$";
        let reg = Regex::new(pat).expect("wrong regex");
        if reg.is_match(val) {
            return Ok("%Y-%m-%d %H:%M:%S");
        }
        let pat = r"^\d{4}/\d{1,2}/\d{1,2} \d{2}:\d{2}:\d{2}\s*$";
        let reg = Regex::new(pat).expect("wrong regex");
        if reg.is_match(val) {
            return Ok("%Y/%m/%d %H:%M:%S");
        }
        Err(PolarsError::Other(
            "Could not find an appropriate format to parse dates, please define a fmt".into(),
        ))
    }

    fn sniff_fmt_date32(&self) -> Result<&'static str> {
        let val = self.get_first_val()?;
        let pat = r"^\d{4}-\d{1,2}-\d{1,2}\s*$";
        let reg = Regex::new(pat).expect("wrong regex");
        if reg.is_match(val) {
            return Ok("%Y-%m-%d");
        }

        let pat = r"^\d{1,2}-\d{1,2}-\d{4}\s*$";
        let reg = Regex::new(pat).expect("wrong regex");
        if reg.is_match(val) {
            return Ok("%d-%m-%Y");
        }

        let pat = r"^\d{4}/\d{1,2}/\d{1,2}\s*$";
        let reg = Regex::new(pat).expect("wrong regex");
        if reg.is_match(val) {
            return Ok("%Y/%m/%d %H:%M:%S");
        }
        Err(PolarsError::Other(
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
                .map(|s| parse_naive_date_from_str(s, fmt).map(naive_date_to_date32))
                .collect(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd =
                        opt_s.map(|s| parse_naive_date_from_str(s, fmt).map(naive_date_to_date32));
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect(),
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
                    parse_naive_datetime_from_str(s, fmt).map(|dt| naive_datetime_to_date64(&dt))
                })
                .collect(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        parse_naive_datetime_from_str(s, fmt)
                            .map(|dt| naive_datetime_to_date64(&dt))
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect(),
        };
        ca.rename(self.name());
        Ok(ca)
    }
}

impl Date64Chunked {
    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    pub fn year(&self) -> Int32Chunked {
        self.apply_kernel_cast::<_, Int32Type>(date64_to_month)
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    pub fn month(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<_, UInt32Type>(date64_to_month)
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
}

impl Date32Chunked {
    /// Extract month from underlying NaiveDateTime representation.
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

    /// Extract day from underlying NaiveDateTime representation.
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
}
