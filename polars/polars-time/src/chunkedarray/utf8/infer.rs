use chrono::{NaiveDate, NaiveDateTime};
use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_core::prelude::*;
use polars_core::utils::arrow::types::NativeType;

use super::patterns;
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
use crate::chunkedarray::utf8::patterns::Pattern;
use crate::chunkedarray::utf8::strptime;

#[derive(Clone)]
pub struct DatetimeInfer<T> {
    patterns: &'static [&'static str],
    latest: &'static str,
    transform: fn(&str, &str) -> Option<T>,
    transform_bytes: fn(&[u8], &[u8], u16) -> Option<T>,
    fmt_len: u16,
    pub logical_type: DataType,
}

impl TryFrom<Pattern> for DatetimeInfer<i64> {
    type Error = PolarsError;

    fn try_from(value: Pattern) -> PolarsResult<Self> {
        match value {
            Pattern::DatetimeDMY => Ok(DatetimeInfer {
                patterns: patterns::DATETIME_D_M_Y,
                latest: patterns::DATETIME_D_M_Y[0],
                transform: transform_datetime_us,
                transform_bytes: transform_datetime_us_bytes,
                fmt_len: 0,
                logical_type: DataType::Datetime(TimeUnit::Microseconds, None),
            }),
            Pattern::DatetimeYMD => Ok(DatetimeInfer {
                patterns: patterns::DATETIME_Y_M_D,
                latest: patterns::DATETIME_Y_M_D[0],
                transform: transform_datetime_us,
                transform_bytes: transform_datetime_us_bytes,
                fmt_len: 0,
                logical_type: DataType::Datetime(TimeUnit::Microseconds, None),
            }),
            _ => Err(PolarsError::ComputeError(
                "could not convert pattern".into(),
            )),
        }
    }
}

#[cfg(feature = "dtype-date")]
impl TryFrom<Pattern> for DatetimeInfer<i32> {
    type Error = PolarsError;

    fn try_from(value: Pattern) -> PolarsResult<Self> {
        match value {
            Pattern::DateDMY => Ok(DatetimeInfer {
                patterns: patterns::DATE_D_M_Y,
                latest: patterns::DATE_D_M_Y[0],
                transform: transform_date,
                transform_bytes: transform_date_bytes,
                fmt_len: 0,
                logical_type: DataType::Date,
            }),
            Pattern::DateYMD => Ok(DatetimeInfer {
                patterns: patterns::DATE_Y_M_D,
                latest: patterns::DATE_Y_M_D[0],
                transform: transform_date,
                transform_bytes: transform_date_bytes,
                fmt_len: 0,
                logical_type: DataType::Date,
            }),
            _ => Err(PolarsError::ComputeError(
                "could not convert pattern".into(),
            )),
        }
    }
}

impl<T: NativeType> DatetimeInfer<T> {
    pub fn parse(&mut self, val: &str) -> Option<T> {
        match (self.transform)(val, self.latest) {
            Some(parsed) => Some(parsed),
            // try other patterns
            None => {
                for fmt in self.patterns {
                    self.fmt_len = 0;
                    if let Some(parsed) = (self.transform)(val, fmt) {
                        self.latest = fmt;
                        return Some(parsed);
                    }
                }
                None
            }
        }
    }

    pub fn parse_bytes(&mut self, val: &[u8]) -> Option<T> {
        if self.fmt_len == 0 {
            self.fmt_len = strptime::fmt_len(self.latest.as_bytes())?;
        }
        match (self.transform_bytes)(val, self.latest.as_bytes(), self.fmt_len) {
            Some(parsed) => Some(parsed),
            // try other patterns
            None => {
                for fmt in self.patterns {
                    if self.fmt_len == 0 {
                        self.fmt_len = strptime::fmt_len(fmt.as_bytes())?;
                    }
                    if let Some(parsed) = (self.transform_bytes)(val, fmt.as_bytes(), self.fmt_len)
                    {
                        self.latest = fmt;
                        return Some(parsed);
                    }
                }
                None
            }
        }
    }

    fn coerce_utf8(&mut self, ca: &Utf8Chunked) -> Series {
        let chunks = ca
            .downcast_iter()
            .map(|array| {
                let iter = array
                    .into_iter()
                    .map(|opt_val| opt_val.and_then(|val| self.parse(val)));
                Box::new(PrimitiveArray::from_trusted_len_iter(iter)) as ArrayRef
            })
            .collect();
        let mut out = match self.logical_type {
            DataType::Date => unsafe { Int32Chunked::from_chunks(ca.name(), chunks) }
                .into_series()
                .cast(&self.logical_type)
                .unwrap(),
            DataType::Datetime(_, _) => unsafe { Int64Chunked::from_chunks(ca.name(), chunks) }
                .into_series()
                .cast(&self.logical_type)
                .unwrap(),
            _ => unreachable!(),
        };
        out.rename(ca.name());
        out
    }
}

#[cfg(feature = "dtype-date")]
fn transform_date(val: &str, fmt: &str) -> Option<i32> {
    NaiveDate::parse_from_str(val, fmt)
        .ok()
        .map(naive_date_to_date)
}

#[cfg(feature = "dtype-date")]
fn transform_date_bytes(val: &[u8], fmt: &[u8], fmt_len: u16) -> Option<i32> {
    unsafe { strptime::parse(val, fmt, fmt_len).map(|ndt| naive_date_to_date(ndt.date())) }
}

fn transform_datetime_ns(val: &str, fmt: &str) -> Option<i64> {
    let out = NaiveDateTime::parse_from_str(val, fmt)
        .ok()
        .map(datetime_to_timestamp_ns);
    out.or_else(|| {
        NaiveDate::parse_from_str(val, fmt)
            .ok()
            .map(|nd| datetime_to_timestamp_ns(nd.and_hms_opt(0, 0, 0).unwrap()))
    })
}

fn transform_datetime_us(val: &str, fmt: &str) -> Option<i64> {
    let out = NaiveDateTime::parse_from_str(val, fmt)
        .ok()
        .map(datetime_to_timestamp_us);
    out.or_else(|| {
        NaiveDate::parse_from_str(val, fmt)
            .ok()
            .map(|nd| datetime_to_timestamp_us(nd.and_hms_opt(0, 0, 0).unwrap()))
    })
}

fn transform_datetime_us_bytes(val: &[u8], fmt: &[u8], fmt_len: u16) -> Option<i64> {
    unsafe { strptime::parse(val, fmt, fmt_len).map(datetime_to_timestamp_us) }
}

fn transform_datetime_ms(val: &str, fmt: &str) -> Option<i64> {
    let out = NaiveDateTime::parse_from_str(val, fmt)
        .ok()
        .map(datetime_to_timestamp_ms);
    out.or_else(|| {
        NaiveDate::parse_from_str(val, fmt)
            .ok()
            .map(|nd| datetime_to_timestamp_ms(nd.and_hms_opt(0, 0, 0).unwrap()))
    })
}

pub fn infer_pattern_single(val: &str) -> Option<Pattern> {
    // Dates come first, because we see datetimes as superset of dates
    infer_pattern_date_single(val).or_else(|| infer_pattern_datetime_single(val))
}

fn infer_pattern_datetime_single(val: &str) -> Option<Pattern> {
    if patterns::DATETIME_D_M_Y.iter().any(|fmt| {
        NaiveDateTime::parse_from_str(val, fmt).is_ok()
            || NaiveDate::parse_from_str(val, fmt).is_ok()
    }) {
        Some(Pattern::DatetimeDMY)
    } else if patterns::DATETIME_Y_M_D.iter().any(|fmt| {
        NaiveDateTime::parse_from_str(val, fmt).is_ok()
            || NaiveDate::parse_from_str(val, fmt).is_ok()
    }) {
        Some(Pattern::DatetimeYMD)
    } else {
        None
    }
}

fn infer_pattern_date_single(val: &str) -> Option<Pattern> {
    if patterns::DATE_D_M_Y
        .iter()
        .any(|fmt| NaiveDate::parse_from_str(val, fmt).is_ok())
    {
        Some(Pattern::DateDMY)
    } else if patterns::DATE_Y_M_D
        .iter()
        .any(|fmt| NaiveDate::parse_from_str(val, fmt).is_ok())
    {
        Some(Pattern::DateYMD)
    } else {
        None
    }
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn to_datetime(
    ca: &Utf8Chunked,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    match ca.first_non_null() {
        None => Ok(Int64Chunked::full_null(ca.name(), ca.len()).into_datetime(tu, tz.cloned())),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern = subset
                .into_iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_datetime_single))
                .ok_or_else(|| {
                    PolarsError::ComputeError(
                        "Could not find an appropriate format to parse dates, please define a fmt"
                            .into(),
                    )
                })?;
            let mut infer = DatetimeInfer::<i64>::try_from(pattern).unwrap();
            match tu {
                TimeUnit::Nanoseconds => infer.transform = transform_datetime_ns,
                TimeUnit::Microseconds => infer.transform = transform_datetime_us,
                TimeUnit::Milliseconds => infer.transform = transform_datetime_ms,
            }
            infer.coerce_utf8(ca).datetime().map(|ca| {
                let mut ca = ca.clone();
                ca.set_time_unit(tu);
                match tz {
                    #[cfg(feature = "timezones")]
                    Some(tz) => Ok(ca.replace_time_zone(Some(tz))?),
                    _ => Ok(ca),
                }
            })?
        }
    }
}
#[cfg(feature = "dtype-date")]
pub(crate) fn to_date(ca: &Utf8Chunked) -> PolarsResult<DateChunked> {
    match ca.first_non_null() {
        None => Ok(Int32Chunked::full_null(ca.name(), ca.len()).into_date()),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern = subset
                .into_iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_date_single))
                .ok_or_else(|| {
                    PolarsError::ComputeError(
                        "Could not find an appropriate format to parse dates, please define a fmt"
                            .into(),
                    )
                })?;
            let mut infer = DatetimeInfer::<i32>::try_from(pattern).unwrap();
            infer.coerce_utf8(ca).date().cloned()
        }
    }
}
