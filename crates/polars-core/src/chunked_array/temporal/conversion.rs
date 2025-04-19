use arrow::temporal_conversions::*;
use chrono::*;

use crate::prelude::*;

pub(crate) const NS_IN_DAY: i64 = 86_400_000_000_000;
pub(crate) const US_IN_DAY: i64 = 86_400_000_000;
pub(crate) const MS_IN_DAY: i64 = 86_400_000;
pub(crate) const SECONDS_IN_DAY: i64 = 86_400;

impl From<&AnyValue<'_>> for NaiveDateTime {
    fn from(v: &AnyValue) -> Self {
        match v {
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => date32_to_datetime(*v),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, _) => match tu {
                TimeUnit::Nanoseconds => timestamp_ns_to_datetime(*v),
                TimeUnit::Microseconds => timestamp_us_to_datetime(*v),
                TimeUnit::Milliseconds => timestamp_ms_to_datetime(*v),
            },
            _ => panic!("can only convert date/datetime to NaiveDateTime"),
        }
    }
}

impl From<&AnyValue<'_>> for NaiveTime {
    fn from(v: &AnyValue) -> Self {
        match v {
            #[cfg(feature = "dtype-time")]
            AnyValue::Time(v) => time64ns_to_time(*v),
            _ => panic!("can only convert date/datetime to NaiveTime"),
        }
    }
}

// Used by lazy for literal conversion
pub fn datetime_to_timestamp_ns(v: NaiveDateTime) -> i64 {
    let us = v.and_utc().timestamp() * 1_000_000_000; //nanos
    us + v.and_utc().timestamp_subsec_nanos() as i64
}

pub fn datetime_to_timestamp_ms(v: NaiveDateTime) -> i64 {
    v.and_utc().timestamp_millis()
}

pub fn datetime_to_timestamp_us(v: NaiveDateTime) -> i64 {
    let us = v.and_utc().timestamp() * 1_000_000;
    us + v.and_utc().timestamp_subsec_micros() as i64
}

pub(crate) fn naive_datetime_to_date(v: NaiveDateTime) -> i32 {
    (datetime_to_timestamp_ms(v) / (MILLISECONDS * SECONDS_IN_DAY)) as i32
}

pub fn get_strftime_format(fmt: &str, dtype: &DataType) -> PolarsResult<String> {
    if fmt == "polars" && !matches!(dtype, DataType::Duration(_)) {
        polars_bail!(InvalidOperation: "'polars' is not a valid `to_string` format for {} dtype expressions", dtype);
    } else {
        let format_string = if fmt != "iso" && fmt != "iso:strict" {
            fmt.to_string()
        } else {
            let sep = if fmt == "iso" { " " } else { "T" };
            #[allow(unreachable_code)]
            match dtype {
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(tu, tz) => match (tu, tz.is_some()) {
                    (TimeUnit::Milliseconds, true) => format!("%F{}%T%.3f%:z", sep),
                    (TimeUnit::Milliseconds, false) => format!("%F{}%T%.3f", sep),
                    (TimeUnit::Microseconds, true) => format!("%F{}%T%.6f%:z", sep),
                    (TimeUnit::Microseconds, false) => format!("%F{}%T%.6f", sep),
                    (TimeUnit::Nanoseconds, true) => format!("%F{}%T%.9f%:z", sep),
                    (TimeUnit::Nanoseconds, false) => format!("%F{}%T%.9f", sep),
                },
                #[cfg(feature = "dtype-date")]
                DataType::Date => "%F".to_string(),
                #[cfg(feature = "dtype-time")]
                DataType::Time => "%T%.f".to_string(),
                _ => {
                    let err = format!(
                        "invalid call to `get_strftime_format`; fmt={:?}, dtype={}",
                        fmt, dtype
                    );
                    unimplemented!("{}", err)
                },
            }
        };
        Ok(format_string)
    }
}
