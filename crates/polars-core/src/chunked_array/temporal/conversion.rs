use arrow::temporal_conversions::*;
use jiff::civil::{DateTime, Time};

use crate::prelude::*;

pub(crate) const NS_IN_DAY: i64 = 86_400_000_000_000;
pub(crate) const US_IN_DAY: i64 = 86_400_000_000;
pub(crate) const MS_IN_DAY: i64 = 86_400_000;
pub(crate) const SECONDS_IN_DAY: i64 = 86_400;

impl From<&AnyValue<'_>> for DateTime {
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
            _ => panic!("can only convert date/datetime to DateTime"),
        }
    }
}

impl From<&AnyValue<'_>> for Time {
    fn from(v: &AnyValue) -> Self {
        match v {
            #[cfg(feature = "dtype-time")]
            AnyValue::Time(v) => time64ns_to_time(*v),
            _ => panic!("can only convert date/datetime to Time"),
        }
    }
}

// Used by lazy for literal conversion
pub fn datetime_to_timestamp_ns(v: DateTime) -> i64 {
    let ns = datetime_to_epoch_nanos_opt(v).expect("datetime out-of-range");
    i64::try_from(ns).expect("timestamp out-of-range for i64 nanoseconds")
}

pub fn datetime_to_timestamp_ms(v: DateTime) -> i64 {
    let ns = datetime_to_epoch_nanos_opt(v).expect("datetime out-of-range");
    i64::try_from(ns.div_euclid(1_000_000)).expect("timestamp out-of-range for i64 milliseconds")
}

pub fn datetime_to_timestamp_us(v: DateTime) -> i64 {
    let ns = datetime_to_epoch_nanos_opt(v).expect("datetime out-of-range");
    i64::try_from(ns.div_euclid(1_000)).expect("timestamp out-of-range for i64 microseconds")
}

pub(crate) fn naive_datetime_to_date(v: DateTime) -> i32 {
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
                    (TimeUnit::Milliseconds, true) => format!("%F{sep}%T%.3f%:z"),
                    (TimeUnit::Milliseconds, false) => format!("%F{sep}%T%.3f"),
                    (TimeUnit::Microseconds, true) => format!("%F{sep}%T%.6f%:z"),
                    (TimeUnit::Microseconds, false) => format!("%F{sep}%T%.6f"),
                    (TimeUnit::Nanoseconds, true) => format!("%F{sep}%T%.9f%:z"),
                    (TimeUnit::Nanoseconds, false) => format!("%F{sep}%T%.9f"),
                },
                #[cfg(feature = "dtype-date")]
                DataType::Date => "%F".to_string(),
                #[cfg(feature = "dtype-time")]
                DataType::Time => "%T%.f".to_string(),
                _ => {
                    let err = format!(
                        "invalid call to `get_strftime_format`; fmt={fmt:?}, dtype={dtype}"
                    );
                    unimplemented!("{}", err)
                },
            }
        };
        Ok(format_string)
    }
}
