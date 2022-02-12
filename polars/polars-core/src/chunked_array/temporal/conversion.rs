use super::*;
use crate::prelude::{AnyValue, TimeUnit};
#[cfg(feature = "dtype-time")]
use arrow::temporal_conversions::time64ns_to_time;
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
};
use chrono::{NaiveDateTime, NaiveTime};

/// Number of seconds in a day
pub(crate) const SECONDS_IN_DAY: i64 = 86_400;

impl From<&AnyValue<'_>> for NaiveDateTime {
    fn from(v: &AnyValue) -> Self {
        match v {
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => NaiveDateTime::from_timestamp(*v as i64 * SECONDS_IN_DAY, 0),
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
#[cfg(feature = "private")]
pub fn datetime_to_timestamp_ns(v: NaiveDateTime) -> i64 {
    v.timestamp_nanos()
}

// Used by lazy for literal conversion
#[cfg(feature = "private")]
pub fn datetime_to_timestamp_ms(v: NaiveDateTime) -> i64 {
    v.timestamp_millis()
}

// Used by lazy for literal conversion
#[cfg(feature = "private")]
pub fn datetime_to_timestamp_us(v: NaiveDateTime) -> i64 {
    let us = v.timestamp() * 1_000_000;
    us + v.timestamp_subsec_micros() as i64
}

pub(crate) fn naive_datetime_to_date(v: NaiveDateTime) -> i32 {
    (datetime_to_timestamp_ms(v) / (MILLISECONDS * SECONDS_IN_DAY)) as i32
}

pub(crate) fn naive_date_to_date(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms(0, 0, 0);
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date(ndt)
}
pub(crate) const NS_IN_DAY: i64 = 86_400_000_000_000;
pub(crate) const US_IN_DAY: i64 = 86_400_000_000;
pub(crate) const MS_IN_DAY: i64 = 86_400_000;
