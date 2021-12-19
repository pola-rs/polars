use super::*;
use crate::prelude::AnyValue;
#[cfg(feature = "dtype-time")]
use arrow::temporal_conversions::time64ns_to_time;
use arrow::temporal_conversions::{timestamp_ms_to_datetime, NANOSECONDS};
use polars_time::export::chrono::{NaiveDateTime, NaiveTime};

/// Number of seconds in a day
pub(crate) const SECONDS_IN_DAY: i64 = 86_400;

impl From<&AnyValue<'_>> for NaiveDateTime {
    fn from(v: &AnyValue) -> Self {
        match v {
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => NaiveDateTime::from_timestamp(*v as i64 * SECONDS_IN_DAY, 0),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v) => timestamp_ms_to_datetime(*v),
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

// datetime is number of milliseconds since the Unix Epoch
// Used by lazy for literal conversion
#[cfg(feature = "private")]
pub fn naive_datetime_to_datetime(v: &NaiveDateTime) -> i64 {
    v.timestamp_nanos()
}

pub(crate) fn naive_datetime_to_date(v: &NaiveDateTime) -> i32 {
    (naive_datetime_to_datetime(v) / (NANOSECONDS * SECONDS_IN_DAY)) as i32
}

pub(crate) fn naive_date_to_date(nd: NaiveDate) -> i32 {
    let nt = NaiveTime::from_hms(0, 0, 0);
    let ndt = NaiveDateTime::new(nd, nt);
    naive_datetime_to_date(&ndt)
}
