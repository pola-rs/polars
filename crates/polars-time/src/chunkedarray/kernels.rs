//! macros that define the extraction of `week`, `weekday`, `year`, `hour` etc. from one value.
//!
//! Each of these is a function of a single physical value, applied over a column by
//! [`unary_elementwise`](polars_core::prelude::arity::unary_elementwise). They used to be array
//! kernels, taking and building an Arrow array of their own; nothing about what they compute
//! needed the array, and reading one value at a time is what lets the caller decide how the
//! column is walked.
#[cfg(feature = "dtype-time")]
use arrow::temporal_conversions::time64ns_to_time_opt;
use arrow::temporal_conversions::{
    date32_to_datetime_opt, timestamp_ms_to_datetime_opt, timestamp_ns_to_datetime_opt,
    timestamp_us_to_datetime_opt,
};
use chrono::{Datelike, Timelike};

use super::super::windows::calendar::*;
use super::*;

trait PolarsIso {
    fn week(&self) -> i8;
    fn iso_year(&self) -> i32;
    /// The day of the week as Monday = 1 through Sunday = 7.
    fn weekday_number(&self) -> i8;
}

impl PolarsIso for NaiveDateTime {
    fn week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
    fn iso_year(&self) -> i32 {
        self.iso_week().year()
    }
    fn weekday_number(&self) -> i8 {
        self.weekday().number_from_monday().try_into().unwrap()
    }
}

impl PolarsIso for NaiveDate {
    fn week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
    fn iso_year(&self) -> i32 {
        self.iso_week().year()
    }
    fn weekday_number(&self) -> i8 {
        self.weekday().number_from_monday().try_into().unwrap()
    }
}

macro_rules! to_temporal_unit {
    ($name: ident, $chrono_method: ident, $to_datetime_fn: expr,
    $primitive_in: ty,
    $primitive_out: ty) => {
        pub(crate) fn $name(value: $primitive_in) -> Option<$primitive_out> {
            $to_datetime_fn(value).map(|dt| dt.$chrono_method() as $primitive_out)
        }
    };
}

macro_rules! to_boolean_temporal_unit {
    ($name: ident, $chrono_method: ident, $boolean_method: ident, $to_datetime_fn: expr, $dtype_in: ty) => {
        pub(crate) fn $name(value: $dtype_in) -> Option<bool> {
            $to_datetime_fn(value).map(|dt| $boolean_method(dt.$chrono_method()))
        }
    };
}

macro_rules! to_calendar_value {
    ($name: ident, $dt: ident, $expr: expr, $to_datetime_fn: expr,
    $primitive_in: ty,
    $primitive_out: ty) => {
        pub(crate) fn $name(value: $primitive_in) -> Option<$primitive_out> {
            $to_datetime_fn(value).map(|$dt| $expr as $primitive_out)
        }
    };
}

// Dates
#[cfg(feature = "dtype-date")]
to_temporal_unit!(date_to_iso_week, week, date32_to_datetime_opt, i32, i8);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(date_to_iso_year, iso_year, date32_to_datetime_opt, i32, i32);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(date_to_year, year, date32_to_datetime_opt, i32, i32);
#[cfg(feature = "dtype-date")]
to_boolean_temporal_unit!(
    date_to_is_leap_year,
    year,
    is_leap_year,
    date32_to_datetime_opt,
    i32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(date_to_month, month, date32_to_datetime_opt, i32, i8);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(date_to_day, day, date32_to_datetime_opt, i32, i8);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(date_to_ordinal, ordinal, date32_to_datetime_opt, i32, i16);
#[cfg(feature = "dtype-date")]
to_calendar_value!(
    date_to_days_in_month,
    dt,
    days_in_month(dt.year(), dt.month() as u8),
    date32_to_datetime_opt,
    i32,
    i8
);

// Times
#[cfg(feature = "dtype-time")]
to_temporal_unit!(time_to_hour, hour, time64ns_to_time_opt, i64, i8);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(time_to_minute, minute, time64ns_to_time_opt, i64, i8);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(time_to_second, second, time64ns_to_time_opt, i64, i8);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_nanosecond,
    nanosecond,
    time64ns_to_time_opt,
    i64,
    i32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_ns,
    ordinal,
    timestamp_ns_to_datetime_opt,
    i64,
    i16
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_ms,
    ordinal,
    timestamp_ms_to_datetime_opt,
    i64,
    i16
);
#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_us,
    ordinal,
    timestamp_us_to_datetime_opt,
    i64,
    i16
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_ns,
    iso_year,
    timestamp_ns_to_datetime_opt,
    i64,
    i32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_us,
    iso_year,
    timestamp_us_to_datetime_opt,
    i64,
    i32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_ms,
    iso_year,
    timestamp_ms_to_datetime_opt,
    i64,
    i32
);
#[cfg(feature = "dtype-datetime")]
to_boolean_temporal_unit!(
    datetime_to_is_leap_year_ns,
    year,
    is_leap_year,
    timestamp_ns_to_datetime_opt,
    i64
);
#[cfg(feature = "dtype-datetime")]
to_boolean_temporal_unit!(
    datetime_to_is_leap_year_us,
    year,
    is_leap_year,
    timestamp_us_to_datetime_opt,
    i64
);
#[cfg(feature = "dtype-datetime")]
to_boolean_temporal_unit!(
    datetime_to_is_leap_year_ms,
    year,
    is_leap_year,
    timestamp_ms_to_datetime_opt,
    i64
);

#[cfg(feature = "dtype-datetime")]
to_calendar_value!(
    datetime_to_days_in_month_ns,
    dt,
    days_in_month(dt.year(), dt.month() as u8),
    timestamp_ns_to_datetime_opt,
    i64,
    i8
);
#[cfg(feature = "dtype-datetime")]
to_calendar_value!(
    datetime_to_days_in_month_us,
    dt,
    days_in_month(dt.year(), dt.month() as u8),
    timestamp_us_to_datetime_opt,
    i64,
    i8
);
#[cfg(feature = "dtype-datetime")]
to_calendar_value!(
    datetime_to_days_in_month_ms,
    dt,
    days_in_month(dt.year(), dt.month() as u8),
    timestamp_ms_to_datetime_opt,
    i64,
    i8
);

/// Defines the same extraction over each of the three timestamp units a datetime column can be in.
macro_rules! datetime_units {
    ($ns: ident, $us: ident, $ms: ident, $chrono_method: ident, $primitive_out: ty) => {
        #[cfg(feature = "dtype-datetime")]
        to_temporal_unit!(
            $ns,
            $chrono_method,
            timestamp_ns_to_datetime_opt,
            i64,
            $primitive_out
        );
        #[cfg(feature = "dtype-datetime")]
        to_temporal_unit!(
            $us,
            $chrono_method,
            timestamp_us_to_datetime_opt,
            i64,
            $primitive_out
        );
        #[cfg(feature = "dtype-datetime")]
        to_temporal_unit!(
            $ms,
            $chrono_method,
            timestamp_ms_to_datetime_opt,
            i64,
            $primitive_out
        );
    };
}

datetime_units!(
    datetime_to_year_ns,
    datetime_to_year_us,
    datetime_to_year_ms,
    year,
    i32
);
datetime_units!(
    datetime_to_month_ns,
    datetime_to_month_us,
    datetime_to_month_ms,
    month,
    i8
);
datetime_units!(
    datetime_to_day_ns,
    datetime_to_day_us,
    datetime_to_day_ms,
    day,
    i8
);
datetime_units!(
    datetime_to_hour_ns,
    datetime_to_hour_us,
    datetime_to_hour_ms,
    hour,
    i8
);
datetime_units!(
    datetime_to_minute_ns,
    datetime_to_minute_us,
    datetime_to_minute_ms,
    minute,
    i8
);
datetime_units!(
    datetime_to_second_ns,
    datetime_to_second_us,
    datetime_to_second_ms,
    second,
    i8
);
datetime_units!(
    datetime_to_nanosecond_ns,
    datetime_to_nanosecond_us,
    datetime_to_nanosecond_ms,
    nanosecond,
    i32
);
datetime_units!(
    datetime_to_weekday_ns,
    datetime_to_weekday_us,
    datetime_to_weekday_ms,
    weekday_number,
    i8
);
datetime_units!(
    datetime_to_iso_week_ns,
    datetime_to_iso_week_us,
    datetime_to_iso_week_ms,
    week,
    i8
);
