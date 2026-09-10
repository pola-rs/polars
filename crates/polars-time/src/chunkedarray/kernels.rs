//! macros that define the extraction of `week`, `weekday`, `year`, `hour` etc. from one value.
#[cfg(feature = "dtype-date")]
use arrow::temporal_conversions::date32_to_datetime_opt;
#[cfg(feature = "dtype-time")]
use arrow::temporal_conversions::time64ns_to_time_opt;
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

/// Each of these carries the timestamp conversion as well as the field, and is called once per
/// element by the elementwise applies below. `#[inline]` is what lets the conversion and the
/// chrono arithmetic behind it fold into the caller's loop, as they did when these were kernels
/// over a whole chunk: without it `date.year` over a million dates costs some 15% more
/// instructions.
macro_rules! to_temporal_unit {
    ($name: ident, $chrono_method: ident, $to_datetime_fn: expr,
    $primitive_in: ty,
    $primitive_out: ty) => {
        #[inline]
        pub(crate) fn $name(value: $primitive_in) -> Option<$primitive_out> {
            $to_datetime_fn(value).map(|dt| dt.$chrono_method() as $primitive_out)
        }
    };
}

macro_rules! to_boolean_temporal_unit {
    ($name: ident, $chrono_method: ident, $boolean_method: ident, $to_datetime_fn: expr, $dtype_in: ty) => {
        #[inline]
        pub(crate) fn $name(value: $dtype_in) -> Option<bool> {
            $to_datetime_fn(value).map(|dt| $boolean_method(dt.$chrono_method()))
        }
    };
}

macro_rules! to_calendar_value {
    ($name: ident, $dt: ident, $expr: expr, $to_datetime_fn: expr,
    $primitive_in: ty,
    $primitive_out: ty) => {
        #[inline]
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

/// Defines the extraction of one field of the wall time an instant stands for.
///
/// The instant is read once, by whichever conversion the column's timestamp unit asks for, and
/// every field is taken off it — so a field costs one function per field, not one per unit.
macro_rules! datetime_field {
    ($($name:ident, $dt:ident, $expr:expr, $out:ty;)*) => {
        $(
            #[cfg(feature = "dtype-datetime")]
            pub(crate) fn $name($dt: NaiveDateTime) -> $out {
                $expr
            }
        )*
    };
}

datetime_field! {
    datetime_year, dt, dt.year(), i32;
    datetime_month, dt, dt.month() as i8, i8;
    datetime_day, dt, dt.day() as i8, i8;
    datetime_hour, dt, dt.hour() as i8, i8;
    datetime_minute, dt, dt.minute() as i8, i8;
    datetime_second, dt, dt.second() as i8, i8;
    datetime_nanosecond, dt, dt.nanosecond() as i32, i32;
    datetime_weekday, dt, dt.weekday_number(), i8;
    datetime_iso_week, dt, dt.week(), i8;
    datetime_iso_year, dt, dt.iso_year(), i32;
    datetime_ordinal, dt, dt.ordinal() as i16, i16;
    datetime_is_leap_year, dt, is_leap_year(dt.year()), bool;
    datetime_days_in_month, dt, days_in_month(dt.year(), dt.month() as u8) as i8, i8;
}
