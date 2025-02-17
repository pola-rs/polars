//! macros that define kernels for extracting
//! `week`, `weekday`, `year`, `hour` etc. from primitive arrays.
use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::compute::arity::unary;
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
}

impl PolarsIso for NaiveDateTime {
    fn week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
    fn iso_year(&self) -> i32 {
        self.iso_week().year()
    }
}

impl PolarsIso for NaiveDate {
    fn week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
    fn iso_year(&self) -> i32 {
        self.iso_week().year()
    }
}

macro_rules! to_temporal_unit {
    ($name: ident, $chrono_method: ident, $to_datetime_fn: expr,
    $primitive_in: ty,
    $primitive_out: ty,
    $dtype_out:expr) => {
        pub(crate) fn $name(arr: &PrimitiveArray<$primitive_in>) -> ArrayRef {
            Box::new(unary(
                arr,
                |value| {
                    $to_datetime_fn(value)
                        .map(|dt| dt.$chrono_method() as $primitive_out)
                        .unwrap_or(value as $primitive_out)
                },
                $dtype_out,
            )) as ArrayRef
        }
    };
}

macro_rules! to_boolean_temporal_unit {
    ($name: ident, $chrono_method: ident, $boolean_method: ident, $to_datetime_fn: expr, $dtype_in: ty) => {
        pub(crate) fn $name(arr: &PrimitiveArray<$dtype_in>) -> ArrayRef {
            let values = arr
                .values()
                .iter()
                .map(|value| {
                    $to_datetime_fn(*value)
                        .map(|dt| $boolean_method(dt.$chrono_method()))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            Box::new(BooleanArray::new(
                ArrowDataType::Boolean,
                values.into(),
                arr.validity().cloned(),
            ))
        }
    };
}

// Dates
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_iso_week,
    week,
    date32_to_datetime_opt,
    i32,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_iso_year,
    iso_year,
    date32_to_datetime_opt,
    i32,
    i32,
    ArrowDataType::Int32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_year,
    year,
    date32_to_datetime_opt,
    i32,
    i32,
    ArrowDataType::Int32
);
#[cfg(feature = "dtype-date")]
to_boolean_temporal_unit!(
    date_to_is_leap_year,
    year,
    is_leap_year,
    date32_to_datetime_opt,
    i32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_month,
    month,
    date32_to_datetime_opt,
    i32,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_day,
    day,
    date32_to_datetime_opt,
    i32,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_ordinal,
    ordinal,
    date32_to_datetime_opt,
    i32,
    i16,
    ArrowDataType::Int16
);

// Times
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_hour,
    hour,
    time64ns_to_time_opt,
    i64,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_minute,
    minute,
    time64ns_to_time_opt,
    i64,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_second,
    second,
    time64ns_to_time_opt,
    i64,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_nanosecond,
    nanosecond,
    time64ns_to_time_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_ns,
    ordinal,
    timestamp_ns_to_datetime_opt,
    i64,
    i16,
    ArrowDataType::Int16
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_ms,
    ordinal,
    timestamp_ms_to_datetime_opt,
    i64,
    i16,
    ArrowDataType::Int16
);
#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_us,
    ordinal,
    timestamp_us_to_datetime_opt,
    i64,
    i16,
    ArrowDataType::Int16
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_ns,
    iso_year,
    timestamp_ns_to_datetime_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_us,
    iso_year,
    timestamp_us_to_datetime_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_ms,
    iso_year,
    timestamp_ms_to_datetime_opt,
    i64,
    i32,
    ArrowDataType::Int32
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
