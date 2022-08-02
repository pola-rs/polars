//! macros that define kernels for extracting
//! `week`, `weekday`, `year`, `hour` etc. from primitive arrays.
use super::*;
use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_arrow::export::arrow::compute::arity::unary;
#[cfg(feature = "dtype-time")]
use polars_arrow::export::arrow::temporal_conversions::time64ns_to_time;
use polars_arrow::export::arrow::temporal_conversions::{
    date32_to_datetime, timestamp_ms_to_datetime, timestamp_ns_to_datetime,
    timestamp_us_to_datetime,
};

trait PolarsWeekDay {
    fn p_weekday(&self) -> u32;
    fn week(&self) -> u32;
}

impl PolarsWeekDay for NaiveDateTime {
    fn p_weekday(&self) -> u32 {
        self.weekday() as u32
    }
    fn week(&self) -> u32 {
        self.iso_week().week()
    }
}

impl PolarsWeekDay for NaiveDate {
    fn p_weekday(&self) -> u32 {
        self.weekday() as u32
    }
    fn week(&self) -> u32 {
        self.iso_week().week()
    }
}

macro_rules! to_temporal_unit {
    ($name: ident, $chrono_method:ident, $to_datetime_fn: expr, $dtype_in: ty, $dtype_out:expr) => {
        pub(crate) fn $name(arr: &PrimitiveArray<$dtype_in>) -> ArrayRef {
            Box::new(unary(
                arr,
                |value| {
                    let dt = $to_datetime_fn(value);
                    dt.$chrono_method()
                },
                $dtype_out,
            )) as ArrayRef
        }
    };
}
// Dates
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_week,
    week,
    date32_to_datetime,
    i32,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_weekday,
    p_weekday,
    date32_to_datetime,
    i32,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_year,
    year,
    date32_to_datetime,
    i32,
    ArrowDataType::Int32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_month,
    month,
    date32_to_datetime,
    i32,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_day,
    day,
    date32_to_datetime,
    i32,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_ordinal,
    ordinal,
    date32_to_datetime,
    i32,
    ArrowDataType::UInt32
);

// Times
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_hour,
    hour,
    time64ns_to_time,
    i64,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_minute,
    minute,
    time64ns_to_time,
    i64,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_second,
    second,
    time64ns_to_time,
    i64,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_nanosecond,
    nanosecond,
    time64ns_to_time,
    i64,
    ArrowDataType::UInt32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_ns,
    ordinal,
    timestamp_ns_to_datetime,
    i64,
    ArrowDataType::UInt32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_ms,
    ordinal,
    timestamp_ms_to_datetime,
    i64,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_ordinal_us,
    ordinal,
    timestamp_us_to_datetime,
    i64,
    ArrowDataType::UInt32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_weekday_ns,
    p_weekday,
    timestamp_ns_to_datetime,
    i64,
    ArrowDataType::UInt32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_weekday_ms,
    p_weekday,
    timestamp_ms_to_datetime,
    i64,
    ArrowDataType::UInt32
);
#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_weekday_us,
    p_weekday,
    timestamp_us_to_datetime,
    i64,
    ArrowDataType::UInt32
);
