use crate::chunked_array::temporal::conversions_utils::*;
use crate::prelude::*;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::compute::arity::unary;
use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use std::sync::Arc;

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
            Arc::new(unary(
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
to_temporal_unit!(
    date32_to_week,
    week,
    date32_as_datetime,
    i32,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date32_to_weekday,
    p_weekday,
    date32_as_datetime,
    i32,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date32_to_year,
    year,
    date32_as_datetime,
    i32,
    ArrowDataType::Int32
);
to_temporal_unit!(
    date32_to_month,
    month,
    date32_as_datetime,
    i32,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date32_to_day,
    day,
    date32_as_datetime,
    i32,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date32_to_ordinal,
    ordinal,
    date32_as_datetime,
    i32,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_week,
    week,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_weekday,
    p_weekday,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_year,
    year,
    date64_as_datetime,
    i64,
    ArrowDataType::Int32
);
to_temporal_unit!(
    date64_to_month,
    month,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_day,
    day,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_hour,
    hour,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_minute,
    minute,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_second,
    second,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_nanosecond,
    nanosecond,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
to_temporal_unit!(
    date64_to_ordinal,
    ordinal,
    date64_as_datetime,
    i64,
    ArrowDataType::UInt32
);
