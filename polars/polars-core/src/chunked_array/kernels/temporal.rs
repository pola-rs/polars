use crate::chunked_array::{builder::get_bitmap, temporal::conversions_utils::*};
use crate::prelude::*;
use arrow::array::{ArrayRef, PrimitiveArray};
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
    ($name: ident, $chrono_method:ident, $to_datetime_fn: expr, $dtype_in: ty, $dtype_out:ty) => {
        pub fn $name(arr: &PrimitiveArray<$dtype_in>) -> ArrayRef {
            let vals = arr.values();
            let (_null_count, null_bit_buffer) = get_bitmap(arr);
            let av = vals
                .iter()
                .map(|&v| {
                    let dt = $to_datetime_fn(v);
                    dt.$chrono_method()
                })
                .collect::<AlignedVec<_>>();
            Arc::new(av.into_primitive_array::<$dtype_out>(null_bit_buffer))
        }
    };
}
to_temporal_unit!(
    date32_to_week,
    week,
    date32_as_datetime,
    Date32Type,
    UInt32Type
);
to_temporal_unit!(
    date32_to_weekday,
    p_weekday,
    date32_as_datetime,
    Date32Type,
    UInt32Type
);
to_temporal_unit!(
    date32_to_year,
    year,
    date32_as_datetime,
    Date32Type,
    Int32Type
);
to_temporal_unit!(
    date32_to_month,
    month,
    date32_as_datetime,
    Date32Type,
    UInt32Type
);
to_temporal_unit!(
    date32_to_day,
    day,
    date32_as_datetime,
    Date32Type,
    UInt32Type
);
to_temporal_unit!(
    date32_to_ordinal,
    ordinal,
    date32_as_datetime,
    Date32Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_week,
    week,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_weekday,
    p_weekday,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_year,
    year,
    date64_as_datetime,
    Date64Type,
    Int32Type
);
to_temporal_unit!(
    date64_to_month,
    month,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_day,
    day,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_hour,
    hour,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_minute,
    minute,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_second,
    second,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_nanosecond,
    nanosecond,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
to_temporal_unit!(
    date64_to_ordinal,
    ordinal,
    date64_as_datetime,
    Date64Type,
    UInt32Type
);
