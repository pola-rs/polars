use crate::chunked_array::{
    builder::{
        aligned_vec_to_primitive_array, build_with_existing_null_bitmap_and_slice, get_bitmap,
    },
    temporal::conversions_utils::*,
};
use crate::prelude::*;
use arrow::array::{Array, ArrayRef, PrimitiveArray};
use chrono::{Datelike, Timelike};
use std::sync::Arc;

pub fn date32_as_duration(arr: &PrimitiveArray<Date32Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);

    let av = vals.iter().map(|days| (days * 3600 * 24) as i64).collect();

    Arc::new(aligned_vec_to_primitive_array::<DurationSecondType>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}

pub fn date64_as_duration(arr: &PrimitiveArray<Date64Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    Arc::new(build_with_existing_null_bitmap_and_slice::<
        DurationMillisecondType,
    >(null_bit_buffer, null_count, vals))
}

macro_rules! to_temporal_unit {
    ($name: ident, $chrono_method:ident, $to_datetime_fn: expr, $dtype_in: ty, $dtype_out:ty) => {
        pub fn $name(arr: &PrimitiveArray<$dtype_in>) -> ArrayRef {
            let vals = arr.value_slice(arr.offset(), arr.len());
            let (null_count, null_bit_buffer) = get_bitmap(arr);
            let av = vals
                .iter()
                .map(|&v| {
                    let dt = $to_datetime_fn(v);
                    dt.$chrono_method()
                })
                .collect();
            Arc::new(aligned_vec_to_primitive_array::<$dtype_out>(
                av,
                null_bit_buffer,
                Some(null_count),
            ))
        }
    };
}

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
