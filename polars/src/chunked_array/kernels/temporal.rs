use crate::chunked_array::{
    builder::{
        aligned_vec_to_primitive_array, build_with_existing_null_bitmap_and_slice, get_bitmap,
    },
    temporal::conversions_utils::*,
};
use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::datatypes::{
    Date32Type, Date64Type, DurationMillisecondType, DurationSecondType, UInt32Type,
};
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

pub fn date64_to_hour(arr: &PrimitiveArray<Date64Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
        .map(|&v| {
            let dt = date64_as_datetime(v);
            dt.hour()
        })
        .collect();
    Arc::new(aligned_vec_to_primitive_array::<UInt32Type>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}

pub fn date64_to_day(arr: &PrimitiveArray<Date64Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
        .map(|&v| {
            let dt = date64_as_datetime(v);
            dt.day()
        })
        .collect();
    Arc::new(aligned_vec_to_primitive_array::<UInt32Type>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}

pub fn date64_to_minute(arr: &PrimitiveArray<Date64Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
        .map(|&v| {
            let dt = date64_as_datetime(v);
            dt.minute()
        })
        .collect();
    Arc::new(aligned_vec_to_primitive_array::<UInt32Type>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}

pub fn date64_to_seconds(arr: &PrimitiveArray<Date64Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
        .map(|&v| {
            let dt = date64_as_datetime(v);
            dt.second()
        })
        .collect();
    Arc::new(aligned_vec_to_primitive_array::<UInt32Type>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}

pub fn date32_to_day(arr: &PrimitiveArray<Date32Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
        .map(|&v| {
            let dt = date32_as_datetime(v);
            dt.day()
        })
        .collect();
    Arc::new(aligned_vec_to_primitive_array::<UInt32Type>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}
