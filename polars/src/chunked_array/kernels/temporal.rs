use crate::chunked_array::builder::{
    aligned_vec_to_primitive_array, build_with_existing_null_bitmap_and_slice, get_bitmap,
};
use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::datatypes::{Date32Type, Date64Type, DurationMillisecondType, DurationSecondType};
use std::sync::Arc;

pub fn date32_as_duration(arr: &PrimitiveArray<Date32Type>) -> ArrayRef {
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);

    let av = vals
        .into_iter()
        .map(|days| (days * 3600 * 24) as i64)
        .collect();

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
