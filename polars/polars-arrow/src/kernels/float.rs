use crate::array::default_arrays::FromData;
use arrow::array::{ArrayRef, BooleanArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use num::Float;
use std::sync::Arc;

pub fn is_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_nan()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}

pub fn is_not_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| !v.is_nan()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}

pub fn is_finite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_finite()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}

pub fn is_infinite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_infinite()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}
