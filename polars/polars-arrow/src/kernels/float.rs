use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use num::Float;

use crate::array::default_arrays::FromData;
use crate::prelude::*;

pub fn is_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_nan()));

    Box::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}

pub fn is_not_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let mut values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| !v.is_nan()));
    if let Some(validity) = arr.validity() {
        values = &values | &!validity
    }

    Box::new(BooleanArray::from_data_default(values, None))
}

pub fn is_finite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_finite()));

    Box::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}

pub fn is_infinite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_infinite()));

    Box::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
}
