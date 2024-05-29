use num_traits::Float;

use crate::array::{ArrayRef, BooleanArray, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::legacy::array::default_arrays::FromData;
use crate::types::NativeType;

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
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| !v.is_nan()));

    Box::new(BooleanArray::from_data_default(
        values,
        arr.validity().cloned(),
    ))
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
