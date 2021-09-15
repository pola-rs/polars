#[cfg(feature = "strings")]
#[cfg_attr(docsrs, doc(cfg(feature = "strings")))]
pub mod strings;
pub(crate) mod take;
pub(crate) mod take_agg;
#[cfg(feature = "temporal")]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;

use crate::datatypes::{DataType, PolarsNumericType};
use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use num::Float;
use polars_arrow::prelude::default_arrays::FromData;
use std::sync::Arc;

/// Casts a `PrimitiveArray` to a different physical type and logical type.
/// This operation is `O(N)`
/// Values that do not fit in the new physical type are converted to nulls.
pub(crate) fn cast_physical<S, T>(arr: &PrimitiveArray<S::Native>, datatype: &DataType) -> ArrayRef
where
    S: PolarsNumericType,
    T: PolarsNumericType,
    T::Native: num::NumCast,
    S::Native: num::NumCast,
{
    let array =
        arrow::compute::cast::primitive_to_primitive::<_, T::Native>(arr, &datatype.to_arrow());
    Arc::new(array)
}

pub(crate) fn is_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_nan()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().clone(),
    ))
}

pub(crate) fn is_not_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| !v.is_nan()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().clone(),
    ))
}

pub(crate) fn is_finite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_finite()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().clone(),
    ))
}

pub(crate) fn is_infinite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: NativeType + Float,
{
    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_infinite()));

    Arc::new(BooleanArray::from_data_default(
        values,
        arr.validity().clone(),
    ))
}
