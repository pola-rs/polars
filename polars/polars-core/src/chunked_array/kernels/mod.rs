#[cfg(feature = "strings")]
#[cfg_attr(docsrs, doc(cfg(feature = "strings")))]
pub mod strings;
pub(crate) mod take;
pub(crate) mod take_agg;
#[cfg(feature = "temporal")]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;

use crate::datatypes::PolarsFloatType;
use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use num::Float;
use std::sync::Arc;

pub(crate) fn is_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let validity = arr.validity();

    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_nan()));

    Arc::new(BooleanArray::from_data(values, arr.validity().clone()))
}

pub(crate) fn is_not_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let validity = arr.validity();

    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| !v.is_nan()));

    Arc::new(BooleanArray::from_data(values, arr.validity().clone()))
}

pub(crate) fn is_finite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let validity = arr.validity();

    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_finite()));

    Arc::new(BooleanArray::from_data(values, arr.validity().clone()))
}

pub(crate) fn is_infinite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let validity = arr.validity();

    let values = Bitmap::from_trusted_len_iter(arr.values().iter().map(|v| v.is_infinite()));

    Arc::new(BooleanArray::from_data(values, arr.validity().clone()))
}
