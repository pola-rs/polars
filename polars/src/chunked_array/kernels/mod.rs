pub mod set;
pub mod temporal;
pub(crate) mod utils;
pub mod zip_with;

use crate::chunked_array::builder::{aligned_vec_to_primitive_array, get_bitmap};
use crate::datatypes::ArrowDataType;
use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::datatypes::{
    ArrowNumericType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use num::NumCast;
use std::sync::Arc;
pub use zip_with::*;

/// This function will panic if the conversion overflows. Don't use it to cast to a smaller size.
pub(crate) fn cast_numeric<S, T>(arr: &PrimitiveArray<S>) -> ArrayRef
where
    S: ArrowNumericType,
    T: ArrowNumericType,
    T::Native: num::NumCast,
    S::Native: num::NumCast,
{
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .into_iter()
        .map(|v| num::cast::cast::<S::Native, T::Native>(*v).unwrap())
        .collect();
    Arc::new(aligned_vec_to_primitive_array::<T>(
        av,
        null_bit_buffer,
        Some(null_count),
    ))
}

pub(crate) fn cast_numeric_from_dtype<S>(arr: &PrimitiveArray<S>, dtype: ArrowDataType) -> ArrayRef
where
    S: ArrowNumericType,
    S::Native: NumCast,
{
    use ArrowDataType::*;
    match dtype {
        UInt64 => cast_numeric::<_, UInt64Type>(arr),
        UInt32 => cast_numeric::<_, UInt32Type>(arr),
        UInt16 => cast_numeric::<_, UInt16Type>(arr),
        UInt8 => cast_numeric::<_, UInt8Type>(arr),
        Int64 => cast_numeric::<_, Int64Type>(arr),
        Int32 => cast_numeric::<_, Int32Type>(arr),
        Int16 => cast_numeric::<_, Int16Type>(arr),
        Int8 => cast_numeric::<_, Int8Type>(arr),
        _ => todo!(),
    }
}
