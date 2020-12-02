pub mod set;
pub(crate) mod take;
#[cfg(feature = "temporal")]
#[doc(cfg(feature = "temporal"))]
pub mod temporal;
pub(crate) mod utils;
pub mod zip_with;

use crate::chunked_array::builder::{aligned_vec_to_primitive_array, get_bitmap};
use crate::datatypes::{ArrowDataType, Float64Type, PolarsNumericType, PolarsPrimitiveType};
use arrow::array::{Array, ArrayData, ArrayRef, PrimitiveArray};
use arrow::datatypes::{
    Float32Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use num::NumCast;
use std::sync::Arc;
pub use zip_with::*;

pub(crate) unsafe fn transmute_array<S, T>(arr: &PrimitiveArray<S>) -> ArrayRef
where
    S: PolarsPrimitiveType,
    T: PolarsPrimitiveType,
{
    let data = arr.data();
    let buf = data.buffers().to_vec();
    let child_data = data.child_data().to_vec();
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let new_data = ArrayData::new(
        T::get_data_type(),
        arr.len(),
        Some(null_count),
        null_bit_buffer,
        data.offset(),
        buf,
        child_data,
    );
    Arc::new(PrimitiveArray::<T>::from(Arc::new(new_data)))
}

pub(crate) unsafe fn transmute_array_from_dtype<S>(
    arr: &PrimitiveArray<S>,
    dtype: ArrowDataType,
) -> ArrayRef
where
    S: PolarsPrimitiveType,
{
    use ArrowDataType::*;
    match dtype {
        UInt64 => transmute_array::<_, UInt64Type>(arr),
        UInt32 => transmute_array::<_, UInt32Type>(arr),
        UInt16 => transmute_array::<_, UInt16Type>(arr),
        UInt8 => transmute_array::<_, UInt8Type>(arr),
        Int64 => transmute_array::<_, Int64Type>(arr),
        Int32 => transmute_array::<_, Int32Type>(arr),
        Int16 => transmute_array::<_, Int16Type>(arr),
        Int8 => transmute_array::<_, Int8Type>(arr),
        _ => todo!(),
    }
}

/// This function will panic if the conversion overflows. Don't use it to cast to a smaller size.
pub(crate) fn cast_numeric<S, T>(arr: &PrimitiveArray<S>) -> ArrayRef
where
    S: PolarsNumericType,
    T: PolarsNumericType,
    T::Native: num::NumCast,
    S::Native: num::NumCast,
{
    let vals = arr.value_slice(arr.offset(), arr.len());
    let (null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
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
    S: PolarsNumericType,
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
        Float32 => cast_numeric::<_, Float32Type>(arr),
        Float64 => cast_numeric::<_, Float64Type>(arr),
        _ => todo!(),
    }
}
