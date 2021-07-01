#[cfg(feature = "strings")]
#[cfg_attr(docsrs, doc(cfg(feature = "strings")))]
pub mod strings;
pub(crate) mod take;
pub(crate) mod take_agg;
#[cfg(feature = "temporal")]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;

use polars_arrow::builder::BooleanArrayBuilder;

use crate::chunked_array::builder::get_bitmap;
use crate::datatypes::{
    ArrowDataType, Float64Type, PolarsFloatType, PolarsNumericType, PolarsPrimitiveType,
};
use crate::prelude::AlignedVec;
use crate::utils::CustomIterTools;
use arrow::array::{Array, ArrayData, ArrayRef, PrimitiveArray};
use arrow::datatypes::{
    Float32Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use num::{Float, NumCast};
use std::sync::Arc;

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
        T::DATA_TYPE,
        arr.len(),
        Some(null_count),
        null_bit_buffer.cloned(),
        data.offset(),
        buf,
        child_data,
    );
    Arc::new(PrimitiveArray::<T>::from(new_data))
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
    let vals = arr.values();
    let (_null_count, null_bit_buffer) = get_bitmap(arr);
    let av = vals
        .iter()
        .map(|v| num::cast::cast::<S::Native, T::Native>(*v).unwrap())
        .collect_trusted::<AlignedVec<T::Native>>();
    Arc::new(av.into_primitive_array::<T>(null_bit_buffer.cloned()))
}

pub(crate) fn is_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let vals = arr.values();
    let (_, null_bit_buffer) = get_bitmap(arr);
    let mut builder = BooleanArrayBuilder::new_no_nulls(vals.len());

    vals.iter().for_each(|v| {
        builder.append_value(v.is_nan());
    });
    let arr = match null_bit_buffer {
        Some(buf) => builder.finish_with_null_buffer(buf.clone()),
        None => builder.finish(),
    };
    Arc::new(arr)
}

pub(crate) fn is_not_nan<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let vals = arr.values();
    let (_, null_bit_buffer) = get_bitmap(arr);
    let mut builder = BooleanArrayBuilder::new_no_nulls(vals.len());

    vals.iter().for_each(|v| {
        builder.append_value(!v.is_nan());
    });
    let arr = match null_bit_buffer {
        Some(buf) => builder.finish_with_null_buffer(buf.clone()),
        None => builder.finish(),
    };
    Arc::new(arr)
}

pub(crate) fn is_finite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let vals = arr.values();
    let (_, null_bit_buffer) = get_bitmap(arr);
    let mut builder = BooleanArrayBuilder::new_no_nulls(vals.len());

    vals.iter().for_each(|v| {
        builder.append_value(v.is_finite());
    });
    let arr = match null_bit_buffer {
        Some(buf) => builder.finish_with_null_buffer(buf.clone()),
        None => builder.finish(),
    };
    Arc::new(arr)
}

pub(crate) fn is_infinite<T>(arr: &PrimitiveArray<T>) -> ArrayRef
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let vals = arr.values();
    let (_, null_bit_buffer) = get_bitmap(arr);
    let mut builder = BooleanArrayBuilder::new_no_nulls(vals.len());

    vals.iter().for_each(|v| {
        builder.append_value(v.is_infinite());
    });
    let arr = match null_bit_buffer {
        Some(buf) => builder.finish_with_null_buffer(buf.clone()),
        None => builder.finish(),
    };
    Arc::new(arr)
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
