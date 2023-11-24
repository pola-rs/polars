use std::ops::Div;

use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::NativeType;
use polars_core::datatypes::ListChunked;
use polars_core::export::num::{NumCast, ToPrimitive};
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::chunked_array::sum::sum_slice;

fn sum_between_offsets<T, S>(values: &[T], offset: &[i64]) -> Vec<S>
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            sum_slice(slice)
        })
        .collect_trusted()
}

fn dispatch_sum<T, S>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    Box::new(PrimitiveArray::from_data_default(
        sum_between_offsets::<_, S>(values, offsets).into(),
        validity.cloned(),
    )) as ArrayRef
}

pub(super) fn sum_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_sum::<i8, i64>(values, offsets, arr.validity()),
                Int16 => dispatch_sum::<i16, i64>(values, offsets, arr.validity()),
                Int32 => dispatch_sum::<i32, i32>(values, offsets, arr.validity()),
                Int64 => dispatch_sum::<i64, i64>(values, offsets, arr.validity()),
                UInt8 => dispatch_sum::<u8, i64>(values, offsets, arr.validity()),
                UInt16 => dispatch_sum::<u16, i64>(values, offsets, arr.validity()),
                UInt32 => dispatch_sum::<u32, u32>(values, offsets, arr.validity()),
                UInt64 => dispatch_sum::<u64, u64>(values, offsets, arr.validity()),
                Float32 => dispatch_sum::<f32, f32>(values, offsets, arr.validity()),
                Float64 => dispatch_sum::<f64, f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}

pub(super) fn sum_with_nulls(ca: &ListChunked, inner_dtype: &DataType) -> Series {
    use DataType::*;
    // TODO: add fast path for smaller ints?
    let mut out = match inner_dtype {
        Boolean => {
            let out: IdxCa =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<IdxSize>()));
            out.into_series()
        },
        UInt32 => {
            let out: UInt32Chunked =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<u32>()));
            out.into_series()
        },
        UInt64 => {
            let out: UInt64Chunked =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<u64>()));
            out.into_series()
        },
        Int32 => {
            let out: Int32Chunked =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<i32>()));
            out.into_series()
        },
        Int64 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<i64>()));
            out.into_series()
        },
        Float32 => {
            let out: Float32Chunked =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<f32>()));
            out.into_series()
        },
        Float64 => {
            let out: Float64Chunked =
                ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum::<f64>()));
            out.into_series()
        },
        // slowest sum_as_series path
        _ => ca
            .apply_amortized(|s| s.as_ref().sum_as_series())
            .explode()
            .unwrap()
            .into_series(),
    };
    out.rename(ca.name());
    out
}

fn mean_between_offsets<T, S>(values: &[T], offset: &[i64]) -> Vec<S>
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum + Div<Output = S>,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            unsafe {
                sum_slice::<_, S>(slice) / NumCast::from(slice.len()).unwrap_unchecked_release()
            }
        })
        .collect_trusted()
}

fn dispatch_mean<T, S>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum + Div<Output = S>,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    Box::new(PrimitiveArray::from_data_default(
        mean_between_offsets::<_, S>(values, offsets).into(),
        validity.cloned(),
    )) as ArrayRef
}

pub(super) fn mean_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_mean::<i8, f64>(values, offsets, arr.validity()),
                Int16 => dispatch_mean::<i16, f64>(values, offsets, arr.validity()),
                Int32 => dispatch_mean::<i32, f64>(values, offsets, arr.validity()),
                Int64 => dispatch_mean::<i64, f64>(values, offsets, arr.validity()),
                UInt8 => dispatch_mean::<u8, f64>(values, offsets, arr.validity()),
                UInt16 => dispatch_mean::<u16, f64>(values, offsets, arr.validity()),
                UInt32 => dispatch_mean::<u32, f64>(values, offsets, arr.validity()),
                UInt64 => dispatch_mean::<u64, f64>(values, offsets, arr.validity()),
                Float32 => dispatch_mean::<f32, f32>(values, offsets, arr.validity()),
                Float64 => dispatch_mean::<f64, f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}

pub(super) fn mean_with_nulls(ca: &ListChunked) -> Series {
    return match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().mean().map(|v| v as f32)))
                .with_name(ca.name());
            out.into_series()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().mean()))
                .with_name(ca.name());
            out.into_series()
        },
    };
}
