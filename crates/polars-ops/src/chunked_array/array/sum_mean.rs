use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::NativeType;
use polars_core::export::num::{NumCast, ToPrimitive};
use polars_core::prelude::*;

use crate::chunked_array::sum::sum_slice;

fn dispatch_sum<T, S>(arr: &dyn Array, width: usize, validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();

    let summed: Vec<_> = (0..values.len())
        .step_by(width)
        .map(|start| {
            let slice = unsafe { values.get_unchecked(start..start + width) };
            sum_slice::<T, S>(slice)
        })
        .collect_trusted();

    Box::new(PrimitiveArray::from_data_default(
        summed.into(),
        validity.cloned(),
    )) as ArrayRef
}

pub(super) fn sum_array_numerical(ca: &ArrayChunked, inner_type: &DataType) -> Series {
    let width = ca.width();
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_sum::<i8, i64>(values, width, arr.validity()),
                Int16 => dispatch_sum::<i16, i64>(values, width, arr.validity()),
                Int32 => dispatch_sum::<i32, i32>(values, width, arr.validity()),
                Int64 => dispatch_sum::<i64, i64>(values, width, arr.validity()),
                UInt8 => dispatch_sum::<u8, i64>(values, width, arr.validity()),
                UInt16 => dispatch_sum::<u16, i64>(values, width, arr.validity()),
                UInt32 => dispatch_sum::<u32, u32>(values, width, arr.validity()),
                UInt64 => dispatch_sum::<u64, u64>(values, width, arr.validity()),
                Float32 => dispatch_sum::<f32, f32>(values, width, arr.validity()),
                Float64 => dispatch_sum::<f64, f64>(values, width, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}

pub(super) fn sum_with_nulls(ca: &ArrayChunked, inner_dtype: &DataType) -> PolarsResult<Series> {
    use DataType::*;
    // TODO: add fast path for smaller ints?
    let mut out = {
        match inner_dtype {
            Boolean => {
                let out: IdxCa = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            UInt32 => {
                let out: UInt32Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            UInt64 => {
                let out: UInt64Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            Int32 => {
                let out: Int32Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            Int64 => {
                let out: Int64Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            Float32 => {
                let out: Float32Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            Float64 => {
                let out: Float64Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            _ => {
                polars_bail!(ComputeError: "summing array with dtype: {} not yet supported", ca.dtype())
            },
        }
    };
    out.rename(ca.name());
    Ok(out)
}
