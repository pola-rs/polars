use arrow::bitmap::Bitmap;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::NativeType;
use num_traits::{NumCast, ToPrimitive};
use polars_core::prelude::*;
use polars_utils::float16::pf16;

use crate::chunked_array::sum::sum_slice;

fn dispatch_sum<T, S>(arr: &dyn PlArray, width: usize, validity: Option<&Bitmap>) -> PlArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let values = arr.as_any().downcast_ref::<PlPrimitiveArray<T>>().unwrap();
    // TODO(polars-array-scalar): the sum reads the values as a slice, so a scalar values buffer is
    // written out here rather than the one value it stands for being multiplied by the width.
    let values = values.to_flat();
    let values = values.as_slice();

    let summed: Vec<_> = (0..values.len())
        .step_by(width)
        .map(|start| {
            let slice = unsafe { values.get_unchecked(start..start + width) };
            sum_slice::<T, S>(slice)
        })
        .collect_trusted();

    // One sum per element, and `validity` holds one bit per element as well.
    PlPrimitiveArray::from_vec(summed)
        .with_validity(validity.cloned())
        .into_boxed()
}

pub(super) fn sum_array_numerical(ca: &ArrayChunked, inner_type: &DataType) -> Series {
    let width = ca.width();
    use DataType::*;

    macro_rules! dispatch {
        ($T:ty, $S:ty, $out_dtype:expr) => {{
            let chunks = ca
                .downcast_iter()
                .map(|arr| {
                    // TODO(polars-array-scalar): the values are read as a slice, so a scalar
                    // chunk is written out here rather than the one element it stands for being
                    // summed once.
                    let arr = arr.to_flat();
                    dispatch_sum::<$T, $S>(arr.values(), width, arr.validity())
                })
                .collect::<Vec<_>>();

            // SAFETY: `dispatch_sum` builds an array of `$S`, the physical type of `$out_dtype`.
            unsafe {
                Series::from_chunks_and_dtype_unchecked(ca.name().clone(), chunks, &$out_dtype)
            }
        }};
    }

    match inner_type {
        Int8 => dispatch!(i8, i64, Int64),
        Int16 => dispatch!(i16, i64, Int64),
        Int32 => dispatch!(i32, i32, Int32),
        Int64 => dispatch!(i64, i64, Int64),
        Int128 => dispatch!(i128, i128, Int128),
        UInt8 => dispatch!(u8, i64, Int64),
        UInt16 => dispatch!(u16, i64, Int64),
        UInt32 => dispatch!(u32, u32, UInt32),
        UInt64 => dispatch!(u64, u64, UInt64),
        UInt128 => dispatch!(u128, u128, UInt128),
        Float16 => dispatch!(pf16, pf16, Float16),
        Float32 => dispatch!(f32, f32, Float32),
        Float64 => dispatch!(f64, f64, Float64),
        _ => unimplemented!(),
    }
}

pub(super) fn sum_with_nulls(ca: &ArrayChunked, inner_dtype: &DataType) -> PolarsResult<Series> {
    use DataType::*;
    let mut out = {
        match inner_dtype {
            Boolean => {
                let out: IdxCa = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            UInt8 => {
                let out: Int64Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            UInt16 => {
                let out: Int64Chunked = ca
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
            #[cfg(feature = "dtype-u128")]
            UInt128 => {
                let out: UInt128Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            Int8 => {
                let out: Int64Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            Int16 => {
                let out: Int64Chunked = ca
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
            #[cfg(feature = "dtype-i128")]
            Int128 => {
                let out: Int128Chunked = ca
                    .amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().sum().ok()))
                    .collect();
                out.into_series()
            },
            #[cfg(feature = "dtype-f16")]
            Float16 => {
                let out: Float16Chunked = ca
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
    out.rename(ca.name().clone());
    Ok(out)
}
