use std::ops::Div;

use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and;
use arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use arrow::types::NativeType;
use num_traits::{NumCast, ToPrimitive};
use polars_utils::float16::pf16;

use super::*;
use crate::chunked_array::sum::sum_slice;

fn sum_between_offsets<T, S>(values: &[T], offset: &[u64]) -> Vec<S>
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    offset
        .windows(2)
        .map(|w| {
            values
                .get(w[0] as usize..w[1] as usize)
                .map(sum_slice)
                .unwrap_or(S::from(0).unwrap())
        })
        .collect()
}

fn dispatch_sum<T, S>(arr: &dyn PlArray, offsets: &[u64], validity: Option<&Bitmap>) -> PlArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let values = arr.as_any().downcast_ref::<PlPrimitiveArray<T>>().unwrap();
    // TODO(polars-array-scalar): the sum reads the values as a slice, so a scalar values buffer
    // is written out here rather than the one value it stands for being multiplied out.
    let values = values.to_flat();
    let out = PlPrimitiveArray::from_vec(sum_between_offsets::<_, S>(values.as_slice(), offsets));
    // One sum per element, and `validity` holds one bit per element as well.
    out.with_validity(validity.cloned().map(PlBitmap::from_bitmap))
        .into_boxed()
}

pub(super) fn sum_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;

    macro_rules! dispatch {
        ($T:ty, $S:ty, $out_dtype:expr) => {{
            let chunks = ca
                .downcast_iter()
                .map(|arr| {
                    // TODO(polars-array-scalar): the offsets are read as a slice, so scalar
                    // offsets are written out here rather than the range every element shares.
                    let arr = arr.to_flat();
                    dispatch_sum::<$T, $S>(arr.values(), arr.offsets().as_slice(), arr.validity())
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

pub(super) fn sum_with_nulls(ca: &ListChunked, inner_dtype: &DataType) -> PolarsResult<Series> {
    use DataType::*;
    let mut out = match inner_dtype {
        Boolean => {
            let out: IdxCa =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<IdxSize>().unwrap()));
            out.into_series()
        },
        UInt8 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<i64>().unwrap()));
            out.into_series()
        },
        UInt16 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<i64>().unwrap()));
            out.into_series()
        },
        UInt32 => {
            let out: UInt32Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<u32>().unwrap()));
            out.into_series()
        },
        UInt64 => {
            let out: UInt64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<u64>().unwrap()));
            out.into_series()
        },
        Int8 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<i64>().unwrap()));
            out.into_series()
        },
        Int16 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<i64>().unwrap()));
            out.into_series()
        },
        Int32 => {
            let out: Int32Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<i32>().unwrap()));
            out.into_series()
        },
        Int64 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<i64>().unwrap()));
            out.into_series()
        },
        #[cfg(feature = "dtype-f16")]
        Float16 => {
            let out: Float16Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<pf16>().unwrap()));
            out.into_series()
        },
        Float32 => {
            let out: Float32Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<f32>().unwrap()));
            out.into_series()
        },
        Float64 => {
            let out: Float64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| s.as_ref().sum::<f64>().unwrap()));
            out.into_series()
        },
        // slowest sum_as_series path
        dt => unsafe {
            // SAFETY: `sum_reduce` doesn't change the dtype
            ca.try_apply_amortized_same_type(|s| {
                s.as_ref()
                    .sum_reduce()
                    .map(|sc| sc.into_series(PlSmallStr::EMPTY))
            })?
        }
        .explode(ExplodeOptions {
            empty_as_null: true,
            keep_nulls: true,
        })
        .unwrap()
        .into_series()
        .cast(dt)?,
    };
    out.rename(ca.name().clone());
    Ok(out)
}

fn mean_between_offsets<T, S>(values: &[T], offset: &[u64]) -> PlPrimitiveArray<S>
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum + Div<Output = S>,
{
    offset
        .windows(2)
        .map(|w| {
            values
                .get(w[0] as usize..w[1] as usize)
                .filter(|sl| !sl.is_empty())
                .map(|sl| sum_slice::<_, S>(sl) / NumCast::from(sl.len()).unwrap())
        })
        .collect()
}

fn dispatch_mean<T, S>(arr: &dyn PlArray, offsets: &[u64], validity: Option<&Bitmap>) -> PlArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum + Div<Output = S>,
{
    let values = arr.as_any().downcast_ref::<PlPrimitiveArray<T>>().unwrap();
    // TODO(polars-array-scalar): as in `dispatch_sum`, a scalar values buffer is written out here.
    let values = values.to_flat();
    let out = mean_between_offsets::<_, S>(values.as_slice(), offsets);
    // Collecting leaves `out` flat, so its mask holds one bit per element like the other one.
    let new_validity = combine_validities_and(out.as_flat().unwrap().validity(), validity);
    out.with_validity(new_validity.map(PlBitmap::from_bitmap))
        .into_boxed()
}

pub(super) fn mean_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;

    macro_rules! dispatch {
        ($T:ty, $S:ty, $out_dtype:expr) => {{
            let chunks = ca
                .downcast_iter()
                .map(|arr| {
                    // TODO(polars-array-scalar): as in `sum_list_numerical`, scalar offsets are
                    // written out here.
                    let arr = arr.to_flat();
                    dispatch_mean::<$T, $S>(arr.values(), arr.offsets().as_slice(), arr.validity())
                })
                .collect::<Vec<_>>();

            // SAFETY: `dispatch_mean` builds an array of `$S`, the physical type of `$out_dtype`.
            unsafe {
                Series::from_chunks_and_dtype_unchecked(ca.name().clone(), chunks, &$out_dtype)
            }
        }};
    }

    match inner_type {
        Int8 => dispatch!(i8, f64, Float64),
        Int16 => dispatch!(i16, f64, Float64),
        Int32 => dispatch!(i32, f64, Float64),
        Int64 => dispatch!(i64, f64, Float64),
        Int128 => dispatch!(i128, f64, Float64),
        UInt8 => dispatch!(u8, f64, Float64),
        UInt16 => dispatch!(u16, f64, Float64),
        UInt32 => dispatch!(u32, f64, Float64),
        UInt64 => dispatch!(u64, f64, Float64),
        UInt128 => dispatch!(u128, f64, Float64),
        Float32 => dispatch!(f32, f32, Float32),
        Float64 => dispatch!(f64, f64, Float64),
        _ => unimplemented!(),
    }
}

pub(super) fn mean_with_nulls(ca: &ListChunked) -> Series {
    match ca.inner_dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let out: Float16Chunked = ca
                .apply_amortized_generic(|s| {
                    use num_traits::FromPrimitive;

                    s.and_then(|s| s.as_ref().mean().map(|v| pf16::from_f64(v).unwrap()))
                })
                .with_name(ca.name().clone());
            out.into_series()
        },
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().mean().map(|v| v as f32)))
                .with_name(ca.name().clone());
            out.into_series()
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Date => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| {
                    s.and_then(|s| s.as_ref().mean().map(|v| (v * (US_IN_DAY as f64)) as i64))
                })
                .with_name(ca.name().clone());
            out.into_datetime(TimeUnit::Microseconds, None)
                .into_series()
        },
        dt if dt.is_temporal() => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().mean().map(|v| v as i64)))
                .with_name(ca.name().clone());
            out.cast(dt).unwrap()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().mean()))
                .with_name(ca.name().clone());
            out.into_series()
        },
    }
}
