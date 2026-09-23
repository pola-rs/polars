use std::ops::Div;

use num_traits::{NumCast, ToPrimitive};
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use polars_arrow::types::NativeType;
use polars_compute::mean::{IntMeanRounding, MeanAcc, MeanSum};
use polars_utils::float16::pf16;

use super::*;
use crate::chunked_array::sum::sum_slice;

fn sum_between_offsets<T, S>(values: &[T], offset: &[i64]) -> Vec<S>
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
                Int128 => dispatch_sum::<i128, i128>(values, offsets, arr.validity()),
                UInt8 => dispatch_sum::<u8, i64>(values, offsets, arr.validity()),
                UInt16 => dispatch_sum::<u16, i64>(values, offsets, arr.validity()),
                UInt32 => dispatch_sum::<u32, u32>(values, offsets, arr.validity()),
                UInt64 => dispatch_sum::<u64, u64>(values, offsets, arr.validity()),
                UInt128 => dispatch_sum::<u128, u128>(values, offsets, arr.validity()),
                Float16 => dispatch_sum::<pf16, pf16>(values, offsets, arr.validity()),
                Float32 => dispatch_sum::<f32, f32>(values, offsets, arr.validity()),
                Float64 => dispatch_sum::<f64, f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name().clone(), chunks)).unwrap()
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

fn mean_between_offsets<T, S>(values: &[T], offset: &[i64]) -> PrimitiveArray<S>
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

fn dispatch_mean<T, S>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum + Div<Output = S>,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    let out = mean_between_offsets::<_, S>(values, offsets);
    let new_validity = combine_validities_and(out.validity(), validity);
    out.with_validity(new_validity).to_boxed()
}

fn dispatch_mean_int<T>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: MeanSum,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    let out: PrimitiveArray<f64> = offsets
        .windows(2)
        .map(|w| {
            values
                .get(w[0] as usize..w[1] as usize)
                .filter(|sl| !sl.is_empty())
                .map(|sl| T::sum_slice(sl).into_f64() / sl.len() as f64)
        })
        .collect();
    let new_validity = combine_validities_and(out.validity(), validity);
    out.with_validity(new_validity).to_boxed()
}

pub(super) fn mean_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_mean_int::<i8>(values, offsets, arr.validity()),
                Int16 => dispatch_mean_int::<i16>(values, offsets, arr.validity()),
                Int32 => dispatch_mean_int::<i32>(values, offsets, arr.validity()),
                Int64 => dispatch_mean_int::<i64>(values, offsets, arr.validity()),
                Int128 => dispatch_mean_int::<i128>(values, offsets, arr.validity()),
                UInt8 => dispatch_mean_int::<u8>(values, offsets, arr.validity()),
                UInt16 => dispatch_mean_int::<u16>(values, offsets, arr.validity()),
                UInt32 => dispatch_mean_int::<u32>(values, offsets, arr.validity()),
                UInt64 => dispatch_mean_int::<u64>(values, offsets, arr.validity()),
                UInt128 => dispatch_mean_int::<u128>(values, offsets, arr.validity()),
                Float32 => dispatch_mean::<f32, f32>(values, offsets, arr.validity()),
                Float64 => dispatch_mean::<f64, f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name().clone(), chunks)).unwrap()
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
                .apply_amortized_generic(|s| s.and_then(|s| temporal_mean_physical(s.as_ref())))
                .with_name(ca.name().clone());
            out.into_datetime(TimeUnit::Microseconds, None)
                .into_series()
        },
        dt if dt.is_temporal() => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| temporal_mean_physical(s.as_ref())))
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

/// Exact physical mean of a temporal series: Datetime(us) for Date, the input unit otherwise.
pub(crate) fn temporal_mean_physical(s: &Series) -> Option<i64> {
    let rounding = if s.dtype().is_duration() {
        IntMeanRounding::Trunc
    } else {
        IntMeanRounding::Floor
    };
    let phys = s.to_physical_repr();
    match s.dtype() {
        DataType::Date => phys.i32().unwrap().int_mean(US_IN_DAY, rounding),
        _ => phys.i64().unwrap().int_mean(1, rounding),
    }
}
