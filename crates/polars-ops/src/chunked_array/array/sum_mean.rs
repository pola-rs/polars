use arrow::legacy::utils::CustomIterTools;
use arrow::types::NativeType;
use num_traits::{NumCast, ToPrimitive};
use polars_core::prelude::*;
use polars_utils::float16::pf16;

use crate::chunked_array::sum::{sum_repeated, sum_slice};

/// The sum of each list of `arr`, one per element, in whatever representation the values are.
fn dispatch_sum<T, S>(arr: &PlFixedSizeListArray) -> PlArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let width = arr.width();
    let length = arr.len();
    let validity = arr.validity().map(PlBitmap::from);
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap();

    // Three ways for every element to sum to the same total: a list of no values at all adds up
    // to nothing, the values repeat one value, so any `width` of them add up alike, or the
    // elements all read the one list. Either way the total is worked out once over a single width
    // and repeated, rather than the values being written out one list per element first.
    //
    // A width of zero is settled here rather than below, where the step over the lists would be
    // no step at all.
    let repeated = if width == 0 {
        Some(sum_slice::<T, S>(&[]))
    } else if let Some(value) = values.scalar_value_ignore_validity() {
        Some(sum_repeated::<T, S>(value, width))
    } else if arr.values_are_scalar() {
        Some(sum_slice::<T, S>(
            values.flat_values().expect("the values are not repeated"),
        ))
    } else {
        None
    };

    if let Some(total) = repeated {
        return PlPrimitiveArray::new_scalar(total, length)
            .with_validity(validity)
            .into_boxed();
    }

    // One list per element and one slot per value: the lists are the slices they already are.
    let values = values.flat_values().expect("the values are not repeated");
    debug_assert_eq!(values.len(), length * width);

    let summed: Vec<_> = (0..values.len())
        .step_by(width)
        .map(|start| {
            // SAFETY: the values hold `width` slots per element, so a list starting at `start` is
            // in bounds of them.
            let slice = unsafe { values.get_unchecked(start..start + width) };
            sum_slice::<T, S>(slice)
        })
        .collect_trusted();

    // One sum per element, and `validity` holds one bit per element as well.
    PlPrimitiveArray::from_vec(summed)
        .with_validity(validity)
        .into_boxed()
}

pub(super) fn sum_array_numerical(ca: &ArrayChunked, inner_type: &DataType) -> Series {
    use DataType::*;

    macro_rules! dispatch {
        ($T:ty, $S:ty, $out_dtype:expr) => {{
            let chunks = ca
                .downcast_iter()
                .map(|arr| dispatch_sum::<$T, $S>(arr))
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
                let out: IdxCa =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            UInt8 => {
                let out: Int64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            UInt16 => {
                let out: Int64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            UInt32 => {
                let out: UInt32Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            UInt64 => {
                let out: UInt64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            #[cfg(feature = "dtype-u128")]
            UInt128 => {
                let out: UInt128Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            Int8 => {
                let out: Int64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            Int16 => {
                let out: Int64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            Int32 => {
                let out: Int32Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            Int64 => {
                let out: Int64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            #[cfg(feature = "dtype-i128")]
            Int128 => {
                let out: Int128Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            #[cfg(feature = "dtype-f16")]
            Float16 => {
                let out: Float16Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            Float32 => {
                let out: Float32Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
                out.into_series()
            },
            Float64 => {
                let out: Float64Chunked =
                    ca.apply_amortized_generic(|s| s.and_then(|s| s.as_ref().sum().ok()));
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
