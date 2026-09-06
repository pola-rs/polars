use std::ops::{Div, Range};

use arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use arrow::types::NativeType;
use num_traits::{NumCast, ToPrimitive};
use polars_utils::float16::pf16;

use super::*;
use polars_array::bitmap::combine_validities_and;

use crate::chunked_array::sum::{sum_repeated, sum_slice};

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

/// The sum of each list of `arr`, one per element, reading the offsets and the values in whatever
/// representation each is in.
///
/// The values hold no null of their own — the caller has checked — so it is only the element mask
/// that carries over onto the answer.
fn dispatch_sum<T, S>(arr: &PlListArray) -> PlArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let length = arr.len();
    let validity = arr.validity().map(PlBitmap::from);
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap();

    // Every element covers the one range, so they all sum to the same total: it is worked out once
    // over that range rather than the lists being laid end to end first.
    if let Some(range) = arr.scalar_offsets() {
        return PlPrimitiveArray::new_scalar(sum_over::<T, S>(values, range), length)
            .with_validity(validity)
            .into_boxed();
    }

    let offsets = arr.flat_offsets().expect("the elements cover ranges of their own");
    let summed = match values.scalar_values() {
        // The values repeat one value, so a list adds up to that value taken as many times as the
        // list is long — again without the buffer being written out.
        Some(value) => offsets
            .windows(2)
            .map(|window| sum_repeated::<T, S>(value, (window[1] - window[0]) as usize))
            .collect(),
        None => sum_between_offsets::<_, S>(
            values.flat_values().expect("the values are not repeated"),
            offsets,
        ),
    };

    // One sum per element, and `validity` holds one bit per element as well.
    PlPrimitiveArray::from_vec(summed)
        .with_validity(validity)
        .into_boxed()
}

/// The sum of the values `range` covers, reading a repeated values buffer as the one value it is.
fn sum_over<T, S>(values: &PlPrimitiveArray<T>, range: Range<usize>) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    match values.scalar_values() {
        Some(value) => sum_repeated::<T, S>(value, range.len()),
        None => sum_slice::<T, S>(
            &values.flat_values().expect("the values are not repeated")[range],
        ),
    }
}

pub(super) fn sum_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
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

/// The average of each list of `arr`, one per element, reading the offsets and the values in
/// whatever representation each is in.
///
/// An empty list has no average, which nulls that element on top of whatever the mask already says.
fn dispatch_mean<T, S>(arr: &PlListArray) -> PlArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum + Div<Output = S>,
{
    let length = arr.len();
    let validity = arr.validity();

    let values = arr
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap();

    // Every element covers the one range, so they all average to the same thing: it is worked out
    // once over that range rather than the lists being laid end to end first.
    if let Some(range) = arr.scalar_offsets() {
        return match mean_over::<T, S>(values, range) {
            Some(mean) => PlPrimitiveArray::new_scalar(mean, length)
                .with_validity(validity.map(PlBitmap::from)),
            // The one range every element covers is empty, so every one of them is null.
            None => PlPrimitiveArray::new_full_null(length),
        }
        .into_boxed();
    }

    let offsets = arr.flat_offsets().expect("the elements cover ranges of their own");
    let out: PlPrimitiveArray<S> = match values.scalar_values() {
        // The values repeat one value, so a list averages to it — worked out through the sum the
        // flat path takes, so the two agree to the last bit.
        Some(value) => offsets
            .windows(2)
            .map(|window| {
                let count = (window[1] - window[0]) as usize;
                (count > 0).then(|| divide_by_count::<S>(sum_repeated::<T, S>(value, count), count))
            })
            .collect(),
        None => mean_between_offsets::<_, S>(
            values.flat_values().expect("the values are not repeated"),
            offsets,
        ),
    };

    // Collecting leaves `out` flat, so its mask holds one bit per element like the other one.
    let new_validity = combine_validities_and(out.validity(), validity);
    out.with_validity(new_validity).into_boxed()
}

/// The average of the values `range` covers, or `None` if it is empty.
fn mean_over<T, S>(values: &PlPrimitiveArray<T>, range: Range<usize>) -> Option<S>
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum + Div<Output = S>,
{
    let count = range.len();
    (count > 0).then(|| divide_by_count::<S>(sum_over::<T, S>(values, range), count))
}

fn divide_by_count<S: NumCast + Div<Output = S>>(total: S, count: usize) -> S {
    total / NumCast::from(count).unwrap()
}

pub(super) fn mean_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;

    macro_rules! dispatch {
        ($T:ty, $S:ty, $out_dtype:expr) => {{
            let chunks = ca
                .downcast_iter()
                .map(|arr| dispatch_mean::<$T, $S>(arr))
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

#[cfg(test)]
mod tests {
    use polars_array::PlArray;

    use super::*;

    fn flat_lists(lists: &[&[i32]]) -> PlListArray {
        let mut offsets = vec![0u64];
        let mut values = Vec::new();
        for list in lists {
            values.extend_from_slice(list);
            offsets.push(values.len() as u64);
        }

        PlListArray::from_offsets(
            PlPrimitiveArray::from_vec(values).into_boxed(),
            offsets.into(),
        )
    }

    fn sums(arr: &PlListArray) -> PlArrayRef {
        dispatch_sum::<i32, i32>(arr)
    }

    fn means(arr: &PlListArray) -> PlArrayRef {
        dispatch_mean::<i32, f64>(arr)
    }

    fn read<T: NativeType>(arr: &PlArrayRef) -> Vec<Option<T>> {
        arr.as_any()
            .downcast_ref::<PlPrimitiveArray<T>>()
            .unwrap()
            .iter()
            .collect()
    }

    /// A list array whose elements all cover one range sums and averages that range once, and the
    /// answer repeats rather than being written out per element.
    #[test]
    fn one_shared_range_is_summed_once() {
        let shared = PlListArray::new_scalar(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).into_boxed(), 4);
        let written_out = flat_lists(&[&[1, 2, 3], &[1, 2, 3], &[1, 2, 3], &[1, 2, 3]]);

        let summed = sums(&shared);
        assert!(summed.is_scalar(), "one range gives one total for every element");
        assert_eq!(read::<i32>(&summed), read::<i32>(&sums(&written_out)));

        let averaged = means(&shared);
        assert!(averaged.is_scalar());
        assert_eq!(read::<f64>(&averaged), read::<f64>(&means(&written_out)));
    }

    /// A values buffer that repeats one value is read as that value however long the lists are.
    #[test]
    fn a_repeated_value_is_never_written_out() {
        let repeated = PlListArray::from_offsets(
            PlPrimitiveArray::new_scalar(5i32, 6).into_boxed(),
            vec![0u64, 1, 1, 4, 6].into(),
        );
        let written_out = flat_lists(&[&[5], &[], &[5, 5, 5], &[5, 5]]);

        assert_eq!(read::<i32>(&sums(&repeated)), [
            Some(5),
            Some(0),
            Some(15),
            Some(10)
        ]);
        assert_eq!(read::<i32>(&sums(&repeated)), read::<i32>(&sums(&written_out)));

        // An empty list has no average, so that element is null on both paths.
        assert_eq!(read::<f64>(&means(&repeated)), [
            Some(5.0),
            None,
            Some(5.0),
            Some(5.0)
        ]);
        assert_eq!(
            read::<f64>(&means(&repeated)),
            read::<f64>(&means(&written_out))
        );
    }

    /// A shared range that is empty leaves every element without an average.
    #[test]
    fn a_shared_empty_range_averages_to_nothing() {
        let empty = PlListArray::new_scalar(PlPrimitiveArray::<i32>::new_empty().into_boxed(), 3);

        assert_eq!(read::<i32>(&sums(&empty)), [Some(0), Some(0), Some(0)]);
        assert_eq!(read::<f64>(&means(&empty)), [None, None, None]);
    }
}
