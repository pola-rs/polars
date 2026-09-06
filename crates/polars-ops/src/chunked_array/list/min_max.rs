use std::ops::Range;

use arrow::types::NativeType;
use polars_array::bitmap::combine_validities_and;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

use crate::chunked_array::list::namespace::has_inner_nulls;

/// Reduces the elements of `values` that each of the ranges `offset` marks off to one element.
///
/// A row is sliced off `values` rather than read out of it, which is `O(1)` and keeps the row in
/// whatever representation the values are in: a values buffer that repeats a single value hands
/// every row that one value, and no run is ever written out.
/// The row `range` covers, sliced off `values` rather than read out of them, or `None` if it is
/// empty. Slicing keeps the row in whatever representation the values are in.
fn row_of<T: NativeType>(
    values: &PlPrimitiveArray<T>,
    range: Range<usize>,
) -> Option<PlPrimitiveArray<T>> {
    (!range.is_empty()).then(|| values.clone().sliced(range.start, range.len()))
}

fn min_between_offsets<T>(values: &PlPrimitiveArray<T>, offset: &[u64]) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;
            if current_offset == *end {
                return None;
            }

            let row = values
                .clone()
                .sliced(current_offset as usize, (*end - current_offset) as usize);
            row.min_ignore_nan_kernel()
        })
        .collect()
}

/// Reduces each list of `arr` to one element, reading the offsets and the values in whatever
/// representation each is in.
fn dispatch_min<T>(arr: &PlListArray) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let length = arr.len();
    let validity = arr.validity();
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap();

    // Every element covers the one range, so they all reduce to the same element: the range is
    // reduced once and repeated rather than the lists being laid end to end first.
    if let Some(range) = arr.scalar_offsets() {
        return match row_of(values, range).and_then(|row| row.min_ignore_nan_kernel()) {
            Some(value) => PlPrimitiveArray::new_scalar(value, length)
                .with_validity(validity.map(PlBitmap::from)),
            // The one range every element covers is empty, so every element reduces to nothing.
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    let offsets = arr
        .flat_offsets()
        .expect("the elements cover ranges of their own");
    let out = min_between_offsets(values, offsets);
    // Collecting leaves `out` flat, so its mask holds one bit per element like the other one.
    let new_validity = combine_validities_and(out.validity(), validity);
    out.with_validity(new_validity)
}

fn min_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    with_match_physical_numeric_polars_type!(inner_type, |$T| {
        let chunks = ca
            .downcast_iter()
            .map(dispatch_min::<<$T as PolarsNumericType>::Native>);

        ChunkedArray::<$T>::from_chunk_iter(ca.name().clone(), chunks).into_series()
    })
}

pub(super) fn list_min_function(ca: &ListChunked) -> PolarsResult<Series> {
    fn inner(ca: &ListChunked) -> PolarsResult<Series> {
        match ca.inner_dtype() {
            DataType::Boolean => {
                let out: BooleanChunked = ca
                    .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().bool().unwrap().min()));
                Ok(out.into_series())
            },
            dt if dt.to_physical().is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(dt.to_physical(), |$T| {
                    let out: ChunkedArray<$T> = ca.to_physical_repr().apply_amortized_generic(|opt_s| {
                            let s = opt_s?;
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                            ca.min()
                    });
                    // restore logical type
                    unsafe { out.into_series().from_physical_unchecked(dt) }
                })
            },
            dt => unsafe {
                // SAFETY: `min_reduce` doesn't change the dtype
                ca.try_apply_amortized_same_type(|s| {
                    let s = s.as_ref();
                    let sc = s.min_reduce()?;
                    Ok(sc.into_series(s.name().clone()))
                })?
            }
            .explode(ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            })
            .unwrap()
            .into_series()
            .cast(dt),
        }
    }

    if has_inner_nulls(ca) {
        return inner(ca);
    };

    match ca.inner_dtype() {
        dt if dt.is_primitive_numeric() => Ok(min_list_numerical(ca, dt)),
        _ => inner(ca),
    }
}

/// Reduces the elements of `values` that each of the ranges `offset` marks off to one element.
///
/// A row is sliced off `values` rather than read out of it, which is `O(1)` and keeps the row in
/// whatever representation the values are in: a values buffer that repeats a single value hands
/// every row that one value, and no run is ever written out.
fn max_between_offsets<T>(values: &PlPrimitiveArray<T>, offset: &[u64]) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;
            if current_offset == *end {
                return None;
            }

            let row = values
                .clone()
                .sliced(current_offset as usize, (*end - current_offset) as usize);
            row.max_ignore_nan_kernel()
        })
        .collect()
}

/// Reduces each list of `arr` to one element, reading the offsets and the values in whatever
/// representation each is in.
fn dispatch_max<T>(arr: &PlListArray) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let length = arr.len();
    let validity = arr.validity();
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap();

    // Every element covers the one range, so they all reduce to the same element: the range is
    // reduced once and repeated rather than the lists being laid end to end first.
    if let Some(range) = arr.scalar_offsets() {
        return match row_of(values, range).and_then(|row| row.max_ignore_nan_kernel()) {
            Some(value) => PlPrimitiveArray::new_scalar(value, length)
                .with_validity(validity.map(PlBitmap::from)),
            // The one range every element covers is empty, so every element reduces to nothing.
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    let offsets = arr
        .flat_offsets()
        .expect("the elements cover ranges of their own");
    let out = max_between_offsets(values, offsets);
    // Collecting leaves `out` flat, so its mask holds one bit per element like the other one.
    let new_validity = combine_validities_and(out.validity(), validity);
    out.with_validity(new_validity)
}

fn max_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    with_match_physical_numeric_polars_type!(inner_type, |$T| {
        let chunks = ca
            .downcast_iter()
            .map(dispatch_max::<<$T as PolarsNumericType>::Native>);

        ChunkedArray::<$T>::from_chunk_iter(ca.name().clone(), chunks).into_series()
    })
}

pub(super) fn list_max_function(ca: &ListChunked) -> PolarsResult<Series> {
    fn inner(ca: &ListChunked) -> PolarsResult<Series> {
        match ca.inner_dtype() {
            DataType::Boolean => {
                let out: BooleanChunked = ca
                    .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().bool().unwrap().max()));
                Ok(out.into_series())
            },
            dt if dt.to_physical().is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(dt.to_physical(), |$T| {
                    let out: ChunkedArray<$T> = ca.to_physical_repr().apply_amortized_generic(|opt_s| {
                            let s = opt_s?;
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                            ca.max()
                    });
                    // restore logical type
                    unsafe { out.into_series().from_physical_unchecked(dt) }
                })
            },
            dt => unsafe {
                // SAFETY: `max_reduce` doesn't change the dtype
                ca.try_apply_amortized_same_type(|s| {
                    let s = s.as_ref();
                    let sc = s.max_reduce()?;
                    Ok(sc.into_series(s.name().clone()))
                })?
            }
            .explode(ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            })
            .unwrap()
            .into_series()
            .cast(dt),
        }
    }

    if has_inner_nulls(ca) {
        return inner(ca);
    };

    match ca.inner_dtype() {
        dt if dt.is_primitive_numeric() => Ok(max_list_numerical(ca, dt)),
        _ => inner(ca),
    }
}

#[cfg(test)]
mod tests {
    use polars_array::{PlArray, PlBitmap};

    use super::*;

    fn lists(values: PlPrimitiveArray<i32>, offsets: &[u64]) -> PlListArray {
        PlListArray::from_offsets(values.into_boxed(), offsets.to_vec().into())
    }

    /// The rows of a values buffer that repeats a single value are each that one value, however
    /// the offsets carve them up: every row is sliced off in `O(1)` and reduces to the value,
    /// and a row of no elements at all reduces to a null.
    #[test]
    fn a_repeated_value_is_reduced_row_by_row() {
        let arr = lists(PlPrimitiveArray::new_scalar(7i32, 6), &[0, 3, 3, 6]);

        assert_eq!(
            dispatch_min::<i32>(&arr),
            PlPrimitiveArray::from_iter([Some(7i32), None, Some(7)]),
        );
        assert_eq!(
            dispatch_max::<i32>(&arr),
            PlPrimitiveArray::from_iter([Some(7i32), None, Some(7)]),
        );
    }

    /// Offsets holding the one range every element covers reduce that range once, and the answer
    /// repeats rather than being written out per element.
    #[test]
    fn one_shared_range_is_reduced_once() {
        let shared = PlListArray::new_scalar(
            PlPrimitiveArray::from_vec(vec![3i32, -1, 9]).into_boxed(),
            3,
        );

        let min = dispatch_min::<i32>(&shared);
        assert!(
            min.is_scalar(),
            "one range gives one answer for every element"
        );
        assert_eq!(min.iter().collect::<Vec<_>>(), [Some(-1); 3]);
        assert_eq!(
            dispatch_max::<i32>(&shared).iter().collect::<Vec<_>>(),
            [Some(9); 3]
        );

        // A shared range holding nothing leaves every element without an answer.
        let empty = PlListArray::new_scalar(PlPrimitiveArray::<i32>::new_empty().into_boxed(), 3);
        assert_eq!(dispatch_min::<i32>(&empty).null_count(), 3);
    }

    /// A row of a values buffer laid out one slot per element is reduced over the range the
    /// offsets mark off, and a null row of the outer array stays null.
    #[test]
    fn every_row_is_reduced_between_its_offsets() {
        let arr = lists(
            PlPrimitiveArray::from_vec(vec![3i32, -1, 9, 4, 8, 2]),
            &[0, 3, 6],
        );

        assert_eq!(
            dispatch_min::<i32>(&arr),
            PlPrimitiveArray::from_vec(vec![-1i32, 2]),
        );
        assert_eq!(
            dispatch_max::<i32>(&arr),
            PlPrimitiveArray::from_vec(vec![9i32, 8]),
        );

        let masked = arr.with_validity(Some(PlBitmap::from_bitmap(
            [true, false].into_iter().collect(),
        )));
        assert_eq!(
            dispatch_max::<i32>(&masked),
            PlPrimitiveArray::from_iter([Some(9i32), None]),
        );
    }
}
