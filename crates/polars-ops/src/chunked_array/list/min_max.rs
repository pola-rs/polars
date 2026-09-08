use std::ops::Range;

use arrow::types::NativeType;
use polars_array::bitmap::combine_validities_and;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

use crate::chunked_array::list::namespace::has_inner_nulls;

/// Reduces the elements of `values` that each of the ranges `offset` marks off to one element.
fn row_of<T: NativeType>(
    values: &PlPrimitiveArray<T>,
    range: Range<usize>,
) -> Option<PlPrimitiveArray<T>> {
    (!range.is_empty()).then(|| values.sliced(range.start, range.len()))
}

/// The range of the values buffer every element `offset` holds the ends of covers.
fn rows_of(offset: &[u64]) -> impl Iterator<Item = Range<usize>> + '_ {
    offset
        .windows(2)
        .map(|window| window[0] as usize..window[1] as usize)
}

/// The values laid out one slot per element, if that is what they are and none of them is null.
///
/// Reading the rows out of a slice keeps a reduction to the bytes it reduces. Cutting an array of
/// its own out of the values instead bumps the refcount of every buffer behind them — an atomic
/// read-modify-write on the way in and another on the way out — once for every row of the column.
fn flat_values_of<T: NativeType>(values: &PlPrimitiveArray<T>) -> Option<&[T]> {
    (values.null_count() == 0)
        .then(|| values.flat_values())
        .flatten()
        .map(|values| values.as_slice())
}

fn min_between_offsets<T>(values: &PlPrimitiveArray<T>, offset: &[u64]) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    if let Some(slice) = flat_values_of(values) {
        return rows_of(offset)
            // SAFETY: the offsets of a list array cover its values, and an empty row reduces to
            // nothing, which is what the kernel answers of an empty slice.
            .map(|range| unsafe { slice.get_unchecked(range) }.min_ignore_nan_kernel())
            .collect();
    }

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

/// Reduces each list of `arr` to one element, in whatever representation each part is in.
fn dispatch_min<T>(arr: &PlListArray) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
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
fn max_between_offsets<T>(values: &PlPrimitiveArray<T>, offset: &[u64]) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    if let Some(slice) = flat_values_of(values) {
        return rows_of(offset)
            // SAFETY: the offsets of a list array cover its values, and an empty row reduces to
            // nothing, which is what the kernel answers of an empty slice.
            .map(|range| unsafe { slice.get_unchecked(range) }.max_ignore_nan_kernel())
            .collect();
    }

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

/// Reduces each list of `arr` to one element, in whatever representation each part is in.
fn dispatch_max<T>(arr: &PlListArray) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
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
    use polars_buffer::Buffer;
    use polars_core::prelude::*;

    use super::{list_max_function, list_min_function};

    /// A list column over `values`, cut into the rows `offsets` marks off.
    fn lists(values: PlPrimitiveArray<i32>, offsets: &[u64]) -> ListChunked {
        let length = offsets.len() - 1;
        let arr = PlListArray::new(
            values.into_boxed(),
            Buffer::from(offsets.to_vec()),
            length,
            None,
        );

        unsafe {
            ListChunked::from_chunks_and_dtype(
                "l".into(),
                vec![arr.into_boxed()],
                DataType::List(Box::new(DataType::Int32)),
            )
        }
    }

    /// The rows are reduced out of the values slice where the values are laid out one slot per
    /// element, and out of an array cut per row where they are not: the two answer the same.
    #[test]
    fn rows_reduce_the_same_however_the_values_are_laid_out() {
        let offsets = [0u64, 3, 3, 5, 6];
        let expected_min = [Some(-1), None, Some(2), Some(9)];
        let expected_max = [Some(7), None, Some(4), Some(9)];

        let flat = PlPrimitiveArray::from_vec(vec![5i32, -1, 7, 4, 2, 9]);
        assert!(flat.values_are_flat());

        let ca = lists(flat, &offsets);
        assert_eq!(
            list_min_function(&ca)
                .unwrap()
                .i32()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            expected_min,
        );
        assert_eq!(
            list_max_function(&ca)
                .unwrap()
                .i32()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            expected_max,
        );

        // Values that repeat one slot are not laid out per element, so the rows are cut out of an
        // array instead — and every non-empty row holds nothing but that one value.
        let scalar = PlPrimitiveArray::new_scalar(3i32, 6);
        assert!(!scalar.values_are_flat());

        let ca = lists(scalar, &offsets);
        let repeated = [Some(3), None, Some(3), Some(3)];
        assert_eq!(
            list_min_function(&ca)
                .unwrap()
                .i32()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            repeated,
        );
        assert_eq!(
            list_max_function(&ca)
                .unwrap()
                .i32()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            repeated,
        );
    }
}
