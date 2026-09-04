use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and;
use arrow::types::NativeType;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

use crate::chunked_array::list::namespace::has_inner_nulls;

/// Reduces the elements of `values` that each of the ranges `offset` marks off to one element.
///
/// A row is sliced off `values` rather than read out of it, which is `O(1)` and keeps the row in
/// whatever representation the values are in: a values buffer that repeats a single value hands
/// every row that one value, and no run is ever written out.
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

fn dispatch_min<T>(
    arr: &dyn PlArray,
    offsets: &[u64],
    validity: Option<&Bitmap>,
) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let values = arr.as_any().downcast_ref::<PlPrimitiveArray<T>>().unwrap();
    let out = min_between_offsets(values, offsets);
    // Collecting leaves `out` flat, so its mask holds one bit per element like the other one.
    let new_validity = combine_validities_and(out.as_flat().unwrap().validity(), validity);
    out.with_validity(new_validity.map(PlBitmap::from_bitmap))
}

fn min_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    with_match_physical_numeric_polars_type!(inner_type, |$T| {
        let chunks = ca.downcast_iter().map(|arr| {
            // TODO(polars-array-scalar): the offsets are read as a slice, so scalar offsets are
            // written out here rather than the single range every element shares being used.
            let arr = arr.to_flat();
            dispatch_min::<<$T as PolarsNumericType>::Native>(
                arr.values(),
                arr.offsets().as_slice(),
                arr.validity(),
            )
        });

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

fn dispatch_max<T>(
    arr: &dyn PlArray,
    offsets: &[u64],
    validity: Option<&Bitmap>,
) -> PlPrimitiveArray<T>
where
    T: NativeType,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let values = arr.as_any().downcast_ref::<PlPrimitiveArray<T>>().unwrap();
    let out = max_between_offsets(values, offsets);
    // Collecting leaves `out` flat, so its mask holds one bit per element like the other one.
    let new_validity = combine_validities_and(out.as_flat().unwrap().validity(), validity);
    out.with_validity(new_validity.map(PlBitmap::from_bitmap))
}

fn max_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    with_match_physical_numeric_polars_type!(inner_type, |$T| {
        let chunks = ca.downcast_iter().map(|arr| {
            // TODO(polars-array-scalar): the offsets are read as a slice, so scalar offsets are
            // written out here rather than the single range every element shares being used.
            let arr = arr.to_flat();
            dispatch_max::<<$T as PolarsNumericType>::Native>(
                arr.values(),
                arr.offsets().as_slice(),
                arr.validity(),
            )
        });

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
    use super::*;

    /// The rows of a values buffer that repeats a single value are each that one value, however
    /// the offsets carve them up: every row is sliced off in `O(1)` and reduces to the value,
    /// and a row of no elements at all reduces to a null.
    #[test]
    fn a_repeated_value_is_reduced_row_by_row() {
        let values = PlPrimitiveArray::new_scalar(7i32, 6);
        let offsets = [0u64, 3, 3, 6];

        assert_eq!(
            dispatch_min::<i32>(&values, &offsets, None),
            PlPrimitiveArray::from_iter([Some(7i32), None, Some(7)]),
        );
        assert_eq!(
            dispatch_max::<i32>(&values, &offsets, None),
            PlPrimitiveArray::from_iter([Some(7i32), None, Some(7)]),
        );
    }

    /// A row of a values buffer laid out one slot per element is reduced over the range the
    /// offsets mark off, and a null row of the outer array stays null.
    #[test]
    fn every_row_is_reduced_between_its_offsets() {
        let values = PlPrimitiveArray::from_vec(vec![3i32, -1, 9, 4, 8, 2]);
        let offsets = [0u64, 3, 6];
        let validity: Bitmap = [true, false].into_iter().collect();

        assert_eq!(
            dispatch_min::<i32>(&values, &offsets, None),
            PlPrimitiveArray::from_vec(vec![-1i32, 2]),
        );
        assert_eq!(
            dispatch_max::<i32>(&values, &offsets, None),
            PlPrimitiveArray::from_vec(vec![9i32, 8]),
        );
        assert_eq!(
            dispatch_max::<i32>(&values, &offsets, Some(&validity)),
            PlPrimitiveArray::from_iter([Some(9i32), None]),
        );
    }
}
