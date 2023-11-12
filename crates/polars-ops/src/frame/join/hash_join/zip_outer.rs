use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub(crate) unsafe fn zip_outer_join_column(
    left_column: &Series,
    right_column: &Series,
    opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
) -> Series {
    match left_column.dtype() {
        DataType::Null => {
            Series::full_null(left_column.name(), opt_join_tuples.len(), &DataType::Null)
        },
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_) => {
            let left_column = left_column.categorical().unwrap();
            let new_rev_map = left_column
                ._merge_categorical_map(right_column.categorical().unwrap())
                .unwrap();
            let left = left_column.physical();
            let right = right_column
                .categorical()
                .unwrap()
                .physical()
                .clone()
                .into_series();

            let cats = zip_outer_join_column_ca(left, &right, opt_join_tuples);
            let cats = cats.u32().unwrap().clone();

            unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(cats, new_rev_map).into_series()
            }
        },
        DataType::Utf8 => {
            let left_column = left_column.cast(&DataType::Binary).unwrap();
            let right_column = right_column.cast(&DataType::Binary).unwrap();
            let out = zip_outer_join_column_ca(
                left_column.binary().unwrap(),
                &right_column,
                opt_join_tuples,
            );
            out.cast_unchecked(&DataType::Utf8).unwrap()
        },
        DataType::Binary => {
            zip_outer_join_column_ca(left_column.binary().unwrap(), right_column, opt_join_tuples)
        },
        DataType::Boolean => {
            zip_outer_join_column_ca(left_column.bool().unwrap(), right_column, opt_join_tuples)
        },
        logical_type => {
            let lhs_phys = left_column.to_physical_repr();
            let rhs_phys = right_column.to_physical_repr();

            let out = with_match_physical_numeric_polars_type!(lhs_phys.dtype(), |$T| {
                let lhs: &ChunkedArray<$T> = lhs_phys.as_ref().as_ref().as_ref();

                zip_outer_join_column_ca(lhs, &rhs_phys, opt_join_tuples)
            });
            out.cast_unchecked(logical_type).unwrap()
        },
    }
}

fn get_value<T, A, F: Fn(A, usize) -> T>(
    opt_left_idx: Option<IdxSize>,
    opt_right_idx: Option<IdxSize>,
    left_arr: A,
    right_arr: A,
    getter: F,
) -> T {
    if let Some(left_idx) = opt_left_idx {
        getter(left_arr, left_idx as usize)
    } else {
        unsafe {
            let right_idx = opt_right_idx.unwrap_unchecked();
            getter(right_arr, right_idx as usize)
        }
    }
}

// TODO! improve this once we have a proper scatter.
// Two scatters should do it. Can also improve the `opt_join_tuples` format then.
unsafe fn zip_outer_join_column_ca<'a, T>(
    left_column: &'a ChunkedArray<T>,
    right_column: &Series,
    opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
) -> Series
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
    T::Physical<'a>: Copy,
{
    let right_ca = left_column
        .unpack_series_matching_type(right_column)
        .unwrap();

    let tuples_iter = opt_join_tuples.iter();

    // No nulls.
    if left_column.null_count() == 0 && right_ca.null_count() == 0 {
        // Single chunk case.
        if left_column.chunks().len() == 1 && right_column.chunks().len() == 1 {
            let left_arr = left_column.downcast_iter().next().unwrap();
            let right_arr = right_ca.downcast_iter().next().unwrap();

            match (left_arr.as_slice(), right_arr.as_slice()) {
                (Some(left_slice), Some(right_slice)) => tuples_iter
                    .map(|(opt_left_idx, opt_right_idx)| {
                        get_value(
                            *opt_left_idx,
                            *opt_right_idx,
                            left_slice,
                            right_slice,
                            |slice, idx| *slice.get_unchecked(idx),
                        )
                    })
                    .collect_ca_trusted_like(left_column)
                    .into_series(),
                _ => tuples_iter
                    .map(|(opt_left_idx, opt_right_idx)| {
                        get_value(
                            *opt_left_idx,
                            *opt_right_idx,
                            left_arr,
                            right_arr,
                            |slice, idx| slice.value_unchecked(idx),
                        )
                    })
                    .collect_ca_trusted_like(left_column)
                    .into_series(),
            }
        } else {
            tuples_iter
                .map(|(opt_left_idx, opt_right_idx)| {
                    get_value(
                        *opt_left_idx,
                        *opt_right_idx,
                        left_column,
                        right_ca,
                        |slice, idx| slice.value_unchecked(idx),
                    )
                })
                .collect_ca_trusted_like(left_column)
                .into_series()
        }

    // Nulls.
    } else {
        // Single chunk case.
        if left_column.chunks().len() == 1 && right_column.chunks().len() == 1 {
            let left_arr = left_column.downcast_iter().next().unwrap();
            let right_arr = right_ca.downcast_iter().next().unwrap();
            tuples_iter
                .map(|(opt_left_idx, opt_right_idx)| {
                    get_value(
                        *opt_left_idx,
                        *opt_right_idx,
                        left_arr,
                        right_arr,
                        |slice, idx| slice.get_unchecked(idx),
                    )
                })
                .collect_ca_trusted_like(left_column)
                .into_series()
        } else {
            tuples_iter
                .map(|(opt_left_idx, opt_right_idx)| {
                    get_value(
                        *opt_left_idx,
                        *opt_right_idx,
                        left_column,
                        right_ca,
                        |slice, idx| slice.get_unchecked(idx),
                    )
                })
                .collect_ca_trusted_like(left_column)
                .into_series()
        }
    }
}
