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
            let left = left_column.logical();
            let right = right_column
                .categorical()
                .unwrap()
                .logical()
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

    // No nulls.
    if left_column.null_count() == 0 && right_ca.null_count() == 0 {
        // Single chunk case.
        if left_column.chunks().len() == 1 && right_column.chunks().len() == 1 {
            let left_arr = left_column.downcast_iter().next().unwrap();
            let right_arr = right_ca.downcast_iter().next().unwrap();

            match (left_arr.as_slice(), right_arr.as_slice()) {
                (Some(left_slice), Some(right_slice)) => opt_join_tuples
                    .iter()
                    .map(|(opt_left_idx, opt_right_idx)| {
                        if let Some(left_idx) = opt_left_idx {
                            *unsafe { left_slice.get_unchecked(*left_idx as usize) }
                        } else {
                            unsafe {
                                let right_idx = opt_right_idx.unwrap_unchecked();
                                *right_slice.get_unchecked(right_idx as usize)
                            }
                        }
                    })
                    .collect_ca_trusted_like(left_column)
                    .into_series(),
                _ => opt_join_tuples
                    .iter()
                    .map(|(opt_left_idx, opt_right_idx)| {
                        if let Some(left_idx) = opt_left_idx {
                            unsafe { left_arr.value_unchecked(*left_idx as usize) }
                        } else {
                            unsafe {
                                let right_idx = opt_right_idx.unwrap_unchecked();
                                right_arr.value_unchecked(right_idx as usize)
                            }
                        }
                    })
                    .collect_ca_trusted_like(left_column)
                    .into_series(),
            }
        } else {
            opt_join_tuples
                .iter()
                .map(|(opt_left_idx, opt_right_idx)| {
                    if let Some(left_idx) = opt_left_idx {
                        unsafe { left_column.value_unchecked(*left_idx as usize) }
                    } else {
                        unsafe {
                            let right_idx = opt_right_idx.unwrap_unchecked();
                            right_ca.value_unchecked(right_idx as usize)
                        }
                    }
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
            opt_join_tuples
                .iter()
                .map(|(opt_left_idx, opt_right_idx)| {
                    if let Some(left_idx) = opt_left_idx {
                        unsafe { left_arr.get_unchecked(*left_idx as usize) }
                    } else {
                        unsafe {
                            let right_idx = opt_right_idx.unwrap_unchecked();
                            right_arr.get_unchecked(right_idx as usize)
                        }
                    }
                })
                .collect_ca_trusted_like(left_column)
                .into_series()
        } else {
            opt_join_tuples
                .iter()
                .map(|(opt_left_idx, opt_right_idx)| {
                    if let Some(left_idx) = opt_left_idx {
                        unsafe { left_column.get_unchecked(*left_idx as usize) }
                    } else {
                        unsafe {
                            let right_idx = opt_right_idx.unwrap_unchecked();
                            right_ca.get_unchecked(right_idx as usize)
                        }
                    }
                })
                .collect_ca_trusted_like(left_column)
                .into_series()
        }
    }
}
