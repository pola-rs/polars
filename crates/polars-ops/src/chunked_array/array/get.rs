use polars_compute::gather::sublist::fixed_size_list::{
    sub_fixed_size_list_get, sub_fixed_size_list_get_literal,
};
use polars_core::prelude::arity::{try_binary_to_series, try_unary_to_series};

use super::*;
use crate::series::convert_and_bound_idx_ca;

/// Get the value by literal index in the array.
/// So index `0` would return the first item of every sub-array
/// and index `-1` would return the last item of every sub-array
/// if an index is out of bounds, it will return a `None`.
pub fn array_get(
    ca: &ArrayChunked,
    index: &Int64Chunked,
    null_on_oob: bool,
) -> PolarsResult<Series> {
    polars_ensure!(ca.width() < IdxSize::MAX as usize, ComputeError: "`arr.get` not supported for such wide arrays");

    // Base case. No overflow.
    if ca.width() * ca.len() < IdxSize::MAX as usize {
        return array_get_impl(ca, index, null_on_oob);
    }

    // If the array width * length would overflow. Do it part-by-part.
    assert!(ca.len() != 1 || index.len() != 1);
    let rows_per_slice = IdxSize::MAX as usize / ca.width();

    let mut ca = ca.clone();
    let mut index = index.clone();
    let current_ca;
    let current_index;
    if ca.len() == 1 {
        current_ca = ca.clone();
    } else {
        (current_ca, ca) = ca.split_at(rows_per_slice as i64);
    }
    if index.len() == 1 {
        current_index = index.clone();
    } else {
        (current_index, index) = index.split_at(rows_per_slice as i64);
    }
    let mut s = array_get_impl(&current_ca, &current_index, null_on_oob)?;

    while !ca.is_empty() && !index.is_empty() {
        let current_ca;
        let current_index;
        if ca.len() == 1 {
            current_ca = ca.clone();
        } else {
            (current_ca, ca) = ca.split_at(rows_per_slice as i64);
        }
        if index.len() == 1 {
            current_index = index.clone();
        } else {
            (current_index, index) = index.split_at(rows_per_slice as i64);
        }
        s.append_owned(array_get_impl(&current_ca, &current_index, null_on_oob)?)?;
    }

    Ok(s)
}

fn array_get_impl(
    ca: &ArrayChunked,
    index: &Int64Chunked,
    null_on_oob: bool,
) -> PolarsResult<Series> {
    match index.len() {
        1 => {
            if let Some(index) = index.get(0) {
                let out = try_unary_to_series(ca, |arr| {
                    sub_fixed_size_list_get_literal(arr, index, null_on_oob)
                })?;
                unsafe { out.from_physical_unchecked(ca.inner_dtype()) }
            } else {
                Ok(Series::full_null(
                    ca.name().clone(),
                    ca.len(),
                    ca.inner_dtype(),
                ))
            }
        },

        len if len == ca.len() => {
            let out = try_binary_to_series(ca, index, |arr, idx_arr| {
                sub_fixed_size_list_get(arr, idx_arr, null_on_oob)
            })?;
            unsafe { out.from_physical_unchecked(ca.inner_dtype()) }
        },

        _len if ca.len() == 1 => {
            if let Some(arr) = ca.get(0) {
                let idx = convert_and_bound_idx_ca(index, arr.len(), null_on_oob)?;
                let s = Series::try_from((ca.name().clone(), vec![arr])).unwrap();
                unsafe {
                    s.take_unchecked(&idx)
                        .from_physical_unchecked(ca.inner_dtype())
                }
            } else {
                Ok(Series::full_null(
                    ca.name().clone(),
                    ca.len(),
                    ca.inner_dtype(),
                ))
            }
        },

        len => polars_bail!(
            ComputeError:
            "`arr.get` expression got an index array of length {} while the array has {} elements",
            len, ca.len()
        ),
    }
}
