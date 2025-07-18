use arrow::array::Array;
use polars_compute::gather::sublist::fixed_size_list::{
    sub_fixed_size_list_get, sub_fixed_size_list_get_literal,
};
use polars_core::utils::align_chunks_binary;

use super::*;

fn array_get_literal(ca: &ArrayChunked, idx: i64, null_on_oob: bool) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| sub_fixed_size_list_get_literal(arr, idx, null_on_oob))
        .collect::<PolarsResult<Vec<_>>>()?;
    Series::try_from((ca.name().clone(), chunks))
        .unwrap()
        .cast(ca.inner_dtype())
}

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
            let index = index.get(0);
            if let Some(index) = index {
                array_get_literal(ca, index, null_on_oob)
            } else {
                Ok(Series::full_null(
                    ca.name().clone(),
                    ca.len(),
                    ca.inner_dtype(),
                ))
            }
        },
        len if len == ca.len() => {
            let out = binary_to_series_arr_get(ca, index, null_on_oob, |arr, idx, nob| {
                sub_fixed_size_list_get(arr, idx, nob)
            });
            out?.cast(ca.inner_dtype())
        },
        len => polars_bail!(
            ComputeError:
            "`arr.get` expression got an index array of length {} while the array has {} elements",
            len, ca.len()
        ),
    }
}

pub fn binary_to_series_arr_get<T, U, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    null_on_oob: bool,
    mut op: F,
) -> PolarsResult<Series>
where
    T: PolarsDataType,
    U: PolarsDataType,
    F: FnMut(&T::Array, &U::Array, bool) -> PolarsResult<Box<dyn Array>>,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr, null_on_oob))
        .collect::<PolarsResult<Vec<_>>>()?;
    Series::try_from((lhs.name().clone(), chunks))
}
