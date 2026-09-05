use polars_array::PlArray;
use polars_compute::gather::sublist::fixed_size_list::{
    sub_fixed_size_list_get, sub_fixed_size_list_get_literal,
};
use polars_core::utils::align_chunks_binary;

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
                let chunks = ca
                    .downcast_iter()
                    .map(|arr| sub_fixed_size_list_get_literal(arr, index, null_on_oob))
                    .collect::<PolarsResult<Vec<_>>>()?;
                let out = values_series(ca, chunks);
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
            let (aligned, index) = align_chunks_binary(ca, index);
            let chunks = aligned
                .downcast_iter()
                .zip(index.downcast_iter())
                .map(|(arr, idx_arr)| sub_fixed_size_list_get(arr, idx_arr, null_on_oob))
                .collect::<PolarsResult<Vec<_>>>()?;
            let out = values_series(ca, chunks);
            unsafe { out.from_physical_unchecked(ca.inner_dtype()) }
        },

        _len if ca.len() == 1 => {
            if let Some(arr) = ca.get(0) {
                let idx = convert_and_bound_idx_ca(index, arr.len(), null_on_oob)?;
                let s = values_series(ca, vec![arr]);
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

/// The values `chunks` hold, as a series of the physical inner type of `ca`.
///
/// The values of an element carry no logical type of their own; the physical inner type is what
/// `from_physical_unchecked` turns back into the logical one.
fn values_series(ca: &ArrayChunked, chunks: Vec<Box<dyn PlArray>>) -> Series {
    unsafe {
        Series::from_chunks_and_dtype_unchecked(
            ca.name().clone(),
            chunks,
            &ca.inner_dtype().to_physical(),
        )
    }
}
