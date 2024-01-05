use arrow::legacy::kernels::fixed_size_list::{
    sub_fixed_size_list_get, sub_fixed_size_list_get_literal,
};
use polars_core::datatypes::ArrayChunked;
use polars_core::prelude::arity::binary_to_series;

use super::*;

fn array_get_literal(ca: &ArrayChunked, idx: i64) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| sub_fixed_size_list_get_literal(arr, idx))
        .collect::<Vec<_>>();
    Series::try_from((ca.name(), chunks))
        .unwrap()
        .cast(&ca.inner_dtype())
}

/// Get the value by literal index in the array.
/// So index `0` would return the first item of every sub-array
/// and index `-1` would return the last item of every sub-array
/// if an index is out of bounds, it will return a `None`.
pub fn array_get(ca: &ArrayChunked, index: &Int64Chunked) -> PolarsResult<Series> {
    match index.len() {
        1 => {
            let index = index.get(0);
            if let Some(index) = index {
                array_get_literal(ca, index)
            } else {
                polars_bail!(ComputeError: "unexpected null index received in `arr.get`")
            }
        },
        len if len == ca.len() => {
            let out = binary_to_series(ca, index, |arr, idx| sub_fixed_size_list_get(arr, idx));
            out?.cast(&ca.inner_dtype())
        },
        len => polars_bail!(
            ComputeError:
            "`arr.get` expression got an index array of length {} while the array has {} elements",
            len, ca.len()
        ),
    }
}
