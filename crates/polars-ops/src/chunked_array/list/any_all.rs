use arrow::array::{BooleanArray, ListArray};
use arrow::bitmap::MutableBitmap;

use super::*;

fn list_all_any<F>(arr: &ListArray<i64>, op: F, is_all: bool) -> PolarsResult<BooleanArray>
where
    F: Fn(&BooleanArray) -> bool,
{
    let offsets = arr.offsets().as_slice();
    let values = arr.values();

    polars_ensure!(values.data_type() == &ArrowDataType::Boolean, ComputeError: "expected boolean elements in list");

    let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
    let validity = arr.validity().cloned();

    // Fast path where all values set (all is free).
    if is_all {
        let all_set = arrow::compute::boolean::all(values);
        if all_set {
            let mut bits = MutableBitmap::with_capacity(arr.len());
            bits.extend_constant(arr.len(), true);
            return Ok(BooleanArray::from_data_default(bits.into(), None).with_validity(validity));
        }
    }

    let mut start = offsets[0] as usize;
    let iter = offsets[1..].iter().map(|&end| {
        let end = end as usize;
        let len = end - start;
        let val = unsafe { values.clone().sliced_unchecked(start, len) };
        start = end;
        op(&val)
    });

    Ok(BooleanArray::from_trusted_len_values_iter(iter).with_validity(validity))
}

pub(super) fn list_all(ca: &ListChunked) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| list_all_any(arr, arrow::compute::boolean::all, true));
    Ok(BooleanChunked::try_from_chunk_iter(ca.name(), chunks)?.into_series())
}

pub(super) fn list_any(ca: &ListChunked) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| list_all_any(arr, arrow::compute::boolean::any, false));
    Ok(BooleanChunked::try_from_chunk_iter(ca.name(), chunks)?.into_series())
}
