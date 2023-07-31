use arrow::array::{BooleanArray, ListArray};

use super::*;

fn list_all_any<F>(arr: &ListArray<i64>, op: F, is_all: bool) -> PolarsResult<ArrayRef>
where
    F: Fn(&BooleanArray) -> bool,
{
    let offsets = arr.offsets().as_slice();
    let values = arr.values();

    polars_ensure!(values.data_type() == &ArrowDataType::Boolean, ComputeError: "expected boolean elements in list");

    let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
    let validity = arr.validity().cloned();

    // fast path where all values set
    // all is free
    let all_set = arrow::compute::boolean::all(values);
    if all_set && is_all {
        return Ok(BooleanChunked::full("", true, arr.len()).chunks()[0]
            .clone()
            .with_validity(validity));
    }

    let mut start = offsets[0] as usize;
    let iter = offsets[1..].iter().map(|&end| {
        let end = end as usize;
        let len = end - start;
        // TODO!
        // we can speed this upp if the boolean array doesn't have nulls
        // Then we can work directly on the byte slice.
        let val = unsafe { values.clone().sliced_unchecked(start, len) };
        start = end;
        op(&val)
    });

    Ok(Box::new(
        BooleanArray::from_trusted_len_values_iter(iter).with_validity(validity),
    ))
}

pub(super) fn list_all(ca: &ListChunked) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| list_all_any(arr, arrow::compute::boolean::all, true))
        .collect::<PolarsResult<Vec<_>>>()?;

    unsafe { Ok(BooleanChunked::from_chunks(ca.name(), chunks).into_series()) }
}
pub(super) fn list_any(ca: &ListChunked) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| list_all_any(arr, arrow::compute::boolean::any, false))
        .collect::<PolarsResult<Vec<_>>>()?;

    unsafe { Ok(BooleanChunked::from_chunks(ca.name(), chunks).into_series()) }
}
