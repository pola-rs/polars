use arrow::array::{BooleanArray, ListArray};
use arrow::bitmap::Bitmap;

use super::*;

fn list_all_any<F>(arr: &ListArray<i64>, op: F, fast_value: bool) -> PolarsResult<BooleanArray>
where
    F: Fn(&BooleanArray) -> bool,
{
    let offsets = arr.offsets().as_slice();
    let values = arr.values();

    polars_ensure!(values.dtype() == &ArrowDataType::Boolean, ComputeError: "expected boolean elements in list");

    let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
    let validity = arr.validity().cloned();

    // Fast path where all/none values set.
    if op(values) == fast_value {
        let bits = Bitmap::new_with_value(fast_value, arr.len());
        return Ok(BooleanArray::from_data_default(bits, None).with_validity(validity));
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
    let chunks = ca.downcast_iter().map(|arr| {
        list_all_any(
            arr,
            |a| polars_compute::boolean::all(a).unwrap_or(true),
            true,
        )
    });
    Ok(BooleanChunked::try_from_chunk_iter(ca.name().clone(), chunks)?.into_series())
}

pub(super) fn list_any(ca: &ListChunked) -> PolarsResult<Series> {
    let chunks = ca.downcast_iter().map(|arr| {
        list_all_any(
            arr,
            |a| polars_compute::boolean::any(a).unwrap_or(false),
            false,
        )
    });
    Ok(BooleanChunked::try_from_chunk_iter(ca.name().clone(), chunks)?.into_series())
}
