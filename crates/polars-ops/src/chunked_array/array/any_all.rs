use arrow::array::{BooleanArray, FixedSizeListArray};
use arrow::bitmap::MutableBitmap;
use arrow::legacy::utils::CustomIterTools;

use super::*;

fn array_all_any<F>(arr: &FixedSizeListArray, op: F, is_all: bool) -> PolarsResult<BooleanArray>
where
    F: Fn(&BooleanArray) -> bool,
{
    let values = arr.values();

    polars_ensure!(values.data_type() == &ArrowDataType::Boolean, ComputeError: "expected boolean elements in array");

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

    let len = arr.size();
    let iter = (0..values.len()).step_by(len).map(|start| {
        // SAFETY: start + len is in bound guarded by invariant of FixedSizeListArray
        let val = unsafe { values.clone().sliced_unchecked(start, len) };
        op(&val)
    });

    Ok(BooleanArray::from_trusted_len_values_iter(
        // SAFETY: we evaluate for every sub-array, the length is equals to arr.len().
        unsafe { iter.trust_my_length(arr.len()) },
    )
    .with_validity(validity))
}

pub(super) fn array_all(ca: &ArrayChunked) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| array_all_any(arr, arrow::compute::boolean::all, true));
    Ok(BooleanChunked::try_from_chunk_iter(ca.name(), chunks)?.into_series())
}

pub(super) fn array_any(ca: &ArrayChunked) -> PolarsResult<Series> {
    let chunks = ca
        .downcast_iter()
        .map(|arr| array_all_any(arr, arrow::compute::boolean::any, false));
    Ok(BooleanChunked::try_from_chunk_iter(ca.name(), chunks)?.into_series())
}
