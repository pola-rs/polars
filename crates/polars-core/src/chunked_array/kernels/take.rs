use std::convert::TryFrom;

use polars_arrow::compute::take::bitmap::take_bitmap_unchecked;
use polars_arrow::compute::take::take_value_indices_from_list;
use polars_arrow::utils::combine_validities_and;

use crate::prelude::*;

/// Take kernel for multiple chunks. We directly return a ChunkedArray because that path chooses the fastest collection path.
pub(crate) fn take_primitive_iter_n_chunks<T: PolarsNumericType, I: IntoIterator<Item = usize>>(
    ca: &ChunkedArray<T>,
    indices: I,
) -> ChunkedArray<T> {
    let taker = ca.take_rand();
    indices.into_iter().map(|idx| taker.get(idx)).collect()
}

/// Take kernel for multiple chunks where an iterator can produce None values.
/// Used in join operations. We directly return a ChunkedArray because that path chooses the fastest collection path.
pub(crate) fn take_primitive_opt_iter_n_chunks<
    T: PolarsNumericType,
    I: IntoIterator<Item = Option<usize>>,
>(
    ca: &ChunkedArray<T>,
    indices: I,
) -> ChunkedArray<T> {
    let taker = ca.take_rand();
    indices
        .into_iter()
        .map(|opt_idx| opt_idx.and_then(|idx| taker.get(idx)))
        .collect()
}

/// This is faster because it does no bounds checks and allocates directly into aligned memory
///
/// # Safety
/// No bounds checks
pub(crate) unsafe fn take_list_unchecked(
    values: &ListArray<i64>,
    indices: &IdxArr,
) -> ListArray<i64> {
    // taking the whole list or a contiguous sublist
    let (list_indices, offsets) = take_value_indices_from_list(values, indices);

    // tmp series so that we can take primitives from it
    let s = Series::try_from(("", values.values().clone() as ArrayRef)).unwrap();
    let taken = s
        .take_unchecked(&IdxCa::from_chunks(
            "",
            vec![Box::new(list_indices) as ArrayRef],
        ))
        .unwrap();

    let taken = taken.array_ref(0).clone();

    let validity = if let Some(validity) = values.validity() {
        let validity = take_bitmap_unchecked(validity, indices.values().as_slice());
        combine_validities_and(Some(&validity), indices.validity())
    } else {
        indices.validity().cloned()
    };

    let dtype = ListArray::<i64>::default_datatype(taken.data_type().clone());
    // Safety:
    // offsets are monotonically increasing
    ListArray::new(dtype, offsets.into(), taken, validity)
}
