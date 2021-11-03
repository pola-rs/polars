use crate::prelude::*;
use arrow::bitmap::MutableBitmap;
use polars_arrow::array::PolarsArray;
use polars_arrow::bit_util::unset_bit_raw;
use polars_arrow::kernels::take::take_value_indices_from_list;
use std::convert::TryFrom;

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

/// Forked and adapted from arrow-rs
/// This is faster because it does no bounds checks and allocates directly into aligned memory
///
/// # Safety
/// No bounds checks
pub(crate) unsafe fn take_list_unchecked(
    values: &ListArray<i64>,
    indices: &UInt32Array,
) -> ListArray<i64> {
    // taking the whole list or a contiguous sublist
    let (list_indices, offsets) = take_value_indices_from_list(values, indices);

    // tmp series so that we can take primitives from it
    let s = Series::try_from(("", values.values().clone() as ArrayRef)).unwrap();
    let taken = s
        .take_unchecked(&UInt32Chunked::new_from_chunks(
            "",
            vec![Arc::new(list_indices) as ArrayRef],
        ))
        .unwrap();
    let taken = taken.chunks()[0].clone();

    let validity =
        // if null count > 0
        if values.has_validity() || indices.has_validity() {
            // determine null buffer, which are a function of `values` and `indices`
            let mut validity = MutableBitmap::with_capacity(indices.len());
            let validity_ptr = validity.as_slice().as_ptr() as *mut u8;
            validity.extend_constant(indices.len(), true);

            {
                offsets.as_slice().windows(2).enumerate().for_each(
                    |(i, window): (usize, &[i64])| {
                        if window[0] == window[1] {
                            // offsets are equal, slot is null
                            unset_bit_raw(validity_ptr, i);
                        }
                    },
                );
            }
            Some(validity.into())
        } else {
            None
        };
    ListArray::from_data(values.data_type().clone(), offsets.into(), taken, validity)
}
