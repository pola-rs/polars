use arrow::array::Array;
use arrow::bitmap::bitmask::BitMask;
use arrow::compute::concatenate::concatenate_validities;
use bytemuck::allocation::zeroed_vec;
use polars_core::prelude::gather::check_bounds_ca;
use polars_core::prelude::*;
use polars_utils::index::check_bounds;


/// # Safety
/// For each index pair, pair.0 < len && pair.1 < ca.null_count() must hold.
unsafe fn gather_skip_null_idx_pairs_unchecked<'a, T: PolarsDataType>(
    ca: &'a ChunkedArray<T>,
    mut index_pairs: Vec<(IdxSize, IdxSize)>,
    len: usize,
) -> Vec<T::ZeroablePhysical<'a>> {
    if index_pairs.len() == 0 {
        return Vec::new();
    }

    // We sort by gather index so we can do the null scan in one pass.
    index_pairs.sort_unstable_by_key(|t| t.1);
    let mut pair_iter = index_pairs.iter().copied();
    let (mut out_idx, mut get_idx);
    (out_idx, get_idx) = pair_iter.next().unwrap();

    let mut out: Vec<T::ZeroablePhysical<'a>> = zeroed_vec(len);
    let mut global_nonnull_scanned = 0;
    'outer: for arr in ca.downcast_iter() {
        let arr_nonnull_len = arr.len() - arr.null_count();
        let mut arr_scan_offset = 0;
        let mut nonnull_before_offset = 0;
        let mask = arr.validity().map(BitMask::from_bitmap).unwrap_or_default();

        // Is our next get_idx in this array?
        while get_idx as usize - global_nonnull_scanned < arr_nonnull_len {
            let nonnull_idx_in_arr = get_idx as usize - global_nonnull_scanned;

            let get_idx_in_arr = if arr.null_count() == 0 {
                // Happy fast path for full non-null array.
                nonnull_idx_in_arr
            } else {
                mask.nth_set_bit_idx(nonnull_idx_in_arr - nonnull_before_offset, arr_scan_offset)
                    .unwrap()
            };

            unsafe {
                let val = arr.value_unchecked(get_idx_in_arr);
                *out.get_unchecked_mut(out_idx as usize) = val.into();
            }

            arr_scan_offset = get_idx_in_arr;
            nonnull_before_offset = get_idx as usize;

            let Some(next_pair) = pair_iter.next() else {
                break 'outer;
            };
            (out_idx, get_idx) = next_pair;
        }

        global_nonnull_scanned += arr_nonnull_len;
    }

    out
}

pub trait ChunkGatherSkipNulls<I: ?Sized>: Sized {
    fn gather_skip_nulls(&self, indices: &I) -> PolarsResult<Self>;
}

impl<T: PolarsDataType> ChunkGatherSkipNulls<[IdxSize]> for ChunkedArray<T> {
    fn gather_skip_nulls(&self, indices: &[IdxSize]) -> PolarsResult<Self> {
        if self.null_count() == 0 {
            return self.take(indices);
        }

        let bound = self.len() - self.null_count();
        check_bounds(indices, bound as IdxSize)?;

        let index_pairs: Vec<_> = indices
            .iter()
            .enumerate()
            .map(|(out_idx, get_idx)| (out_idx as IdxSize, *get_idx))
            .collect();
        let gathered =
            unsafe { gather_skip_null_idx_pairs_unchecked(self, index_pairs, indices.len()) };
        let arr = T::Array::from_zeroable_vec(gathered, self.dtype().clone());
        Ok(ChunkedArray::from_chunk_iter_like(self, [arr]))
    }
}

impl<T: PolarsDataType> ChunkGatherSkipNulls<IdxCa> for ChunkedArray<T> {
    fn gather_skip_nulls(&self, indices: &IdxCa) -> PolarsResult<Self> {
        if self.null_count() == 0 {
            return self.take(indices);
        }

        let bound = self.len() - self.null_count();
        check_bounds_ca(indices, bound as IdxSize)?;

        let index_pairs: Vec<_> = if indices.null_count() == 0 {
            indices
                .downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .enumerate()
                .map(|(out_idx, get_idx)| (out_idx as IdxSize, *get_idx))
                .collect()
        } else {
            // Filter *after* the enumerate so we place the non-null gather
            // requests at the right places.
            indices
                .downcast_iter()
                .flat_map(|arr| arr.iter())
                .enumerate()
                .filter_map(|(out_idx, get_idx)| Some((out_idx as IdxSize, *get_idx?)))
                .collect()
        };
        let gathered = unsafe {
            gather_skip_null_idx_pairs_unchecked(self, index_pairs, indices.as_ref().len())
        };

        let mut arr = T::Array::from_zeroable_vec(gathered, self.dtype().clone());
        if indices.null_count() > 0 {
            let array_refs: Vec<&dyn Array> = indices.chunks().iter().map(|x| &**x).collect();
            arr = arr.with_validity_typed(concatenate_validities(&array_refs));
        }
        Ok(ChunkedArray::from_chunk_iter_like(self, [arr]))
    }
}
