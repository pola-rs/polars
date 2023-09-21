use arrow::array::Array;
use arrow::bitmap::bitmask::BitMask;
use bytemuck::allocation::zeroed_vec;
use polars_core::prelude::*;
use polars_utils::index::check_bounds;

unsafe fn gather_skip_nulls_idx_pairs_unchecked<'a, T: PolarsDataType>(
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
        let mut arr_nonnull_scanned = 0;
        let mask = arr.validity().map(BitMask::from_bitmap).unwrap_or_default();

        // Is our next get_idx in this array?
        while get_idx as usize - global_nonnull_scanned < arr_nonnull_len {
            let nonnull_idx_in_arr = get_idx as usize - global_nonnull_scanned;

            let get_idx_in_arr = if arr.null_count() == 0 {
                // Happy fast path for full non-null array.
                nonnull_idx_in_arr
            } else {
                // Skip blocks of 32 until our index is found inside the next 32 bits of the mask.
                let mut next_u32_mask = mask.get_u32(arr_scan_offset);
                while arr_nonnull_scanned + next_u32_mask.count_ones() as usize
                    <= nonnull_idx_in_arr
                {
                    arr_scan_offset += 32;
                    arr_nonnull_scanned += next_u32_mask.count_ones() as usize;
                    next_u32_mask = mask.get_u32(arr_scan_offset);
                }

                // Find index in mask.
                if next_u32_mask == u32::MAX {
                    // Happy fast path for dense non-null section.
                    arr_scan_offset += nonnull_idx_in_arr + 1 - arr_nonnull_scanned;
                    arr_nonnull_scanned = nonnull_idx_in_arr + 1;
                } else {
                    while arr_nonnull_scanned <= nonnull_idx_in_arr {
                        arr_scan_offset += 1;
                        arr_nonnull_scanned += (next_u32_mask & 1) as usize;
                        next_u32_mask >>= 1;
                    }
                }

                arr_scan_offset - 1
            };

            unsafe {
                let val = arr.value_unchecked(get_idx_in_arr);
                *out.get_unchecked_mut(out_idx as usize) = val.into();
            }

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

impl<T: PolarsDataType, I: AsRef<[IdxSize]>> ChunkGatherSkipNulls<I> for ChunkedArray<T> {
    fn gather_skip_nulls(&self, indices: &I) -> PolarsResult<Self> {
        if self.null_count() == 0 {
            return self.take(indices);
        }

        let bound = self.len() - self.null_count();
        check_bounds(indices.as_ref(), bound as IdxSize)?;

        let index_pairs: Vec<_> = indices
            .as_ref()
            .iter()
            .enumerate()
            .map(|(out_idx, get_idx)| (out_idx as IdxSize, *get_idx))
            .collect();
        let gathered = unsafe {
            gather_skip_nulls_idx_pairs_unchecked(self, index_pairs, indices.as_ref().len())
        };
        let arr = T::Array::from_zeroable_vec(gathered, self.dtype().clone());
        Ok(ChunkedArray::from_chunk_iter_like(self, [arr]))
    }
}
