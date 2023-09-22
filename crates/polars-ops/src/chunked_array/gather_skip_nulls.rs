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


#[cfg(test)]
mod test {
    use std::ops::Range;

    use rand::distributions::uniform::SampleUniform;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use super::*;

    fn random_vec<T: SampleUniform + PartialOrd + Clone, R: Rng>(
        rng: &mut R,
        val: Range<T>,
        len_range: Range<usize>,
    ) -> Vec<T> {
        let n = rng.gen_range(len_range);
        (0..n).map(|_| rng.gen_range(val.clone())).collect()
    }

    fn random_filter<T: Clone, R: Rng>(rng: &mut R, v: &[T], pr: Range<f64>) -> Vec<Option<T>> {
        let p = rng.gen_range(pr);
        let rand_filter = |x| Some(x).filter(|_| rng.gen::<f64>() < p);
        v.iter().cloned().map(rand_filter).collect()
    }

    fn ref_gather_nulls(v: Vec<Option<u32>>, idx: Vec<Option<usize>>) -> Option<Vec<Option<u32>>> {
        if idx.iter().any(|oi| oi.map(|i| i >= v.len()) == Some(true)) {
            return None;
        }
        Some(idx.into_iter().map(|i| v[i?]).collect())
    }

    fn test_equal_ref(ca: &UInt32Chunked, idx_ca: &IdxCa) {
        let ref_ca: Vec<Option<u32>> = ca.into_iter().collect();
        let ref_idx_ca: Vec<Option<usize>> =
            (&idx_ca).into_iter().map(|i| Some(i? as usize)).collect();
        let gather = ca.gather_skip_nulls(idx_ca).ok();
        let ref_gather = ref_gather_nulls(ref_ca, ref_idx_ca);
        assert!(gather.map(|ca| ca.into_iter().collect()) == ref_gather);
    }

    #[rustfmt::skip]
    #[test]
    fn gather_skip_nulls() {
        let mut rng = SmallRng::seed_from_u64(0xdeadbeef);

        for _test in 0..20 {
            let num_elem_chunks = rng.gen_range(1..10);
            let elem_chunks: Vec<_> = (0..num_elem_chunks).map(|_| random_vec(&mut rng, 0..u32::MAX, 0..10)).collect();
            let null_elem_chunks: Vec<_> = elem_chunks.iter().map(|c| random_filter(&mut rng, c, 0.7..1.0)).collect();
            let num_elems: usize = elem_chunks.iter().map(|c| c.len()).sum();

            let num_idx_chunks = rng.gen_range(1..10);
            let idx_chunks: Vec<_> = (0..num_idx_chunks).map(|_| random_vec(&mut rng, 0..num_elems as IdxSize, 0..10)).collect();
            let null_idx_chunks: Vec<_> = idx_chunks.iter().map(|c| random_filter(&mut rng, c, 0.7..1.0)).collect();
            
            let nonnull_ca = UInt32Chunked::from_chunk_iter("", elem_chunks.iter().cloned().map(|v| v.into_iter().collect_arr()));
            let ca = UInt32Chunked::from_chunk_iter("", null_elem_chunks.iter().cloned().map(|v| v.into_iter().collect_arr()));
            let nonnull_idx_ca = IdxCa::from_chunk_iter("", idx_chunks.iter().cloned().map(|v| v.into_iter().collect_arr()));
            let idx_ca = IdxCa::from_chunk_iter("", null_idx_chunks.iter().cloned().map(|v| v.into_iter().collect_arr()));
            
            test_equal_ref(&ca, &idx_ca);
            test_equal_ref(&ca, &nonnull_idx_ca);
            test_equal_ref(&nonnull_ca, &idx_ca);
            test_equal_ref(&nonnull_ca, &nonnull_idx_ca);
        }
    }
}
