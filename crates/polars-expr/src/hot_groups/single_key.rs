use std::hash::BuildHasher;

use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::SingleKeys;
use crate::hot_groups::fixed_index_table::FixedIndexTable;

pub struct SingleKeyHashHotGrouper<T: PolarsDataType> {
    dtype: DataType,
    table: FixedIndexTable<T::Physical<'static>>,
    evicted_keys: Vec<T::Physical<'static>>,
    null_idx: IdxSize,
    random_state: PlRandomState,
}

impl<K, T: PolarsDataType> SingleKeyHashHotGrouper<T>
where
    ChunkedArray<T>: IntoSeries,
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq + Send + Sync + 'static,
{
    pub fn new(dtype: DataType, max_groups: usize) -> Self {
        Self {
            dtype,
            table: FixedIndexTable::new(max_groups.try_into().unwrap()),
            evicted_keys: Vec::new(),
            null_idx: IdxSize::MAX,
            random_state: PlRandomState::default(),
        }
    }

    #[inline(always)]
    fn insert_key<R: BuildHasher>(
        &mut self,
        k: T::Physical<'static>,
        random_state: &R,
    ) -> Option<EvictIdx> {
        let h = random_state.tot_hash_one(&k);
        self.table.insert_key(
            h,
            k,
            |a, b| a.tot_eq(b),
            |k| k,
            |k, ev_k| self.evicted_keys.push(core::mem::replace(ev_k, k)),
        )
    }

    #[inline(always)]
    fn insert_null(&mut self) -> Option<EvictIdx> {
        if self.null_idx == IdxSize::MAX {
            self.null_idx = self.table.push_unmapped_key(T::Physical::default());
        }
        Some(EvictIdx::new(self.null_idx, false))
    }

    fn finalize_keys(&self, keys: Vec<T::Physical<'static>>, add_mask: bool) -> HashKeys {
        let mut keys = T::Array::from_vec(
            keys,
            self.dtype.to_physical().to_arrow(CompatLevel::newest()),
        );
        if add_mask && self.null_idx < IdxSize::MAX {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(keys.len(), true);
            validity.set(self.null_idx as usize, false);
            keys = keys.with_validity_typed(Some(validity.freeze()));
        }

        unsafe {
            let s = Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![Box::new(keys)],
                &self.dtype,
            );
            HashKeys::Single(SingleKeys {
                keys: s,
                null_is_valid: self.null_idx < IdxSize::MAX,
                random_state: self.random_state,
            })
        }
    }
}

impl<K, T> HotGrouper for SingleKeyHashHotGrouper<T>
where
    ChunkedArray<T>: IntoSeries,
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq + Clone + Send + Sync + 'static,
{
    fn new_empty(&self, max_groups: usize) -> Box<dyn HotGrouper> {
        Box::new(Self::new(self.dtype.clone(), max_groups))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn insert_keys(
        &mut self,
        hash_keys: &HashKeys,
        hot_idxs: &mut Vec<IdxSize>,
        hot_group_idxs: &mut Vec<EvictIdx>,
        cold_idxs: &mut Vec<IdxSize>,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };

        // Preserve random state if non-empty.
        if !hash_keys.keys.is_empty() {
            self.random_state = hash_keys.random_state;
        }

        let keys: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        hot_idxs.reserve(keys.len());
        hot_group_idxs.reserve(keys.len());
        cold_idxs.reserve(keys.len());

        let mut push_g = |idx: usize, opt_g: Option<EvictIdx>| unsafe {
            if let Some(g) = opt_g {
                hot_idxs.push_unchecked(idx as IdxSize);
                hot_group_idxs.push_unchecked(g);
            } else {
                cold_idxs.push_unchecked(idx as IdxSize);
            }
        };

        let mut idx = 0;
        for arr in keys.downcast_iter() {
            if arr.has_nulls() {
                if hash_keys.null_is_valid {
                    for opt_k in arr.iter() {
                        if let Some(k) = opt_k {
                            push_g(idx, self.insert_key(k, &hash_keys.random_state));
                        } else {
                            push_g(idx, self.insert_null());
                        }
                        idx += 1;
                    }
                } else {
                    for opt_k in arr.iter() {
                        if let Some(k) = opt_k {
                            push_g(idx, self.insert_key(k, &hash_keys.random_state));
                        }
                        idx += 1;
                    }
                }
            } else {
                for k in arr.values_iter() {
                    let g = self.insert_key(k, &hash_keys.random_state);
                    push_g(idx, g);
                    idx += 1;
                }
            }
        }
    }

    fn keys(&self) -> HashKeys {
        self.finalize_keys(self.table.keys().to_vec(), true)
    }

    fn num_evictions(&self) -> usize {
        self.evicted_keys.len()
    }

    fn take_evicted_keys(&mut self) -> HashKeys {
        let keys = core::mem::take(&mut self.evicted_keys);
        self.finalize_keys(keys, false)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
