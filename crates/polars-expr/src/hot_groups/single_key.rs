use polars_arrow::array::Array;
use polars_arrow::bitmap::MutableBitmap;
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
    fn insert_key(
        &mut self,
        h: u64,
        k: T::Physical<'static>,
        force_hot: bool,
        next_h: u64,
    ) -> Option<EvictIdx> {
        self.table.insert_key(
            h,
            k,
            force_hot,
            next_h,
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
                random_state: self.random_state.clone(),
            })
        }
    }
}

impl<K, T> HotGrouper for SingleKeyHashHotGrouper<T>
where
    for<'a> T: PolarsPhysicalType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq + Clone + Send + Sync + 'static,
{
    fn new_empty(&self, max_groups: usize) -> Box<dyn HotGrouper> {
        Box::new(Self::new(self.dtype.clone(), max_groups))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn num_slots(&self) -> usize {
        self.table.num_slots()
    }

    fn double(&mut self) {
        self.table.double();
    }

    fn insert_keys(
        &mut self,
        hash_keys: &HashKeys,
        hot_idxs: &mut Vec<IdxSize>,
        hot_group_idxs: &mut Vec<EvictIdx>,
        cold_idxs: &mut Vec<IdxSize>,
        force_hot: bool,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };

        // Preserve random state if non-empty.
        if !hash_keys.keys.is_empty() {
            self.random_state = hash_keys.random_state.clone();
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

        let random_state = &hash_keys.random_state;
        let mut idx = 0;
        for arr in keys.downcast_iter() {
            let n = arr.len();
            let hash_at = |i: usize| {
                if i < n {
                    random_state.tot_hash_one(unsafe { arr.value_unchecked(i) })
                } else {
                    u64::MAX
                }
            };
            let mut next_h = hash_at(0);
            if arr.has_nulls() {
                if hash_keys.null_is_valid {
                    for (i, opt_k) in arr.iter().enumerate() {
                        let h = next_h;
                        next_h = hash_at(i + 1);
                        if let Some(k) = opt_k {
                            push_g(idx, self.insert_key(h, k, force_hot, next_h));
                        } else {
                            push_g(idx, self.insert_null());
                        }
                        idx += 1;
                    }
                } else {
                    for (i, opt_k) in arr.iter().enumerate() {
                        let h = next_h;
                        next_h = hash_at(i + 1);
                        if let Some(k) = opt_k {
                            push_g(idx, self.insert_key(h, k, force_hot, next_h));
                        }
                        idx += 1;
                    }
                }
            } else {
                for (i, k) in arr.values_iter().enumerate() {
                    let h = next_h;
                    next_h = hash_at(i + 1);
                    push_g(idx, self.insert_key(h, k, force_hot, next_h));
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
