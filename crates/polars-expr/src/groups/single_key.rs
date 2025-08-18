use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use polars_utils::idx_map::total_idx_map::{Entry, TotalIndexMap};
use polars_utils::total_ord::{TotalEq, TotalHash};
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::{HashKeys, for_each_hash_single};

#[derive(Default)]
pub struct SingleKeyHashGrouper<T: PolarsDataType> {
    idx_map: TotalIndexMap<T::Physical<'static>, ()>,
    null_idx: IdxSize,
}

impl<K, T: PolarsDataType> SingleKeyHashGrouper<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq,
{
    pub fn new() -> Self {
        Self {
            idx_map: TotalIndexMap::default(),
            null_idx: IdxSize::MAX,
        }
    }

    #[inline(always)]
    fn insert_key(&mut self, key: T::Physical<'static>) -> IdxSize {
        match self.idx_map.entry(key) {
            Entry::Occupied(o) => o.index(),
            Entry::Vacant(v) => {
                let index = v.index();
                v.insert(());
                index
            },
        }
    }

    #[inline(always)]
    fn insert_null(&mut self) -> IdxSize {
        if self.null_idx == IdxSize::MAX {
            self.null_idx = self.idx_map.push_unmapped_entry(T::Physical::default(), ());
        }
        self.null_idx
    }

    #[inline(always)]
    fn contains_key(&self, key: &T::Physical<'static>) -> bool {
        self.idx_map.get(key).is_some()
    }

    #[inline(always)]
    fn contains_null(&self) -> bool {
        self.null_idx < IdxSize::MAX
    }

    fn finalize_keys(&self, schema: &Schema, keys: Vec<T::Physical<'static>>) -> DataFrame {
        let (name, dtype) = schema.get_at_index(0).unwrap();
        let mut keys =
            T::Array::from_vec(keys, dtype.to_physical().to_arrow(CompatLevel::newest()));
        if self.null_idx < IdxSize::MAX {
            let mut validity = MutableBitmap::new();
            validity.extend_constant(keys.len(), true);
            validity.set(self.null_idx as usize, false);
            keys = keys.with_validity_typed(Some(validity.freeze()));
        }
        unsafe {
            let s =
                Series::from_chunks_and_dtype_unchecked(name.clone(), vec![Box::new(keys)], dtype);
            DataFrame::new(vec![Column::from(s)]).unwrap()
        }
    }
}

impl<K, T: PolarsDataType> Grouper for SingleKeyHashGrouper<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: Default + TotalHash + TotalEq + Clone + Send + Sync + 'static,
{
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_groups(&self) -> IdxSize {
        self.idx_map.len()
    }

    unsafe fn insert_keys_subset(
        &mut self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        group_idxs: Option<&mut Vec<IdxSize>>,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };
        let ca: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        let arr = ca.downcast_as_array();

        unsafe {
            if arr.has_nulls() {
                if hash_keys.null_is_valid {
                    let groups = subset.iter().map(|idx| {
                        let opt_k = arr.get_unchecked(*idx as usize);
                        if let Some(k) = opt_k {
                            self.insert_key(k)
                        } else {
                            self.insert_null()
                        }
                    });
                    if let Some(group_idxs) = group_idxs {
                        group_idxs.reserve(subset.len());
                        group_idxs.extend(groups);
                    } else {
                        groups.for_each(drop);
                    }
                } else {
                    let groups = subset.iter().filter_map(|idx| {
                        let opt_k = arr.get_unchecked(*idx as usize);
                        opt_k.map(|k| self.insert_key(k))
                    });
                    if let Some(group_idxs) = group_idxs {
                        group_idxs.reserve(subset.len());
                        group_idxs.extend(groups);
                    } else {
                        groups.for_each(drop);
                    }
                }
            } else {
                let groups = subset.iter().map(|idx| {
                    let k = arr.value_unchecked(*idx as usize);
                    self.insert_key(k)
                });
                if let Some(group_idxs) = group_idxs {
                    group_idxs.reserve(subset.len());
                    group_idxs.extend(groups);
                } else {
                    groups.for_each(drop);
                }
            }
        }
    }

    fn get_keys_in_group_order(&self, schema: &Schema) -> DataFrame {
        unsafe {
            let mut key_rows = Vec::with_capacity(self.idx_map.len() as usize);
            for key in self.idx_map.iter_keys() {
                key_rows.push_unchecked(key.clone());
            }
            self.finalize_keys(schema, key_rows)
        }
    }

    /// # Safety
    /// All groupers must be a SingleKeyHashGrouper<T>.
    unsafe fn probe_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        probe_matches: &mut Vec<IdxSize>,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };
        let ca: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        let arr = ca.downcast_as_array();
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            let null_p = partitioner.null_partition();
            for_each_hash_single(ca, &hash_keys.random_state, |idx, opt_h| {
                let has_group = if let Some(h) = opt_h {
                    let p = partitioner.hash_to_partition(h);
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const SingleKeyHashGrouper<T>);
                    let key = arr.value_unchecked(idx as usize);
                    grouper.contains_key(&key)
                } else {
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(null_p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const SingleKeyHashGrouper<T>);
                    grouper.contains_null()
                };

                if has_group != invert {
                    probe_matches.push(idx);
                }
            });
        }
    }

    /// # Safety
    /// All groupers must be a SingleKeyHashGrouper<T>.
    unsafe fn contains_key_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        hash_keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        contains_key: &mut BitmapBuilder,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };
        let ca: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        let arr = ca.downcast_as_array();
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            let null_p = partitioner.null_partition();
            for_each_hash_single(ca, &hash_keys.random_state, |idx, opt_h| {
                let has_group = if let Some(h) = opt_h {
                    let p = partitioner.hash_to_partition(h);
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const SingleKeyHashGrouper<T>);
                    let key = arr.value_unchecked(idx as usize);
                    grouper.contains_key(&key)
                } else {
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(null_p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const SingleKeyHashGrouper<T>);
                    grouper.contains_null()
                };

                contains_key.push(has_group != invert);
            });
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
