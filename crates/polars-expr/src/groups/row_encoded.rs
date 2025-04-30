use arrow::array::Array;
use polars_row::RowEncodingOptions;
use polars_utils::idx_map::bytes_idx_map::{BytesIndexMap, Entry};
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;

use self::row_encode::get_row_encoding_context;
use super::*;
use crate::hash_keys::HashKeys;

#[derive(Default)]
pub struct RowEncodedHashGrouper {
    idx_map: BytesIndexMap<()>,
}

impl RowEncodedHashGrouper {
    pub fn new() -> Self {
        Self {
            idx_map: BytesIndexMap::new(),
        }
    }

    fn insert_key(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        match self.idx_map.entry(hash, key) {
            Entry::Occupied(o) => o.index(),
            Entry::Vacant(v) => {
                let index = v.index();
                v.insert(());
                index
            },
        }
    }

    fn contains_key(&self, hash: u64, key: &[u8]) -> bool {
        self.idx_map.contains_key(hash, key)
    }

    fn finalize_keys(&self, key_schema: &Schema, mut key_rows: Vec<&[u8]>) -> DataFrame {
        let key_dtypes = key_schema
            .iter()
            .map(|(_name, dt)| dt.to_physical().to_arrow(CompatLevel::newest()))
            .collect::<Vec<_>>();
        let ctxts = key_schema
            .iter()
            .map(|(_, dt)| get_row_encoding_context(dt, false))
            .collect::<Vec<_>>();
        let fields = vec![RowEncodingOptions::new_unsorted(); key_dtypes.len()];
        let key_columns =
            unsafe { polars_row::decode::decode_rows(&mut key_rows, &fields, &ctxts, &key_dtypes) };

        let cols = key_schema
            .iter()
            .zip(key_columns)
            .map(|((name, dt), col)| {
                let s = Series::try_from((name.clone(), col)).unwrap();
                unsafe { s.from_physical_unchecked(dt) }
                    .unwrap()
                    .into_column()
            })
            .collect();
        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }
}

impl Grouper for RowEncodedHashGrouper {
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
        keys: &HashKeys,
        subset: &[IdxSize],
        group_idxs: Option<&mut Vec<IdxSize>>,
    ) {
        let HashKeys::RowEncoded(keys) = keys else {
            unreachable!()
        };

        unsafe {
            if let Some(group_idxs) = group_idxs {
                group_idxs.reserve(subset.len());
                keys.for_each_hash_subset(subset, |idx, opt_hash| {
                    if let Some(hash) = opt_hash {
                        let key = keys.keys.value_unchecked(idx as usize);
                        group_idxs.push_unchecked(self.insert_key(hash, key));
                    }
                });
            } else {
                keys.for_each_hash_subset(subset, |idx, opt_hash| {
                    if let Some(hash) = opt_hash {
                        let key = keys.keys.value_unchecked(idx as usize);
                        self.insert_key(hash, key);
                    }
                });
            }
        }
    }

    fn get_keys_in_group_order(&self, schema: &Schema) -> DataFrame {
        unsafe {
            let mut key_rows: Vec<&[u8]> = Vec::with_capacity(self.idx_map.len() as usize);
            for (_, key) in self.idx_map.iter_hash_keys() {
                key_rows.push_unchecked(key);
            }
            self.finalize_keys(schema, key_rows)
        }
    }

    /// # Safety
    /// All groupers must be a RowEncodedHashGrouper.
    unsafe fn probe_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        probe_matches: &mut Vec<IdxSize>,
    ) {
        let HashKeys::RowEncoded(keys) = keys else {
            unreachable!()
        };
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            if keys.keys.has_nulls() {
                for (idx, hash) in keys.hashes.values_iter().enumerate_idx() {
                    let has_group = if let Some(key) = keys.keys.get_unchecked(idx as usize) {
                        let p = partitioner.hash_to_partition(*hash);
                        let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                        let grouper =
                            &*(dyn_grouper as *const dyn Grouper as *const RowEncodedHashGrouper);
                        grouper.contains_key(*hash, key)
                    } else {
                        false
                    };

                    if has_group != invert {
                        probe_matches.push(idx);
                    }
                }
            } else {
                for (idx, (hash, key)) in keys
                    .hashes
                    .values_iter()
                    .zip(keys.keys.values_iter())
                    .enumerate_idx()
                {
                    let p = partitioner.hash_to_partition(*hash);
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const RowEncodedHashGrouper);
                    if grouper.contains_key(*hash, key) != invert {
                        probe_matches.push(idx);
                    }
                }
            }
        }
    }

    /// # Safety
    /// All groupers must be a RowEncodedHashGrouper.
    unsafe fn contains_key_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        contains_key: &mut BitmapBuilder,
    ) {
        let HashKeys::RowEncoded(keys) = keys else {
            unreachable!()
        };
        assert!(partitioner.num_partitions() == groupers.len());

        unsafe {
            if keys.keys.has_nulls() {
                for (idx, hash) in keys.hashes.values_iter().enumerate_idx() {
                    let has_group = if let Some(key) = keys.keys.get_unchecked(idx as usize) {
                        let p = partitioner.hash_to_partition(*hash);
                        let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                        let grouper =
                            &*(dyn_grouper as *const dyn Grouper as *const RowEncodedHashGrouper);
                        grouper.contains_key(*hash, key)
                    } else {
                        false
                    };

                    contains_key.push(has_group != invert);
                }
            } else {
                for (hash, key) in keys.hashes.values_iter().zip(keys.keys.values_iter()) {
                    let p = partitioner.hash_to_partition(*hash);
                    let dyn_grouper: &dyn Grouper = &**groupers.get_unchecked(p);
                    let grouper =
                        &*(dyn_grouper as *const dyn Grouper as *const RowEncodedHashGrouper);
                    contains_key.push(grouper.contains_key(*hash, key) != invert);
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
