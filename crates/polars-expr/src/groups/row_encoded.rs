use hashbrown::hash_table::{Entry, HashTable};
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_unordered;
use polars_utils::aliases::PlRandomState;
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;

use super::*;

struct Group {
    key_hash: u64,
    key_offset: usize,
    key_length: u32,
    group_idx: IdxSize,
}

#[derive(Default)]
pub struct RowEncodedHashGrouper {
    table: HashTable<Group>,
    key_data: Vec<u8>,
    random_state: PlRandomState,
}

impl RowEncodedHashGrouper {
    pub fn new(random_state: PlRandomState) -> Self {
        Self { random_state, ..Default::default() }
    }

    fn insert_key(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        let num_groups = self.table.len();
        let entry = self.table.entry(
            hash,
            |g| unsafe {
                if hash != g.key_hash {
                    return false;
                }

                let group_key = self.key_data.get_unchecked(g.key_offset..g.key_offset + g.key_length as usize);
                key == group_key
            },
            |g| g.key_hash,
        );
        
        match entry {
            Entry::Occupied(e) => e.get().group_idx,
            Entry::Vacant(e) => {
                let group_idx: IdxSize = num_groups.try_into().unwrap();
                let group = Group {
                    key_hash: hash,
                    key_offset: self.key_data.len(),
                    key_length: key.len().try_into().unwrap(),
                    group_idx
                };
                e.insert(group);
                group_idx
            }
        }
    }
}

impl Grouper for RowEncodedHashGrouper {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new(self.random_state.clone()))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn insert_keys(&mut self, keys: &[Column], group_idxs: &mut Vec<IdxSize>) {
        let series = keys
            .iter()
            .map(|c| c.as_materialized_series().clone())
            .collect_vec();
        let keys_encoded = _get_rows_encoded_unordered(&series[..])
            .unwrap()
            .into_array();
        assert!(keys_encoded.len() == keys.len());

        group_idxs.clear();
        group_idxs.reserve(keys.len());
        for key in keys_encoded.values_iter() {
            let hash = self.random_state.hash_one(key);
            unsafe {
                group_idxs.push_unchecked(self.insert_key(hash, key));
            }
        }
    }

    fn combine(&mut self, other: Box<dyn Grouper>, group_idxs: &mut Vec<IdxSize>) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        group_idxs.clear();
        group_idxs.reserve(other.table.len());
        self.table.reserve(other.table.len(), |g| g.key_hash); // TODO: cardinality estimation.
        for group in other.table.iter() {
            unsafe {
                let group_key = other.key_data.get_unchecked(group.key_offset..group.key_offset + group.key_length as usize);
                group_idxs.push_unchecked(self.insert_key(group.key_hash, group_key));
            }
        }
    }

    fn partition_into(
        &self,
        _seed: u64,
        _partitions: &mut [Box<dyn Grouper>],
        _partition_idxs: &mut Vec<IdxSize>,
        _group_idxs: &mut Vec<IdxSize>,
    ) {
        unimplemented!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
