use std::mem::MaybeUninit;

use hashbrown::hash_table::{Entry, HashTable};
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_unordered;
use polars_row::EncodingField;
use polars_utils::aliases::PlRandomState;
use polars_utils::hashing::{folded_multiply, hash_to_partition};
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;
use rand::Rng;

use super::*;

struct Group {
    key_hash: u64,
    key_offset: usize,
    key_length: u32,
    group_idx: IdxSize,
}

impl Group {
    unsafe fn key<'k>(&self, key_data: &'k [u8]) -> &'k [u8] {
        key_data.get_unchecked(self.key_offset..self.key_offset + self.key_length as usize)
    }
}

#[derive(Default)]
pub struct RowEncodedHashGrouper {
    key_schema: Arc<Schema>,
    table: HashTable<Group>,
    key_data: Vec<u8>,

    // Used for computing canonical hashes.
    random_state: PlRandomState, 

    // Internal random seed used to keep hash iteration order decorrelated.
    // We simply store a random odd number and multiply the canonical hash by it.
    seed: u64,
}

impl RowEncodedHashGrouper {
    pub fn new(key_schema: Arc<Schema>, random_state: PlRandomState) -> Self {
        Self {
            key_schema,
            random_state,
            seed: rand::random::<u64>() | 1,
            ..Default::default()
        }
    }

    fn insert_key(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        let num_groups = self.table.len();
        let entry = self.table.entry(
            hash.wrapping_mul(self.seed),
            |g| unsafe { hash == g.key_hash && key == g.key(&self.key_data) },
            |g| g.key_hash.wrapping_mul(self.seed),
        );

        match entry {
            Entry::Occupied(e) => e.get().group_idx,
            Entry::Vacant(e) => {
                let group_idx: IdxSize = num_groups.try_into().unwrap();
                let group = Group {
                    key_hash: hash,
                    key_offset: self.key_data.len(),
                    key_length: key.len().try_into().unwrap(),
                    group_idx,
                };
                self.key_data.extend(key);
                e.insert(group);
                group_idx
            },
        }
    }

    /// Insert a key, without checking that it is unique.
    fn insert_key_unique(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        let group_idx = self.table.len().try_into().unwrap();
        let group = Group {
            key_hash: hash,
            key_offset: self.key_data.len(),
            key_length: key.len().try_into().unwrap(),
            group_idx,
        };
        self.key_data.extend(key);
        self.table.insert_unique(hash.wrapping_mul(self.seed), group, |g| g.key_hash.wrapping_mul(self.seed));
        group_idx
    }

    fn finalize_keys(&self, mut key_rows: Vec<&[u8]>) -> DataFrame {
        let key_dtypes = self
            .key_schema
            .iter()
            .map(|(_name, dt)| dt.to_physical().to_arrow(CompatLevel::newest()))
            .collect::<Vec<_>>();
        let fields = vec![EncodingField::new_unsorted(); key_dtypes.len()];
        let key_columns =
            unsafe { polars_row::decode::decode_rows(&mut key_rows, &fields, &key_dtypes) };

        let cols = self
            .key_schema
            .iter()
            .zip(key_columns)
            .map(|((name, dt), col)| {
                let s = Series::try_from((name.clone(), col)).unwrap();
                unsafe { s.to_logical_repr_unchecked(dt) }
                    .unwrap()
                    .into_column()
            })
            .collect();
        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }
}

impl Grouper for RowEncodedHashGrouper {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new(
            self.key_schema.clone(),
            self.random_state.clone(),
        ))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn insert_keys(&mut self, keys: &DataFrame, group_idxs: &mut Vec<IdxSize>) {
        let series = keys
            .get_columns()
            .iter()
            .map(|c| c.as_materialized_series().clone())
            .collect_vec();
        let keys_encoded = _get_rows_encoded_unordered(&series[..])
            .unwrap()
            .into_array();
        assert!(keys_encoded.len() == keys[0].len());

        group_idxs.clear();
        group_idxs.reserve(keys_encoded.len());
        for key in keys_encoded.values_iter() {
            let hash = self.random_state.hash_one(key);
            unsafe {
                group_idxs.push_unchecked(self.insert_key(hash, key));
            }
        }
    }

    fn combine(&mut self, other: &dyn Grouper, group_idxs: &mut Vec<IdxSize>) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        // TODO: cardinality estimation.
        self.table.reserve(other.table.len(), |g| g.key_hash.wrapping_mul(self.seed));

        unsafe {
            group_idxs.clear();
            group_idxs.reserve(other.table.len());
            let idx_out = group_idxs.spare_capacity_mut();
            for group in other.table.iter() {
                let group_key = group.key(&other.key_data);
                let new_idx = self.insert_key(group.key_hash, group_key);
                *idx_out.get_unchecked_mut(group.group_idx as usize) = MaybeUninit::new(new_idx);
            }
            group_idxs.set_len(other.table.len());
        }
    }

    fn get_keys_in_group_order(&self) -> DataFrame {
        let mut key_rows: Vec<&[u8]> = Vec::with_capacity(self.table.len());
        unsafe {
            let out = key_rows.spare_capacity_mut();
            for group in &self.table {
                *out.get_unchecked_mut(group.group_idx as usize) =
                    MaybeUninit::new(group.key(&self.key_data));
            }
            key_rows.set_len(self.table.len());
        }
        self.finalize_keys(key_rows)
    }

    fn get_keys_groups(&self, group_idxs: &mut Vec<IdxSize>) -> DataFrame {
        group_idxs.clear();
        group_idxs.reserve(self.table.len());
        self.finalize_keys(
            self.table
                .iter()
                .map(|group| unsafe {
                    group_idxs.push(group.group_idx);
                    group.key(&self.key_data)
                })
                .collect(),
        )
    }

    fn partition(
        &self,
        seed: u64,
        num_partitions: usize,
        partition_idxs: &mut Vec<IdxSize>,
        group_idxs: &mut Vec<IdxSize>,
    ) -> Vec<Box<dyn Grouper>> {
        assert!(num_partitions > 0);

        // Two-pass algorithm to prevent reallocations.
        let mut partition_size = vec![(0, 0); num_partitions]; // (keys, bytes)
        unsafe {
            for group in self.table.iter() {
                let ph = folded_multiply(group.key_hash, seed | 1);
                let p_idx = hash_to_partition(ph, num_partitions);
                let (p_keys, p_bytes) = partition_size.get_unchecked_mut(p_idx as usize);
                *p_keys += 1;
                *p_bytes += group.key_length as usize;
            }
        }

        let mut rng = rand::thread_rng();
        let mut partitions = partition_size
            .into_iter()
            .map(|(keys, bytes)| Self {
                key_schema: self.key_schema.clone(),
                table: HashTable::with_capacity(keys),
                key_data: Vec::with_capacity(bytes),
                random_state: self.random_state.clone(),
                seed: rng.gen::<u64>() | 1,
            })
            .collect_vec();

        unsafe {
            partition_idxs.clear();
            group_idxs.clear();
            partition_idxs.reserve(self.table.len());
            group_idxs.reserve(self.table.len());
            let partition_idxs_out = partition_idxs.spare_capacity_mut();
            let group_idxs_out = group_idxs.spare_capacity_mut();
            for group in self.table.iter() {
                let ph = folded_multiply(group.key_hash, seed | 1);
                let p_idx = hash_to_partition(ph, num_partitions);
                let p = partitions.get_unchecked_mut(p_idx);
                let group_idx = p.insert_key_unique(group.key_hash, group.key(&self.key_data));
                *partition_idxs_out.get_unchecked_mut(group.group_idx as usize) = MaybeUninit::new(p_idx as IdxSize);
                *group_idxs_out.get_unchecked_mut(group.group_idx as usize) = MaybeUninit::new(group_idx);
            }
            partition_idxs.set_len(self.table.len());
            group_idxs.set_len(self.table.len());
        }
        
        partitions.into_iter().map(|p| Box::new(p) as _).collect()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
