use hashbrown::hash_table::{Entry as TEntry, HashTable, OccupiedEntry as TOccupiedEntry, VacantEntry as TVacantEntry};
use polars_utils::IdxSize;

const BASE_KEY_DATA_CAPACITY: usize = 1024;

struct Key {
    key_hash: u64,
    key_buffer: u32,
    key_offset: usize,
    key_length: u32,
}

impl Key {
    unsafe fn get<'k>(&self, key_data: &'k [Vec<u8>]) -> &'k [u8] {
        let buf = key_data.get_unchecked(self.key_buffer as usize);
        buf.get_unchecked(self.key_offset..self.key_offset + self.key_length as usize)
    }
}


/// An IndexMap where the keys are always [u8] slices which are pre-hashed.
pub struct BytesIndexMap<V> {
    table: HashTable<IdxSize>,
    tuples: Vec<(Key, V)>,
    key_data: Vec<Vec<u8>>,

    // Internal random seed used to keep hash iteration order decorrelated.
    // We simply store a random odd number and multiply the canonical hash by it.
    seed: u64,
}

impl<V> Default for BytesIndexMap<V> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            tuples: Vec::new(),
            key_data: vec![Vec::with_capacity(BASE_KEY_DATA_CAPACITY)],
            seed: rand::random::<u64>() | 1,
        }
    }
}

impl<V> BytesIndexMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional, |i| unsafe {
            let tuple = self.tuples.get_unchecked(*i as usize);
            tuple.0.key_hash.wrapping_mul(self.seed)
        });
        self.tuples.reserve(additional);
    }

    pub fn entry<'k>(&mut self, key: &'k [u8], hash: u64) -> Entry<'_, 'k, V> {
        let entry = self.table.entry(
            hash.wrapping_mul(self.seed),
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                hash == t.0.key_hash && key == t.0.get(&self.key_data)
            },
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                t.0.key_hash.wrapping_mul(self.seed)
            },
        );
        
        match entry {
            TEntry::Occupied(o) => Entry::Occupied(OccupiedEntry {
                entry: o,
                tuples: &mut self.tuples,
                key_data: &mut self.key_data,
            }),
            TEntry::Vacant(v) => Entry::Vacant(VacantEntry {
                key,
                hash,
                entry: v,
                tuples: &mut self.tuples,
                key_data: &mut self.key_data,
            }),
        }
    }
}

pub enum Entry<'a, 'k, V> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, 'k, V>),
}

pub struct OccupiedEntry<'a, V> {
    entry: TOccupiedEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(Key, V)>,
    key_data: &'a mut Vec<Vec<u8>>,
}

impl<'a, V> OccupiedEntry<'a,  V> {
    pub fn index(&self) -> IdxSize {
        *self.entry.get()
    }
}

pub struct VacantEntry<'a, 'k, V> {
    key: &'k [u8],
    hash: u64,
    entry: TVacantEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(Key, V)>,
    key_data: &'a mut Vec<Vec<u8>>,
}

impl<'a, 'k, V> VacantEntry<'a, 'k, V> {
    pub fn index(&self) -> IdxSize {
        self.tuples.len() as IdxSize
    }
    
    pub fn insert(self, value: V) -> &'a mut V {
        unsafe {
            let tuple_idx: IdxSize = self.tuples.len().try_into().unwrap();

            let mut num_buffers = self.key_data.len() as u32;
            let mut active_buf = self.key_data.last_mut().unwrap_unchecked();
            let key_len = self.key.len();
            if active_buf.len() + key_len > active_buf.capacity() {
                let ideal_next_cap = BASE_KEY_DATA_CAPACITY.checked_shl(num_buffers).unwrap();
                let next_capacity = std::cmp::max(ideal_next_cap, key_len);
                self.key_data.push(Vec::with_capacity(next_capacity));
                active_buf = self.key_data.last_mut().unwrap_unchecked();
                num_buffers += 1;
            }

            let tuple_key = Key {
                key_hash: self.hash,
                key_buffer: num_buffers - 1,
                key_offset: active_buf.len(),
                key_length: self.key.len().try_into().unwrap(),
            };
            self.tuples.push((tuple_key, value));
            active_buf.extend_from_slice(self.key);
            self.entry.insert(tuple_idx);
            &mut self.tuples.last_mut().unwrap_unchecked().1
        }
    }
}


/*
use hashbrown::hash_table::{Entry, HashTable};
use polars_row::EncodingField;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::vec::PushUnchecked;

use super::*;
use crate::hash_keys::HashKeys;

const BASE_KEY_DATA_CAPACITY: usize = 1024;

struct Key {
    key_hash: u64,
    key_buffer: u32,
    key_offset: usize,
    key_length: u32,
}

impl Key {
    unsafe fn get<'k>(&self, key_data: &'k [Vec<u8>]) -> &'k [u8] {
        let buf = key_data.get_unchecked(self.key_buffer as usize);
        buf.get_unchecked(self.key_offset..self.key_offset + self.key_length as usize)
    }
}

#[derive(Default)]
pub struct RowEncodedHashtupleer {
    key_schema: Arc<Schema>,
    table: HashTable<IdxSize>,
    group_keys: Vec<Key>,
    key_data: Vec<Vec<u8>>,

    // Internal random seed used to keep hash iteration order decorrelated.
    // We simply store a random odd number and multiply the canonical hash by it.
    seed: u64,
}

impl RowEncodedHashGrouper {
    pub fn new(key_schema: Arc<Schema>) -> Self {
        Self {
            key_schema,
            seed: rand::random::<u64>() | 1,
            key_data: vec![Vec::with_capacity(BASE_KEY_DATA_CAPACITY)],
            ..Default::default()
        }
    }

    fn insert_key(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        let entry = self.table.entry(
            hash.wrapping_mul(self.seed),
            |g| unsafe {
                let gk = self.group_keys.get_unchecked(*g as usize);
                hash == gk.key_hash && key == gk.get(&self.key_data)
            },
            |g| unsafe {
                let gk = self.group_keys.get_unchecked(*g as usize);
                gk.key_hash.wrapping_mul(self.seed)
            },
        );

        match entry {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => unsafe {
                let mut num_buffers = self.key_data.len() as u32;
                let mut active_buf = self.key_data.last_mut().unwrap_unchecked();
                let key_len = key.len();
                if active_buf.len() + key_len > active_buf.capacity() {
                    let ideal_next_cap = BASE_KEY_DATA_CAPACITY.checked_shl(num_buffers).unwrap();
                    let next_capacity = std::cmp::max(ideal_next_cap, key_len);
                    self.key_data.push(Vec::with_capacity(next_capacity));
                    active_buf = self.key_data.last_mut().unwrap_unchecked();
                    num_buffers += 1;
                }

                let group_idx: IdxSize = self.group_keys.len().try_into().unwrap();
                let group_key = Key {
                    key_hash: hash,
                    key_buffer: num_buffers - 1,
                    key_offset: active_buf.len(),
                    key_length: key.len().try_into().unwrap(),
                };
                self.group_keys.push(group_key);
                active_buf.extend_from_slice(key);
                e.insert(group_idx);
                group_idx
            },
        }
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
        Box::new(Self::new(self.key_schema.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional, |g| unsafe {
            let gk = self.group_keys.get_unchecked(*g as usize);
            gk.key_hash.wrapping_mul(self.seed)
        });
        self.group_keys.reserve(additional);
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn insert_keys(&mut self, keys: HashKeys, group_idxs: &mut Vec<IdxSize>) {
        let HashKeys::RowEncoded(keys) = keys else {
            unreachable!()
        };
        group_idxs.clear();
        group_idxs.reserve(keys.hashes.len());
        for (hash, key) in keys.hashes.iter().zip(keys.keys.values_iter()) {
            unsafe {
                group_idxs.push_unchecked(self.insert_key(*hash, key));
            }
        }
    }

    fn combine(&mut self, other: &dyn Grouper, group_idxs: &mut Vec<IdxSize>) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        // TODO: cardinality estimation.
        self.table.reserve(other.group_keys.len(), |g| unsafe {
            let gk = self.group_keys.get_unchecked(*g as usize);
            gk.key_hash.wrapping_mul(self.seed)
        });

        unsafe {
            group_idxs.clear();
            group_idxs.reserve(other.table.len());
            for group_key in &other.group_keys {
                let new_idx = self.insert_key(group_key.key_hash, group_key.get(&other.key_data));
                group_idxs.push_unchecked(new_idx);
            }
        }
    }

    unsafe fn gather_combine(
        &mut self,
        other: &dyn Grouper,
        subset: &[IdxSize],
        group_idxs: &mut Vec<IdxSize>,
    ) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        // TODO: cardinality estimation.
        self.table.reserve(subset.len(), |g| unsafe {
            let gk = self.group_keys.get_unchecked(*g as usize);
            gk.key_hash.wrapping_mul(self.seed)
        });
        self.group_keys.reserve(subset.len());

        unsafe {
            group_idxs.clear();
            group_idxs.reserve(subset.len());
            for i in subset {
                let group_key = other.group_keys.get_unchecked(*i as usize);
                let new_idx = self.insert_key(group_key.key_hash, group_key.get(&other.key_data));
                group_idxs.push_unchecked(new_idx);
            }
        }
    }

    fn get_keys_in_group_order(&self) -> DataFrame {
        let mut key_rows: Vec<&[u8]> = Vec::with_capacity(self.table.len());
        unsafe {
            for group_key in &self.group_keys {
                key_rows.push_unchecked(group_key.get(&self.key_data));
            }
        }
        self.finalize_keys(key_rows)
    }

    fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
    ) {
        let num_partitions = partitioner.num_partitions();
        assert!(partition_idxs.len() == num_partitions);
        assert!(sketches.len() == num_partitions);

        // Two-pass algorithm to prevent reallocations.
        let mut partition_sizes = vec![0; num_partitions];
        unsafe {
            for group_key in &self.group_keys {
                let p_idx = partitioner.hash_to_partition(group_key.key_hash);
                *partition_sizes.get_unchecked_mut(p_idx) += 1;
                sketches.get_unchecked_mut(p_idx).insert(group_key.key_hash);
            }
        }

        for (partition, sz) in partition_idxs.iter_mut().zip(partition_sizes) {
            partition.clear();
            partition.reserve(sz);
        }

        unsafe {
            for (i, group_key) in self.group_keys.iter().enumerate() {
                let p_idx = partitioner.hash_to_partition(group_key.key_hash);
                let p = partition_idxs.get_unchecked_mut(p_idx);
                p.push_unchecked(i as IdxSize);
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
*/