use hashbrown::hash_table::{
    Entry as TEntry, HashTable, OccupiedEntry as TOccupiedEntry, VacantEntry as TVacantEntry,
};

use crate::IdxSize;

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

    pub fn len(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn get(&self, hash: u64, key: &[u8]) -> Option<&V> {
        let idx = self.table.find(hash.wrapping_mul(self.seed), |i| unsafe {
            let t = self.tuples.get_unchecked(*i as usize);
            hash == t.0.key_hash && key == t.0.get(&self.key_data)
        })?;
        unsafe { Some(&self.tuples.get_unchecked(*idx as usize).1) }
    }

    pub fn entry<'k>(&mut self, hash: u64, key: &'k [u8]) -> Entry<'_, 'k, V> {
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

    /// Gets the hash, key and value at the given index by insertion order.
    #[inline(always)]
    pub fn get_index(&self, idx: IdxSize) -> Option<(u64, &[u8], &V)> {
        let t = self.tuples.get(idx as usize)?;
        Some((t.0.key_hash, unsafe { t.0.get(&self.key_data) }, &t.1))
    }

    /// Gets the hash, key and value at the given index by insertion order.
    ///
    /// # Safety
    /// The index must be less than len().
    #[inline(always)]
    pub unsafe fn get_index_unchecked(&self, idx: IdxSize) -> (u64, &[u8], &V) {
        let t = self.tuples.get_unchecked(idx as usize);
        (t.0.key_hash, t.0.get(&self.key_data), &t.1)
    }

    /// Iterates over the (hash, key) pairs in insertion order.
    pub fn iter_hash_keys(&self) -> impl Iterator<Item = (u64, &[u8])> {
        self.tuples
            .iter()
            .map(|t| unsafe { (t.0.key_hash, t.0.get(&self.key_data)) })
    }

    /// Iterates over the values in insertion order.
    pub fn iter_values(&self) -> impl Iterator<Item = &V> {
        self.tuples.iter().map(|t| &t.1)
    }
}

pub enum Entry<'a, 'k, V> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, 'k, V>),
}

pub struct OccupiedEntry<'a, V> {
    entry: TOccupiedEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(Key, V)>,
}

impl<'a, V> OccupiedEntry<'a, V> {
    pub fn index(&self) -> IdxSize {
        *self.entry.get()
    }

    pub fn into_mut(self) -> &'a mut V {
        let idx = self.index();
        unsafe { &mut self.tuples.get_unchecked_mut(idx as usize).1 }
    }
}

pub struct VacantEntry<'a, 'k, V> {
    hash: u64,
    key: &'k [u8],
    entry: TVacantEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(Key, V)>,
    key_data: &'a mut Vec<Vec<u8>>,
}

#[allow(clippy::needless_lifetimes)]
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
