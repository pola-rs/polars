use arrow::array::View;
use hashbrown::hash_table::{
    Entry as TEntry, HashTable, OccupiedEntry as TOccupiedEntry, VacantEntry as TVacantEntry,
};
use polars_utils::IdxSize;

const BASE_KEY_BUFFER_CAPACITY: usize = 1024;

struct Key {
    hash: u64,
    view: View,
}

/// An IndexMap where the keys are [u8] slices or `View`s which are pre-hashed.
/// Does not support deletion.
pub struct BinaryViewIndexMap<V> {
    table: HashTable<IdxSize>,
    tuples: Vec<(Key, V)>,
    buffers: Vec<Vec<u8>>,

    // Internal random seed used to keep hash iteration order decorrelated.
    // We simply store a random odd number and multiply the canonical hash by it.
    seed: u64,
}

impl<V> Default for BinaryViewIndexMap<V> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            tuples: Vec::new(),
            buffers: vec![],
            seed: rand::random::<u64>() | 1,
        }
    }
}

impl<V> BinaryViewIndexMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional, |i| unsafe {
            let tuple = self.tuples.get_unchecked(*i as usize);
            tuple.0.hash.wrapping_mul(self.seed)
        });
        self.tuples.reserve(additional);
    }

    pub fn len(&self) -> IdxSize {
        self.tuples.len() as IdxSize
    }

    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    pub fn buffers(&self) -> &[Vec<u8>] {
        &self.buffers
    }

    #[inline]
    pub fn get(&self, hash: u64, key: &[u8]) -> Option<&V> {
        unsafe {
            if key.len() <= View::MAX_INLINE_SIZE as usize {
                self.get_inline_view(hash, &View::new_inline_unchecked(key))
            } else {
                self.get_long_key(hash, key)
            }
        }
    }

    /// # Safety
    /// The view must be valid in combination with the given buffers.
    #[inline]
    pub unsafe fn get_view<B: AsRef<[u8]>>(
        &self,
        hash: u64,
        key: &View,
        buffers: &[B],
    ) -> Option<&V> {
        unsafe {
            if key.length <= View::MAX_INLINE_SIZE {
                self.get_inline_view(hash, key)
            } else {
                self.get_long_key(hash, key.get_external_slice_unchecked(buffers))
            }
        }
    }

    /// # Safety
    /// The view must be inlined.
    pub unsafe fn get_inline_view(&self, hash: u64, key: &View) -> Option<&V> {
        unsafe {
            debug_assert!(key.length <= View::MAX_INLINE_SIZE);
            let idx = self.table.find(hash.wrapping_mul(self.seed), |i| {
                let t = self.tuples.get_unchecked(*i as usize);
                *key == t.0.view
            })?;
            Some(&self.tuples.get_unchecked(*idx as usize).1)
        }
    }

    /// # Safety
    /// key.len() > View::MAX_INLINE_SIZE
    pub unsafe fn get_long_key(&self, hash: u64, key: &[u8]) -> Option<&V> {
        unsafe {
            debug_assert!(key.len() > View::MAX_INLINE_SIZE as usize);
            let idx = self.table.find(hash.wrapping_mul(self.seed), |i| {
                let t = self.tuples.get_unchecked(*i as usize);
                hash == t.0.hash
                    && key.len() == t.0.view.length as usize
                    && key == t.0.view.get_external_slice_unchecked(&self.buffers)
            })?;
            Some(&self.tuples.get_unchecked(*idx as usize).1)
        }
    }

    #[inline]
    pub fn entry<'k>(&mut self, hash: u64, key: &'k [u8]) -> Entry<'_, 'k, V> {
        unsafe {
            if key.len() <= View::MAX_INLINE_SIZE as usize {
                self.entry_inline_view(hash, View::new_inline_unchecked(key))
            } else {
                self.entry_long_key(hash, key)
            }
        }
    }

    /// # Safety
    /// The view must be valid in combination with the given buffers.
    #[inline]
    pub unsafe fn entry_view<'k, B: AsRef<[u8]>>(
        &mut self,
        hash: u64,
        key: View,
        buffers: &'k [B],
    ) -> Entry<'_, 'k, V> {
        unsafe {
            if key.length <= View::MAX_INLINE_SIZE {
                self.entry_inline_view(hash, key)
            } else {
                self.entry_long_key(hash, key.get_external_slice_unchecked(buffers))
            }
        }
    }

    /// # Safety
    /// The view must be inlined.
    pub unsafe fn entry_inline_view<'k>(&mut self, hash: u64, key: View) -> Entry<'_, 'k, V> {
        debug_assert!(key.length <= View::MAX_INLINE_SIZE);
        let entry = self.table.entry(
            hash.wrapping_mul(self.seed),
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                key == t.0.view
            },
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                t.0.hash.wrapping_mul(self.seed)
            },
        );

        match entry {
            TEntry::Occupied(o) => Entry::Occupied(OccupiedEntry {
                entry: o,
                tuples: &mut self.tuples,
            }),
            TEntry::Vacant(v) => Entry::Vacant(VacantEntry {
                view: key,
                external: None,
                hash,
                entry: v,
                tuples: &mut self.tuples,
                buffers: &mut self.buffers,
            }),
        }
    }

    /// # Safety
    /// key.len() > View::MAX_INLINE_SIZE
    pub unsafe fn entry_long_key<'k>(&mut self, hash: u64, key: &'k [u8]) -> Entry<'_, 'k, V> {
        debug_assert!(key.len() > View::MAX_INLINE_SIZE as usize);
        let entry = self.table.entry(
            hash.wrapping_mul(self.seed),
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                hash == t.0.hash
                    && key.len() == t.0.view.length as usize
                    && key == t.0.view.get_external_slice_unchecked(&self.buffers)
            },
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                t.0.hash.wrapping_mul(self.seed)
            },
        );

        match entry {
            TEntry::Occupied(o) => Entry::Occupied(OccupiedEntry {
                entry: o,
                tuples: &mut self.tuples,
            }),
            TEntry::Vacant(v) => Entry::Vacant(VacantEntry {
                view: View::default(),
                external: Some(key),
                hash,
                entry: v,
                tuples: &mut self.tuples,
                buffers: &mut self.buffers,
            }),
        }
    }

    /// Insert an empty entry which will never be mapped to. Returns the index of the entry.
    ///
    /// This is useful for entries which are handled externally.
    pub fn push_unmapped_empty_entry(&mut self, value: V) -> IdxSize {
        let ret = self.tuples.len() as IdxSize;
        let key = Key {
            hash: 0,
            view: View::default(),
        };
        self.tuples.push((key, value));
        ret
    }

    /// Gets the hash, key and value at the given index by insertion order.
    #[inline(always)]
    pub fn get_index(&self, idx: IdxSize) -> Option<(u64, &[u8], &V)> {
        let t = self.tuples.get(idx as usize)?;
        Some((
            t.0.hash,
            unsafe { t.0.view.get_slice_unchecked(&self.buffers) },
            &t.1,
        ))
    }

    /// Gets the hash, key and value at the given index by insertion order.
    ///
    /// # Safety
    /// The index must be less than len().
    #[inline(always)]
    pub unsafe fn get_index_unchecked(&self, idx: IdxSize) -> (u64, &[u8], &V) {
        let t = unsafe { self.tuples.get_unchecked(idx as usize) };
        unsafe { (t.0.hash, t.0.view.get_slice_unchecked(&self.buffers), &t.1) }
    }

    /// Gets the hash, view and value at the given index by insertion order.
    ///
    /// # Safety
    /// The index must be less than len().
    #[inline(always)]
    pub unsafe fn get_index_view_unchecked(&self, idx: IdxSize) -> (u64, View, &V) {
        let t = unsafe { self.tuples.get_unchecked(idx as usize) };
        (t.0.hash, t.0.view, &t.1)
    }

    /// Iterates over the (hash, key) pairs in insertion order.
    pub fn iter_hash_keys(&self) -> impl Iterator<Item = (u64, &[u8])> {
        self.tuples
            .iter()
            .map(|t| unsafe { (t.0.hash, t.0.view.get_slice_unchecked(&self.buffers)) })
    }

    /// Iterates over the (hash, key_view) pairs in insertion order.
    pub fn iter_hash_views(&self) -> impl Iterator<Item = (u64, View)> {
        self.tuples.iter().map(|t| (t.0.hash, t.0.view))
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
    #[inline]
    pub fn index(&self) -> IdxSize {
        *self.entry.get()
    }

    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        let idx = self.index();
        unsafe { &mut self.tuples.get_unchecked_mut(idx as usize).1 }
    }
}

pub struct VacantEntry<'a, 'k, V> {
    hash: u64,
    view: View,                 // Empty when key is not inlined.
    external: Option<&'k [u8]>, // Only set when not inlined.
    entry: TVacantEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(Key, V)>,
    buffers: &'a mut Vec<Vec<u8>>,
}

#[allow(clippy::needless_lifetimes)]
impl<'a, 'k, V> VacantEntry<'a, 'k, V> {
    #[inline]
    pub fn index(&self) -> IdxSize {
        self.tuples.len() as IdxSize
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        unsafe {
            let tuple_idx: IdxSize = self.tuples.len().try_into().unwrap();
            let view = if let Some(key) = self.external {
                if self
                    .buffers
                    .last()
                    .is_none_or(|buf| buf.len() + key.len() > buf.capacity())
                {
                    let ideal_next_cap = BASE_KEY_BUFFER_CAPACITY
                        .checked_shl(self.buffers.len() as u32)
                        .unwrap();
                    let next_capacity = std::cmp::max(ideal_next_cap, key.len());
                    self.buffers.push(Vec::with_capacity(next_capacity));
                }
                let buffer_idx = (self.buffers.len() - 1) as u32;
                let active_buf = self.buffers.last_mut().unwrap_unchecked();
                let offset = active_buf.len() as u32;
                active_buf.extend_from_slice(key);
                View::new_from_bytes(key, buffer_idx, offset)
            } else {
                self.view
            };
            let tuple_key = Key {
                hash: self.hash,
                view,
            };
            self.tuples.push((tuple_key, value));
            self.entry.insert(tuple_idx);
            &mut self.tuples.last_mut().unwrap_unchecked().1
        }
    }
}
