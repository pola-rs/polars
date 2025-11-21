use hashbrown::hash_table::{
    Entry as TEntry, HashTable, OccupiedEntry as TOccupiedEntry, VacantEntry as TVacantEntry,
};

use crate::IdxSize;
use crate::aliases::PlRandomState;
use crate::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};

/// An IndexMap where the keys are hashed and compared with TotalOrd/TotalEq.
pub struct TotalIndexMap<K, V> {
    table: HashTable<IdxSize>,
    tuples: Vec<(K, V)>,
    random_state: PlRandomState,
}

impl<K, V> Default for TotalIndexMap<K, V> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            tuples: Vec::new(),
            random_state: PlRandomState::default(),
        }
    }
}

impl<K: TotalHash + TotalEq, V> TotalIndexMap<K, V> {
    pub fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional, |i| unsafe {
            let tuple = self.tuples.get_unchecked(*i as usize);
            self.random_state.tot_hash_one(&tuple.0)
        });
        self.tuples.reserve(additional);
    }

    pub fn len(&self) -> IdxSize {
        self.tuples.len() as IdxSize
    }

    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = self.random_state.tot_hash_one(key);
        let idx = self.table.find(hash, |i| unsafe {
            let t = self.tuples.get_unchecked(*i as usize);
            hash == self.random_state.tot_hash_one(&t.0) && key.tot_eq(&t.0)
        })?;
        unsafe { Some(&self.tuples.get_unchecked(*idx as usize).1) }
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let hash = self.random_state.tot_hash_one(&key);
        let entry = self.table.entry(
            hash,
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                hash == self.random_state.tot_hash_one(&t.0) && key.tot_eq(&t.0)
            },
            |i| unsafe {
                let t = self.tuples.get_unchecked(*i as usize);
                self.random_state.tot_hash_one(&t.0)
            },
        );

        match entry {
            TEntry::Occupied(o) => Entry::Occupied(OccupiedEntry {
                entry: o,
                tuples: &mut self.tuples,
            }),
            TEntry::Vacant(v) => Entry::Vacant(VacantEntry {
                key,
                entry: v,
                tuples: &mut self.tuples,
            }),
        }
    }

    /// Insert a key which will never be mapped to. Returns the index of the entry.
    ///
    /// This is useful for entries which are handled externally.
    pub fn push_unmapped_entry(&mut self, key: K, value: V) -> IdxSize {
        let ret = self.tuples.len() as IdxSize;
        self.tuples.push((key, value));
        ret
    }

    /// Gets the key and value at the given index by insertion order.
    #[inline(always)]
    pub fn get_index(&self, idx: IdxSize) -> Option<(&K, &V)> {
        let t = self.tuples.get(idx as usize)?;
        Some((&t.0, &t.1))
    }

    /// Gets the key and value at the given index by insertion order.
    ///
    /// # Safety
    /// The index must be less than len().
    #[inline(always)]
    pub unsafe fn get_index_unchecked(&self, idx: IdxSize) -> (&K, &V) {
        let t = unsafe { self.tuples.get_unchecked(idx as usize) };
        (&t.0, &t.1)
    }

    /// Iterates over the keys in insertion order.
    pub fn iter_keys(&self) -> impl Iterator<Item = &K> {
        self.tuples.iter().map(|t| &t.0)
    }

    /// Iterates over the values in insertion order.
    pub fn iter_values(&self) -> impl Iterator<Item = &V> {
        self.tuples.iter().map(|t| &t.1)
    }
}

pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K, V> {
    entry: TOccupiedEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(K, V)>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub fn index(&self) -> IdxSize {
        *self.entry.get()
    }

    pub fn into_mut(self) -> &'a mut V {
        let idx = self.index();
        unsafe { &mut self.tuples.get_unchecked_mut(idx as usize).1 }
    }
}

pub struct VacantEntry<'a, K, V> {
    key: K,
    entry: TVacantEntry<'a, IdxSize>,
    tuples: &'a mut Vec<(K, V)>,
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    pub fn index(&self) -> IdxSize {
        self.tuples.len() as IdxSize
    }

    pub fn insert(self, value: V) -> &'a mut V {
        unsafe {
            let tuple_idx: IdxSize = self.tuples.len().try_into().unwrap();
            self.tuples.push((self.key, value));
            self.entry.insert(tuple_idx);
            &mut self.tuples.last_mut().unwrap_unchecked().1
        }
    }
}
