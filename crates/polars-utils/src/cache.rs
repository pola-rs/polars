use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

use foldhash::fast::RandomState;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use slotmap::{Key, SlotMap, new_key_type};

/// A cached function that use `LruCache`.
pub struct LruCachedFunc<T, R, F> {
    func: F,
    cache: LruCache<T, R>,
}

impl<T, R, F> LruCachedFunc<T, R, F>
where
    F: FnMut(T) -> R,
    T: std::hash::Hash + Eq + Clone,
    R: Copy,
{
    pub fn new(func: F, size: usize) -> Self {
        Self {
            func,
            cache: LruCache::with_capacity(size.max(1)),
        }
    }

    pub fn eval(&mut self, x: T, use_cache: bool) -> R {
        if use_cache {
            *self
                .cache
                .get_or_insert_with(&x, |xr| (self.func)(xr.clone()))
        } else {
            (self.func)(x)
        }
    }
}

new_key_type! {
    struct LruKey;
}

pub struct LruCache<K, V, S = RandomState> {
    table: HashTable<LruKey>,
    elements: SlotMap<LruKey, LruEntry<K, V>>,
    max_capacity: usize,
    most_recent: LruKey,
    least_recent: LruKey,
    build_hasher: S,
}

struct LruEntry<K, V> {
    key: K,
    value: V,
    list: LruListNode,
}

#[derive(Copy, Clone, Default)]
struct LruListNode {
    more_recent: LruKey,
    less_recent: LruKey,
}

impl<K, V> LruCache<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::default())
    }
}

impl<K, V, S> LruCache<K, V, S> {
    pub fn with_capacity_and_hasher(max_capacity: usize, build_hasher: S) -> Self {
        assert!(max_capacity > 0);
        Self {
            // Allocate one more capacity to prevent double-lookup or realloc
            // when doing get_or_insert when full.
            table: HashTable::with_capacity(max_capacity + 1),
            elements: SlotMap::with_capacity_and_key(max_capacity + 1),
            max_capacity,
            most_recent: LruKey::null(),
            least_recent: LruKey::null(),
            build_hasher,
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> LruCache<K, V, S> {
    fn lru_list_unlink(&mut self, lru_key: LruKey) {
        let list = self.elements[lru_key].list;
        if let Some(more_recent) = self.elements.get_mut(list.more_recent) {
            more_recent.list.less_recent = list.less_recent;
        } else {
            self.most_recent = list.less_recent;
        }
        if let Some(less_recent) = self.elements.get_mut(list.less_recent) {
            less_recent.list.more_recent = list.more_recent;
        } else {
            self.least_recent = list.more_recent;
        }
    }

    fn lru_list_insert_mru(&mut self, lru_key: LruKey) {
        let prev_most_recent_key = self.most_recent;
        self.most_recent = lru_key;
        if let Some(prev_most_recent) = self.elements.get_mut(prev_most_recent_key) {
            prev_most_recent.list.more_recent = lru_key;
        } else {
            self.least_recent = lru_key;
        }
        let list = &mut self.elements[lru_key].list;
        list.more_recent = LruKey::null();
        list.less_recent = prev_most_recent_key;
    }

    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        if self.elements.is_empty() {
            return None;
        }
        let lru_key = self.least_recent;
        let hash = self.build_hasher.hash_one(&self.elements[lru_key].key);
        self.lru_list_unlink(lru_key);
        let lru_entry = self.elements.remove(lru_key).unwrap();
        self.table
            .find_entry(hash, |k| *k == lru_key)
            .unwrap()
            .remove();
        Some((lru_entry.key, lru_entry.value))
    }

    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.build_hasher.hash_one(key);
        let lru_key = *self
            .table
            .find(hash, |lru_key| self.elements[*lru_key].key.borrow() == key)?;
        self.lru_list_unlink(lru_key);
        self.lru_list_insert_mru(lru_key);
        let lru_node = self.elements.get(lru_key).unwrap();
        Some(&lru_node.value)
    }

    /// Returns the old value, if any.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.build_hasher.hash_one(&key);
        match self.table.entry(
            hash,
            |lru_key| self.elements[*lru_key].key == key,
            |lru_key| self.build_hasher.hash_one(&self.elements[*lru_key].key),
        ) {
            Entry::Occupied(o) => {
                let lru_key = *o.get();
                self.lru_list_unlink(lru_key);
                self.lru_list_insert_mru(lru_key);
                Some(core::mem::replace(&mut self.elements[lru_key].value, value))
            },

            Entry::Vacant(v) => {
                let lru_entry = LruEntry {
                    key,
                    value,
                    list: LruListNode::default(),
                };
                let lru_key = self.elements.insert(lru_entry);
                v.insert(lru_key);
                self.lru_list_insert_mru(lru_key);
                if self.elements.len() > self.max_capacity {
                    self.pop_lru();
                }
                None
            },
        }
    }

    pub fn get_or_insert_with<Q, F: FnOnce(&Q) -> V>(&mut self, key: &Q, f: F) -> &mut V
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
    {
        enum Never {}
        let Ok(ret) = self.try_get_or_insert_with::<Q, Never, _>(key, |k| Ok(f(k)));
        ret
    }

    pub fn try_get_or_insert_with<Q, E, F: FnOnce(&Q) -> Result<V, E>>(
        &mut self,
        key: &Q,
        f: F,
    ) -> Result<&mut V, E>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
    {
        let hash = self.build_hasher.hash_one(key);
        match self.table.entry(
            hash,
            |lru_key| self.elements[*lru_key].key.borrow() == key,
            |lru_key| self.build_hasher.hash_one(&self.elements[*lru_key].key),
        ) {
            Entry::Occupied(o) => {
                let lru_key = *o.get();
                if lru_key != self.most_recent {
                    self.lru_list_unlink(lru_key);
                    self.lru_list_insert_mru(lru_key);
                }
                Ok(&mut self.elements[lru_key].value)
            },

            Entry::Vacant(v) => {
                let lru_entry = LruEntry {
                    value: f(key)?,
                    key: key.to_owned(),
                    list: LruListNode::default(),
                };
                let lru_key = self.elements.insert(lru_entry);
                v.insert(lru_key);
                self.lru_list_insert_mru(lru_key);
                if self.elements.len() > self.max_capacity {
                    self.pop_lru();
                }
                Ok(&mut self.elements[lru_key].value)
            },
        }
    }
}
