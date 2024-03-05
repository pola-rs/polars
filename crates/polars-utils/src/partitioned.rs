use hashbrown::hash_map::{HashMap, RawEntryBuilder, RawEntryBuilderMut};

use crate::hashing::hash_to_partition;
use crate::slice::GetSaferUnchecked;

pub struct PartitionedHashMap<K, V, S = ahash::RandomState> {
    inner: Vec<HashMap<K, V, S>>,
}

impl<K, V, S> PartitionedHashMap<K, V, S> {
    pub fn new(inner: Vec<HashMap<K, V, S>>) -> Self {
        Self { inner }
    }

    #[inline]
    pub fn raw_entry_mut(&mut self, h: u64) -> RawEntryBuilderMut<'_, K, V, S> {
        let partition = hash_to_partition(h, self.inner.len());
        let current_table = unsafe { self.inner.get_unchecked_release_mut(partition) };
        current_table.raw_entry_mut()
    }

    #[inline]
    pub fn raw_entry(&self, h: u64) -> RawEntryBuilder<'_, K, V, S> {
        let partition = hash_to_partition(h, self.inner.len());
        let current_table = unsafe { self.inner.get_unchecked_release(partition) };
        current_table.raw_entry()
    }

    pub fn inner(&self) -> &[HashMap<K, V, S>] {
        self.inner.as_ref()
    }

    pub fn inner_mut(&mut self) -> &mut Vec<HashMap<K, V, S>> {
        &mut self.inner
    }
}
