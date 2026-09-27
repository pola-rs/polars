use std::sync::Arc;

use hashbrown::hash_table::{Entry as TEntry, HashTable};
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_utils::IdxSize;

use super::keys::KeyRowKeys;
use super::layout::KeyRowLayout;
use super::{VERIFY_BATCH_SIZE, push_long_bytes};

/// One batch of a batched lookup. Keys are first matched by hash alone: a key whose
/// hash is stored becomes a candidate for that stored key and, when inserting, a key
/// whose hash is not stored is inserted provisionally. The rows of the new keys are
/// then written and all candidates verified, column by column.
///
/// Positions are within the batch, key indices are into the incoming keys and map
/// indices are the insertion-order indices of stored keys.
struct LookupBatch {
    cand_pos: Vec<usize>,
    cand_key_idxs: Vec<IdxSize>,
    cand_map_idxs: Vec<IdxSize>,
    cand_rows: Vec<*const u64>,
    /// Per candidate whether it is the same key as its stored key.
    cand_ok: Vec<bool>,
    new_pos: Vec<usize>,
    new_key_idxs: Vec<IdxSize>,
    new_map_idxs: Vec<IdxSize>,
    new_rows: Vec<*mut u64>,
}

impl Default for LookupBatch {
    fn default() -> Self {
        let n = VERIFY_BATCH_SIZE;
        Self {
            cand_pos: Vec::with_capacity(n),
            cand_key_idxs: Vec::with_capacity(n),
            cand_map_idxs: Vec::with_capacity(n),
            cand_rows: Vec::with_capacity(n),
            cand_ok: Vec::with_capacity(n),
            new_pos: Vec::with_capacity(n),
            new_key_idxs: Vec::with_capacity(n),
            new_map_idxs: Vec::with_capacity(n),
            new_rows: Vec::with_capacity(n),
        }
    }
}

impl LookupBatch {
    fn clear(&mut self) {
        self.cand_pos.clear();
        self.cand_key_idxs.clear();
        self.cand_map_idxs.clear();
        self.new_pos.clear();
        self.new_key_idxs.clear();
        self.new_map_idxs.clear();
    }

    #[inline(always)]
    fn push_candidate(&mut self, pos: usize, key_idx: IdxSize, map_idx: IdxSize) {
        self.cand_pos.push(pos);
        self.cand_key_idxs.push(key_idx);
        self.cand_map_idxs.push(map_idx);
    }

    #[inline(always)]
    fn push_new(&mut self, pos: usize, key_idx: IdxSize, map_idx: IdxSize) {
        self.new_pos.push(pos);
        self.new_key_idxs.push(key_idx);
        self.new_map_idxs.push(map_idx);
    }

    /// Writes the rows of the new keys into their zeroed entries.
    ///
    /// # Safety
    /// The new keys must be in-bounds for `keys`, and their entries for `entries`.
    /// The row pointers are taken from `entries` after all entries of the batch were
    /// pushed and are only used during this call. Storing long bytes grows `buffers`,
    /// a separate allocation, so the rows cannot move while they are written.
    unsafe fn write_new(
        &mut self,
        keys: &KeyRowKeys,
        entries: &mut [u64],
        entry_words: usize,
        buffers: &mut Vec<Vec<u8>>,
    ) {
        let base = entries.as_mut_ptr();
        self.new_rows.clear();
        self.new_rows.extend(
            self.new_map_idxs
                .iter()
                .map(|j| base.add(*j as usize * entry_words + 1)),
        );
        keys.write_rows(&self.new_key_idxs, &self.new_rows, |bytes| {
            push_long_bytes(buffers, bytes)
        });
    }

    /// Verifies every candidate against its stored key.
    ///
    /// # Safety
    /// The candidates must be in-bounds for `keys`, and their entries for `entries`.
    /// The row pointers are only used during this call.
    unsafe fn verify(
        &mut self,
        keys: &KeyRowKeys,
        entries: &[u64],
        entry_words: usize,
        buffers: &[Vec<u8>],
    ) {
        self.cand_rows.clear();
        self.cand_rows.extend(
            self.cand_map_idxs
                .iter()
                .map(|j| entries.as_ptr().add(*j as usize * entry_words + 1)),
        );
        self.cand_ok.clear();
        self.cand_ok.resize(self.cand_key_idxs.len(), true);
        keys.verify(
            &self.cand_key_idxs,
            &self.cand_rows,
            &mut self.cand_ok,
            |view| view.get_external_slice_unchecked(buffers),
        );
    }

    fn has_mismatches(&self) -> bool {
        self.cand_ok.contains(&false)
    }

    /// The position and key index of each candidate that is a different key.
    fn mismatches(&self) -> impl Iterator<Item = (usize, IdxSize)> + '_ {
        self.cand_ok
            .iter()
            .enumerate()
            .filter(|(_, ok)| !**ok)
            .map(|(c, _)| (self.cand_pos[c], self.cand_key_idxs[c]))
    }
}

/// The size of a map, to roll back to.
struct Checkpoint {
    num_keys: IdxSize,
    num_buffers: usize,
    last_buffer_len: usize,
}

/// An IndexMap from key rows to values. It owns copies of the bytes of long views,
/// which the views in its rows point into.
pub(crate) struct KeyRowIndexMap<V> {
    table: HashTable<IdxSize>,
    /// Per key its hash followed by its row.
    entries: Vec<u64>,
    values: Vec<V>,
    buffers: Vec<Vec<u8>>,
    layout: Arc<KeyRowLayout>,
    /// A random odd number that hashes are multiplied by before they are probed.
    seed: u64,
}

impl<V> KeyRowIndexMap<V> {
    pub(crate) fn new(layout: Arc<KeyRowLayout>) -> Self {
        Self {
            table: HashTable::new(),
            entries: Vec::new(),
            values: Vec::new(),
            buffers: Vec::new(),
            layout,
            seed: rand::random::<u64>() | 1,
        }
    }

    pub(crate) fn layout(&self) -> &Arc<KeyRowLayout> {
        &self.layout
    }

    fn entry_words(&self) -> usize {
        self.layout.stride_words + 1
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        let (entries, entry_words, seed) = (&self.entries, self.entry_words(), self.seed);
        self.table.reserve(additional, |i| unsafe {
            entries
                .get_unchecked(*i as usize * entry_words)
                .wrapping_mul(seed)
        });
        self.entries.reserve(additional * entry_words);
        self.values.reserve(additional);
    }

    pub(crate) fn len(&self) -> IdxSize {
        self.values.len() as IdxSize
    }

    /// Gets the index by insertion order of key `i` of `keys`.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of this map.
    #[inline(always)]
    pub(crate) unsafe fn get_index_of(&self, keys: &KeyRowKeys, i: usize) -> Option<IdxSize> {
        let hash = keys.hashes.value_unchecked(i);
        let entry_words = self.entry_words();
        self.table
            .find(hash.wrapping_mul(self.seed), |j| {
                let entry = self
                    .entries
                    .get_unchecked(*j as usize * entry_words..(*j as usize + 1) * entry_words);
                *entry.get_unchecked(0) == hash
                    && keys.eq_stored(i, entry.get_unchecked(1..), |view| {
                        view.get_external_slice_unchecked(&self.buffers)
                    })
            })
            .copied()
    }

    /// Returns the index of key `i` of `keys`, inserting it with `value()` if it is
    /// new, and whether it was inserted.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of this map.
    #[inline(always)]
    pub(crate) unsafe fn get_or_insert_with(
        &mut self,
        keys: &KeyRowKeys,
        i: usize,
        value: impl FnOnce() -> V,
    ) -> (IdxSize, bool) {
        let entry_words = self.entry_words();
        let hash = keys.hashes.value_unchecked(i);
        let (entries, buffers, seed) = (&self.entries, &self.buffers, self.seed);
        let entry = self.table.entry(
            hash.wrapping_mul(seed),
            |j| {
                let entry = entries
                    .get_unchecked(*j as usize * entry_words..(*j as usize + 1) * entry_words);
                *entry.get_unchecked(0) == hash
                    && keys.eq_stored(i, entry.get_unchecked(1..), |view| {
                        view.get_external_slice_unchecked(buffers)
                    })
            },
            |j| {
                entries
                    .get_unchecked(*j as usize * entry_words)
                    .wrapping_mul(seed)
            },
        );
        match entry {
            TEntry::Occupied(o) => (*o.get(), false),
            TEntry::Vacant(v) => {
                let idx = self.values.len() as IdxSize;
                v.insert(idx);
                Self::push_entry(
                    &mut self.entries,
                    &mut self.buffers,
                    entry_words,
                    keys,
                    i,
                    hash,
                );
                self.values.push(value());
                (idx, true)
            },
        }
    }

    /// # Safety
    /// `i` must be in-bounds.
    #[inline(always)]
    unsafe fn push_entry(
        entries: &mut Vec<u64>,
        buffers: &mut Vec<Vec<u8>>,
        entry_words: usize,
        keys: &KeyRowKeys,
        i: usize,
        hash: u64,
    ) {
        entries.push(hash);
        let start = entries.len();
        entries.resize(start + entry_words - 1, 0);
        keys.write_row(i, &mut entries[start..], |bytes| {
            push_long_bytes(buffers, bytes)
        });
    }

    #[inline(always)]
    fn find_hash(&self, hash: u64) -> Option<IdxSize> {
        let entry_words = self.entry_words();
        self.table
            .find(hash.wrapping_mul(self.seed), |j| unsafe {
                *self.entries.get_unchecked(*j as usize * entry_words) == hash
            })
            .copied()
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            num_keys: self.len(),
            num_buffers: self.buffers.len(),
            last_buffer_len: self.buffers.last().map_or(0, Vec::len),
        }
    }

    /// Removes the keys inserted since `checkpoint`, with their rows and long bytes.
    /// Their values must not have been pushed.
    fn rollback(&mut self, checkpoint: &Checkpoint) {
        debug_assert_eq!(self.values.len(), checkpoint.num_keys as usize);
        let entry_words = self.entry_words();
        let num_keys = (self.entries.len() / entry_words) as IdxSize;
        for map_idx in checkpoint.num_keys..num_keys {
            let hash = self.entries[map_idx as usize * entry_words];
            self.table
                .find_entry(hash.wrapping_mul(self.seed), |j| *j == map_idx)
                .unwrap()
                .remove();
        }
        self.entries
            .truncate(checkpoint.num_keys as usize * entry_words);
        self.buffers.truncate(checkpoint.num_buffers);
        if let Some(buffer) = self.buffers.last_mut() {
            buffer.truncate(checkpoint.last_buffer_len);
        }
    }

    /// Pushes the index of each key `key_idxs[r]` of `keys` to `out`, or
    /// `IdxSize::MAX` when the key is absent or null.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of this map.
    pub(crate) unsafe fn get_indices_of(
        &self,
        keys: &KeyRowKeys,
        key_idxs: &[IdxSize],
        out: &mut Vec<IdxSize>,
    ) {
        debug_assert!(keys.has_layout(&self.layout));
        let entry_words = self.entry_words();
        let mut batch = LookupBatch::default();
        for chunk in key_idxs.chunks(VERIFY_BATCH_SIZE) {
            let start = out.len();
            batch.clear();
            for (pos, key_idx) in chunk.iter().enumerate() {
                let is_valid = keys
                    .validity
                    .as_ref()
                    .is_none_or(|v| v.get_bit_unchecked(*key_idx as usize));
                let hash = keys.hashes.value_unchecked(*key_idx as usize);
                match self.find_hash(hash).filter(|_| is_valid) {
                    Some(map_idx) => {
                        out.push(map_idx);
                        batch.push_candidate(pos, *key_idx, map_idx);
                    },
                    None => out.push(IdxSize::MAX),
                }
            }
            batch.verify(keys, &self.entries, entry_words, &self.buffers);
            for (pos, key_idx) in batch.mismatches() {
                *out.get_unchecked_mut(start + pos) = self
                    .get_index_of(keys, key_idx as usize)
                    .unwrap_or(IdxSize::MAX);
            }
        }
    }

    /// Pushes the index of each key `key_idxs[r]` of `keys` to `out`, inserting
    /// missing keys. New keys get indices in the order they first occur, and
    /// `value(r)` is called exactly once per inserted key, with `r` its first
    /// occurrence.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of this map.
    pub(crate) unsafe fn get_or_insert_batch(
        &mut self,
        keys: &KeyRowKeys,
        key_idxs: &[IdxSize],
        mut value: impl FnMut(usize) -> V,
        out: &mut Vec<IdxSize>,
    ) {
        debug_assert!(keys.has_layout(&self.layout));
        let entry_words = self.entry_words();
        let seed = self.seed;
        let mut batch = LookupBatch::default();
        for (c, chunk) in key_idxs.chunks(VERIFY_BATCH_SIZE).enumerate() {
            let (start, chunk_start) = (out.len(), c * VERIFY_BATCH_SIZE);
            let checkpoint = self.checkpoint();
            let mut next_map_idx = checkpoint.num_keys;
            batch.clear();
            for (pos, key_idx) in chunk.iter().enumerate() {
                let hash = keys.hashes.value_unchecked(*key_idx as usize);
                let entries = &self.entries;
                let entry = self.table.entry(
                    hash.wrapping_mul(seed),
                    |j| *entries.get_unchecked(*j as usize * entry_words) == hash,
                    |j| {
                        entries
                            .get_unchecked(*j as usize * entry_words)
                            .wrapping_mul(seed)
                    },
                );
                match entry {
                    TEntry::Occupied(o) => {
                        let map_idx = *o.get();
                        out.push(map_idx);
                        batch.push_candidate(pos, *key_idx, map_idx);
                    },
                    TEntry::Vacant(v) => {
                        v.insert(next_map_idx);
                        self.entries.push(hash);
                        self.entries.resize(self.entries.len() + entry_words - 1, 0);
                        out.push(next_map_idx);
                        batch.push_new(pos, *key_idx, next_map_idx);
                        next_map_idx += 1;
                    },
                }
            }
            batch.write_new(keys, &mut self.entries, entry_words, &mut self.buffers);
            batch.verify(keys, &self.entries, entry_words, &self.buffers);
            if !batch.has_mismatches() {
                self.values
                    .extend(batch.new_pos.iter().map(|pos| value(chunk_start + pos)));
                continue;
            }

            self.rollback(&checkpoint);
            out.truncate(start);
            for (pos, key_idx) in chunk.iter().enumerate() {
                let (map_idx, _) =
                    self.get_or_insert_with(keys, *key_idx as usize, || value(chunk_start + pos));
                out.push(map_idx);
            }
        }
    }

    /// # Safety
    /// `idx` must be less than `len()`.
    #[inline(always)]
    pub(crate) unsafe fn value_unchecked(&self, idx: IdxSize) -> &V {
        self.values.get_unchecked(idx as usize)
    }

    /// # Safety
    /// `idx` must be less than `len()`.
    #[inline(always)]
    pub(crate) unsafe fn value_unchecked_mut(&mut self, idx: IdxSize) -> &mut V {
        self.values.get_unchecked_mut(idx as usize)
    }

    pub(crate) fn get_value(&self, idx: IdxSize) -> Option<&V> {
        self.values.get(idx as usize)
    }

    /// Returns the keys as columns of `schema`, in insertion order.
    pub(crate) fn keys_frame(&self, schema: &Schema) -> DataFrame {
        let buffers = self
            .buffers
            .iter()
            .map(|b| Buffer::from(b.clone()))
            .collect();
        self.layout.decode(
            schema,
            &self.entries,
            self.entry_words(),
            1,
            self.values.len(),
            &buffers,
        )
    }
}
