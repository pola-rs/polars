use std::sync::Arc;

use hashbrown::hash_table::{Entry as TEntry, HashTable};
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_utils::IdxSize;

use super::keys::KeyRowKeys;
use super::layout::KeyRowLayout;
use super::{VERIFY_BATCH_SIZE, push_long_bytes};

/// Candidates found by hash, to be verified together.
struct VerifyBatch {
    pos: Vec<usize>,
    idxs: Vec<IdxSize>,
    entries: Vec<IdxSize>,
    rows: Vec<*const u64>,
    ok: Vec<bool>,
    new_pos: Vec<usize>,
    new_idxs: Vec<IdxSize>,
    new_entries: Vec<IdxSize>,
    new_rows: Vec<*mut u64>,
}

impl Default for VerifyBatch {
    fn default() -> Self {
        let n = VERIFY_BATCH_SIZE;
        Self {
            pos: Vec::with_capacity(n),
            idxs: Vec::with_capacity(n),
            entries: Vec::with_capacity(n),
            rows: Vec::with_capacity(n),
            ok: Vec::with_capacity(n),
            new_pos: Vec::with_capacity(n),
            new_idxs: Vec::with_capacity(n),
            new_entries: Vec::with_capacity(n),
            new_rows: Vec::with_capacity(n),
        }
    }
}

impl VerifyBatch {
    fn clear(&mut self) {
        self.pos.clear();
        self.idxs.clear();
        self.entries.clear();
        self.new_pos.clear();
        self.new_idxs.clear();
        self.new_entries.clear();
    }

    #[inline(always)]
    fn push_new(&mut self, pos: usize, i: IdxSize, entry: IdxSize) {
        self.new_pos.push(pos);
        self.new_idxs.push(i);
        self.new_entries.push(entry);
    }

    /// Writes the rows of the new entries, which are zeroed.
    ///
    /// # Safety
    /// The new entries must be in-bounds for `keys` and `entries`.
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
            self.new_entries
                .iter()
                .map(|j| base.add(*j as usize * entry_words + 1)),
        );
        keys.write_rows(&self.new_idxs, &self.new_rows, |bytes| {
            push_long_bytes(buffers, bytes)
        });
    }

    #[inline(always)]
    fn push(&mut self, pos: usize, i: IdxSize, entry: IdxSize) {
        self.pos.push(pos);
        self.idxs.push(i);
        self.entries.push(entry);
    }

    /// # Safety
    /// The candidates must be in-bounds for `keys` and `entries`.
    unsafe fn verify(
        &mut self,
        keys: &KeyRowKeys,
        entries: &[u64],
        entry_words: usize,
        buffers: &[Vec<u8>],
    ) {
        self.rows.clear();
        self.rows.extend(
            self.entries
                .iter()
                .map(|j| entries.as_ptr().add(*j as usize * entry_words + 1)),
        );
        self.ok.clear();
        self.ok.resize(self.idxs.len(), true);
        keys.verify(&self.idxs, &self.rows, &mut self.ok, |view| {
            view.get_external_slice_unchecked(buffers)
        });
    }

    /// The position and key index of each candidate that is a different key.
    fn mismatches(&self) -> impl Iterator<Item = (usize, IdxSize)> + '_ {
        self.ok
            .iter()
            .enumerate()
            .filter(|(_, ok)| !**ok)
            .map(|(c, _)| (self.pos[c], self.idxs[c]))
    }
}

/// An IndexMap from key rows to values. It owns copies of the bytes of long views.
pub struct KeyRowIndexMap<V> {
    table: HashTable<IdxSize>,
    /// Per key its hash followed by its row.
    entries: Vec<u64>,
    values: Vec<V>,
    buffers: Vec<Vec<u8>>,
    layout: Option<Arc<KeyRowLayout>>,
    /// A random odd number that hashes are multiplied by before they are probed.
    seed: u64,
}

impl<V> Default for KeyRowIndexMap<V> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            entries: Vec::new(),
            values: Vec::new(),
            buffers: Vec::new(),
            layout: None,
            seed: rand::random::<u64>() | 1,
        }
    }
}

impl<V> KeyRowIndexMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry_words(&self) -> usize {
        self.layout.as_ref().map_or(0, |l| l.stride + 1)
    }

    pub fn reserve(&mut self, additional: usize) {
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
    /// `i` must be in-bounds, and `keys` must have the layout of the keys in this map.
    #[inline(always)]
    pub unsafe fn get_index_of(&self, keys: &KeyRowKeys, i: usize) -> Option<IdxSize> {
        let layout = self.layout.as_deref()?;
        let hash = keys.hashes.value_unchecked(i);
        let entry_words = layout.stride + 1;
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
    /// `i` must be in-bounds, and `keys` must have the layout of the keys in this map.
    #[inline(always)]
    pub unsafe fn get_or_insert_with(
        &mut self,
        keys: &KeyRowKeys,
        i: usize,
        value: impl FnOnce() -> V,
    ) -> (IdxSize, bool) {
        let entry_words = self
            .layout
            .get_or_insert_with(|| keys.layout.clone())
            .stride
            + 1;
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

    /// # Safety
    /// The map must have a layout.
    #[inline(always)]
    unsafe fn find_hash(&self, hash: u64, entry_words: usize) -> Option<IdxSize> {
        self.table
            .find(hash.wrapping_mul(self.seed), |j| {
                *self.entries.get_unchecked(*j as usize * entry_words) == hash
            })
            .copied()
    }

    /// Pushes the index of each key `idxs[r]` of `keys` to `out`, or `IdxSize::MAX`
    /// when the key is absent or null.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of the keys in
    /// this map.
    pub unsafe fn get_indices_of(
        &self,
        keys: &KeyRowKeys,
        idxs: &[IdxSize],
        out: &mut Vec<IdxSize>,
    ) {
        let Some(layout) = self.layout.as_deref() else {
            out.extend(std::iter::repeat_n(IdxSize::MAX, idxs.len()));
            return;
        };
        let entry_words = layout.stride + 1;
        let mut batch = VerifyBatch::default();
        for chunk in idxs.chunks(VERIFY_BATCH_SIZE) {
            let start = out.len();
            batch.clear();
            for (r, i) in chunk.iter().enumerate() {
                let is_valid = keys
                    .validity
                    .as_ref()
                    .is_none_or(|v| v.get_bit_unchecked(*i as usize));
                let hash = keys.hashes.value_unchecked(*i as usize);
                match self.find_hash(hash, entry_words).filter(|_| is_valid) {
                    Some(j) => {
                        out.push(j);
                        batch.push(r, *i, j);
                    },
                    None => out.push(IdxSize::MAX),
                }
            }
            batch.verify(keys, &self.entries, entry_words, &self.buffers);
            for (r, i) in batch.mismatches() {
                *out.get_unchecked_mut(start + r) =
                    self.get_index_of(keys, i as usize).unwrap_or(IdxSize::MAX);
            }
        }
    }

    /// Pushes the index of each key `idxs[r]` of `keys` to `out`, inserting missing
    /// keys with `value(r)`. New keys get indices in the order they first occur.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of the keys in
    /// this map.
    pub unsafe fn get_or_insert_batch(
        &mut self,
        keys: &KeyRowKeys,
        idxs: &[IdxSize],
        mut value: impl FnMut(usize) -> V,
        out: &mut Vec<IdxSize>,
    ) {
        let entry_words = self
            .layout
            .get_or_insert_with(|| keys.layout.clone())
            .stride
            + 1;
        let seed = self.seed;
        let mut batch = VerifyBatch::default();
        for (c, chunk) in idxs.chunks(VERIFY_BATCH_SIZE).enumerate() {
            let (start, chunk_start) = (out.len(), c * VERIFY_BATCH_SIZE);
            let first_new = self.len();
            let mut next = first_new;
            let num_buffers = self.buffers.len();
            let last_buffer_len = self.buffers.last().map_or(0, Vec::len);
            batch.clear();
            for (r, i) in chunk.iter().enumerate() {
                let hash = keys.hashes.value_unchecked(*i as usize);
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
                        let j = *o.get();
                        out.push(j);
                        batch.push(r, *i, j);
                    },
                    TEntry::Vacant(v) => {
                        v.insert(next);
                        self.entries.push(hash);
                        self.entries.resize(self.entries.len() + entry_words - 1, 0);
                        out.push(next);
                        batch.push_new(r, *i, next);
                        next += 1;
                    },
                }
            }
            batch.write_new(keys, &mut self.entries, entry_words, &mut self.buffers);
            batch.verify(keys, &self.entries, entry_words, &self.buffers);
            if !batch.ok.contains(&false) {
                self.values
                    .extend(batch.new_pos.iter().map(|r| value(chunk_start + r)));
                continue;
            }

            for j in first_new..next {
                let hash = *self.entries.get_unchecked(j as usize * entry_words);
                self.table
                    .find_entry(hash.wrapping_mul(seed), |k| *k == j)
                    .unwrap()
                    .remove();
            }
            self.entries.truncate(first_new as usize * entry_words);
            self.buffers.truncate(num_buffers);
            if let Some(buffer) = self.buffers.last_mut() {
                buffer.truncate(last_buffer_len);
            }
            out.truncate(start);
            for (r, i) in chunk.iter().enumerate() {
                out.push(
                    self.get_or_insert_with(keys, *i as usize, || value(chunk_start + r))
                        .0,
                );
            }
        }
    }

    /// # Safety
    /// `idx` must be less than `len()`.
    #[inline(always)]
    pub unsafe fn value_unchecked(&self, idx: IdxSize) -> &V {
        self.values.get_unchecked(idx as usize)
    }

    /// # Safety
    /// `idx` must be less than `len()`.
    #[inline(always)]
    pub unsafe fn value_unchecked_mut(&mut self, idx: IdxSize) -> &mut V {
        self.values.get_unchecked_mut(idx as usize)
    }

    pub fn get_value(&self, idx: IdxSize) -> Option<&V> {
        self.values.get(idx as usize)
    }

    /// Returns the keys as columns of `schema`, in insertion order.
    pub fn keys_frame(&self, schema: &Schema) -> DataFrame {
        let Some(layout) = &self.layout else {
            return DataFrame::empty_with_schema(schema);
        };
        let buffers = self
            .buffers
            .iter()
            .map(|b| Buffer::from(b.clone()))
            .collect();
        layout.decode(
            schema,
            &self.entries,
            layout.stride + 1,
            1,
            self.values.len(),
            &buffers,
        )
    }
}
