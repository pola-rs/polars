use std::hash::BuildHasher;

use hashbrown::HashTable;
use hashbrown::hash_table::Entry as TableEntry;
use polars_arrow::array::{Array, BinaryArray, View};
use polars_arrow::bitmap::BitmapBuilder;
use polars_core::prelude::*;
use polars_utils::IdxSize;
use polars_utils::aliases::PlRandomState;

use super::{SMALL_MAX, finish_chunk};

/// Deduplicated byte strings, probed by length prefilter and then linear scan or hash table.
///
/// Values are stored as views: up to 12 bytes inline, longer ones in `buffers`. Short strings
/// hash and compare as a single 16 byte block, without touching the string data.
pub(super) struct BinaryLookup {
    kind: BinaryLookupKind,
    buffers: Vec<Vec<u8>>,
    hasher: PlRandomState,
    /// Bit `l` is set if a value of length `l` is present; bit 63 covers all longer values.
    len_mask: u64,
}

enum BinaryLookupKind {
    Small(Vec<View>),
    Hash(HashTable<View>),
}

/// Keeps the length and the inline bytes of an inline view, ignoring any padding.
#[inline(always)]
fn inline_key(view: View) -> u128 {
    debug_assert!(view.is_inline());
    view.as_u128() & (u128::MAX >> ((View::MAX_INLINE_SIZE - view.length) * 8))
}

/// `key` is the inline key of `view` if it is inline, and `bytes` its data otherwise.
#[inline(always)]
fn eq_stored(
    buffers: &[Vec<u8>],
    stored: View,
    view: View,
    key: Option<u128>,
    bytes: &[u8],
) -> bool {
    match key {
        Some(key) => stored.is_inline() && inline_key(stored) == key,
        None => {
            stored.length == view.length
                && stored.prefix == view.prefix
                // SAFETY: the stored view was created from these buffers.
                && unsafe { stored.get_external_slice_unchecked(buffers) } == bytes
        },
    }
}

impl BinaryLookup {
    pub(super) fn new<'a>(values: impl Iterator<Item = &'a [u8]>) -> Self {
        let hasher = PlRandomState::default();
        let mut buffers: Vec<Vec<u8>> = Vec::new();
        let mut table = HashTable::new();
        let mut len_mask = 0u64;

        let hash_stored = |hasher: &PlRandomState, buffers: &[Vec<u8>], view: &View| {
            if view.is_inline() {
                hasher.hash_one(inline_key(*view))
            } else {
                // SAFETY: the view was created from these buffers.
                hasher.hash_one(unsafe { view.get_external_slice_unchecked(buffers) })
            }
        };
        for bytes in values {
            // Only the length and prefix are read before the value is stored.
            let probe = View::new_from_bytes(bytes, 0, 0);
            let key = probe.is_inline().then(|| inline_key(probe));
            let hash = match key {
                Some(key) => hasher.hash_one(key),
                None => hasher.hash_one(bytes),
            };
            let entry = table.entry(
                hash,
                |&stored| eq_stored(&buffers, stored, probe, key, bytes),
                |stored| hash_stored(&hasher, &buffers, stored),
            );
            if let TableEntry::Vacant(entry) = entry {
                entry.insert(View::new_with_buffers(bytes, 0, &mut buffers));
                len_mask |= 1 << bytes.len().min(63);
            }
        }

        let kind = if table.len() <= SMALL_MAX {
            BinaryLookupKind::Small(table.into_iter().collect())
        } else {
            BinaryLookupKind::Hash(table)
        };
        Self {
            kind,
            buffers,
            hasher,
            len_mask,
        }
    }

    /// `bytes` is only read if `view` is not inline.
    #[inline(always)]
    fn contains(&self, view: View, bytes: &[u8]) -> bool {
        if (self.len_mask >> view.length.min(63)) & 1 == 0 {
            return false;
        }
        let key = view.is_inline().then(|| inline_key(view));
        match &self.kind {
            BinaryLookupKind::Small(views) => views
                .iter()
                .any(|&stored| eq_stored(&self.buffers, stored, view, key, bytes)),
            BinaryLookupKind::Hash(table) => {
                let hash = match key {
                    Some(key) => self.hasher.hash_one(key),
                    None => self.hasher.hash_one(bytes),
                };
                table
                    .find(hash, |&stored| {
                        eq_stored(&self.buffers, stored, view, key, bytes)
                    })
                    .is_some()
            },
        }
    }

    pub(super) fn probe(
        &self,
        ca: &BinaryChunked,
        nulls_equal: bool,
        has_null: bool,
    ) -> BooleanChunked {
        let chunks = ca.downcast_iter().map(|arr| {
            let mut out = BitmapBuilder::with_capacity(arr.len());
            let buffers = arr.data_buffers();
            out.extend_trusted_len_iter(arr.views().iter().map(|&view| {
                let bytes = if view.is_inline() {
                    &[][..]
                } else {
                    // SAFETY: the view belongs to this array.
                    unsafe { view.get_external_slice_unchecked(buffers) }
                };
                self.contains(view, bytes)
            }));
            finish_chunk(out.freeze(), arr.validity(), nulls_equal, has_null)
        });
        BooleanChunked::from_chunk_iter(ca.name().clone(), chunks)
    }
}

/// Row encoded nested values, kept as one array and indexed by row.
pub(super) struct RowEncodedLookup {
    rows: BinaryArray<i64>,
    table: HashTable<IdxSize>,
    hasher: PlRandomState,
}

impl RowEncodedLookup {
    pub(super) fn new(rows: BinaryArray<i64>) -> Self {
        let hasher = PlRandomState::default();
        let mut table = HashTable::with_capacity(rows.len());
        for (idx, bytes) in rows.values_iter().enumerate() {
            let hash = hasher.hash_one(bytes);
            table
                .entry(
                    hash,
                    |&i| rows.value(i as usize) == bytes,
                    |&i| hasher.hash_one(rows.value(i as usize)),
                )
                .or_insert(idx as IdxSize);
        }
        Self {
            rows,
            table,
            hasher,
        }
    }

    pub(super) fn probe(
        &self,
        ca: &BinaryOffsetChunked,
        nulls_equal: bool,
        has_null: bool,
    ) -> BooleanChunked {
        let chunks = ca.downcast_iter().map(|arr| {
            let mut out = BitmapBuilder::with_capacity(arr.len());
            out.extend_trusted_len_iter(arr.values_iter().map(|bytes| {
                let hash = self.hasher.hash_one(bytes);
                self.table
                    .find(hash, |&i| self.rows.value(i as usize) == bytes)
                    .is_some()
            }));
            finish_chunk(out.freeze(), arr.validity(), nulls_equal, has_null)
        });
        BooleanChunked::from_chunk_iter(ca.name().clone(), chunks)
    }
}
