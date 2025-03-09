#![allow(clippy::unnecessary_cast)] // Clippy doesn't recognize that IdxSize and u64 can be different.
#![allow(unsafe_op_in_unsafe_fn)]

use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::Array;
use polars_utils::idx_map::bytes_idx_map::{BytesIndexMap, Entry};
use polars_utils::idx_vec::UnitVec;
use polars_utils::itertools::Itertools;
use polars_utils::unitvec;

use super::*;
use crate::hash_keys::HashKeys;

#[derive(Default)]
pub struct RowEncodedIdxTable {
    // These AtomicU64s actually are IdxSizes, but we use the top bit of the
    // first index in each to mark keys during probing.
    idx_map: BytesIndexMap<UnitVec<AtomicU64>>,
    idx_offset: IdxSize,
    null_keys: Vec<IdxSize>,
}

impl RowEncodedIdxTable {
    pub fn new() -> Self {
        Self {
            idx_map: BytesIndexMap::new(),
            idx_offset: 0,
            null_keys: Vec::new(),
        }
    }
}

impl RowEncodedIdxTable {
    #[inline(always)]
    fn probe_one<const MARK_MATCHES: bool>(
        &self,
        key_idx: IdxSize,
        hash: u64,
        key: &[u8],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
    ) -> bool {
        if let Some(idxs) = self.idx_map.get(hash, key) {
            for idx in &idxs[..] {
                // Create matches, making sure to clear top bit.
                table_match.push((idx.load(Ordering::Relaxed) & !(1 << 63)) as IdxSize);
                probe_match.push(key_idx);
            }

            // Mark if necessary. This action is idempotent so doesn't
            // need any synchronization on the load, nor does it need a
            // fetch_or to do it atomically.
            if MARK_MATCHES {
                let first_idx = unsafe { idxs.get_unchecked(0) };
                let first_idx_val = first_idx.load(Ordering::Relaxed);
                if first_idx_val >> 63 == 0 {
                    first_idx.store(first_idx_val | (1 << 63), Ordering::Release);
                }
            }
            true
        } else {
            false
        }
    }

    fn probe_impl<'a, const MARK_MATCHES: bool, const EMIT_UNMATCHED: bool>(
        &self,
        hash_keys: impl Iterator<Item = (IdxSize, u64, Option<&'a [u8]>)>,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        limit: IdxSize,
    ) -> IdxSize {
        let mut keys_processed = 0;
        for (key_idx, hash, key) in hash_keys {
            let found_match = if let Some(key) = key {
                self.probe_one::<MARK_MATCHES>(key_idx, hash, key, table_match, probe_match)
            } else {
                false
            };

            if EMIT_UNMATCHED && !found_match {
                table_match.push(IdxSize::MAX);
                probe_match.push(key_idx);
            }

            keys_processed += 1;
            if table_match.len() >= limit as usize {
                break;
            }
        }
        keys_processed
    }

    fn probe_dispatch<'a>(
        &self,
        hash_keys: impl Iterator<Item = (IdxSize, u64, Option<&'a [u8]>)>,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> IdxSize {
        match (mark_matches, emit_unmatched) {
            (false, false) => {
                self.probe_impl::<false, false>(hash_keys, table_match, probe_match, limit)
            },
            (false, true) => {
                self.probe_impl::<false, true>(hash_keys, table_match, probe_match, limit)
            },
            (true, false) => {
                self.probe_impl::<true, false>(hash_keys, table_match, probe_match, limit)
            },
            (true, true) => {
                self.probe_impl::<true, true>(hash_keys, table_match, probe_match, limit)
            },
        }
    }
}

impl IdxTable for RowEncodedIdxTable {
    fn new_empty(&self) -> Box<dyn IdxTable> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_keys(&self) -> IdxSize {
        self.idx_map.len()
    }

    fn insert_keys(&mut self, hash_keys: &HashKeys, track_unmatchable: bool) {
        let HashKeys::RowEncoded(hash_keys) = hash_keys else {
            unreachable!()
        };
        let new_idx_offset = (self.idx_offset as usize)
            .checked_add(hash_keys.keys.len())
            .unwrap();
        assert!(
            new_idx_offset < IdxSize::MAX as usize,
            "overly large index in RowEncodedIdxTable"
        );

        for (i, (hash, key)) in hash_keys
            .hashes
            .values_iter()
            .zip(hash_keys.keys.iter())
            .enumerate_idx()
        {
            let idx = self.idx_offset + i;
            if let Some(key) = key {
                match self.idx_map.entry(*hash, key) {
                    Entry::Occupied(o) => {
                        o.into_mut().push(AtomicU64::new(idx as u64));
                    },
                    Entry::Vacant(v) => {
                        v.insert(unitvec![AtomicU64::new(idx as u64)]);
                    },
                }
            } else if track_unmatchable {
                self.null_keys.push(idx);
            }
        }

        self.idx_offset = new_idx_offset as IdxSize;
    }

    unsafe fn insert_keys_subset(
        &mut self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        track_unmatchable: bool,
    ) {
        let HashKeys::RowEncoded(hash_keys) = hash_keys else {
            unreachable!()
        };
        let new_idx_offset = (self.idx_offset as usize)
            .checked_add(subset.len())
            .unwrap();
        assert!(
            new_idx_offset < IdxSize::MAX as usize,
            "overly large index in RowEncodedIdxTable"
        );

        for (i, subset_idx) in subset.iter().enumerate_idx() {
            let hash = unsafe { hash_keys.hashes.value_unchecked(*subset_idx as usize) };
            let key = unsafe { hash_keys.keys.get_unchecked(*subset_idx as usize) };
            let idx = self.idx_offset + i;
            if let Some(key) = key {
                match self.idx_map.entry(hash, key) {
                    Entry::Occupied(o) => {
                        o.into_mut().push(AtomicU64::new(idx as u64));
                    },
                    Entry::Vacant(v) => {
                        v.insert(unitvec![AtomicU64::new(idx as u64)]);
                    },
                }
            } else if track_unmatchable {
                self.null_keys.push(idx);
            }
        }

        self.idx_offset = new_idx_offset as IdxSize;
    }

    fn probe(
        &self,
        hash_keys: &HashKeys,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> IdxSize {
        let HashKeys::RowEncoded(hash_keys) = hash_keys else {
            unreachable!()
        };

        if hash_keys.keys.has_nulls() {
            let iter = hash_keys
                .hashes
                .values_iter()
                .copied()
                .zip(hash_keys.keys.iter())
                .enumerate_idx()
                .map(|(i, (h, k))| (i, h, k));
            self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                limit,
            )
        } else {
            let iter = hash_keys
                .hashes
                .values_iter()
                .copied()
                .zip(hash_keys.keys.values_iter().map(Some))
                .enumerate_idx()
                .map(|(i, (h, k))| (i, h, k));
            self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                limit,
            )
        }
    }

    unsafe fn probe_subset(
        &self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> IdxSize {
        let HashKeys::RowEncoded(hash_keys) = hash_keys else {
            unreachable!()
        };

        if hash_keys.keys.has_nulls() {
            let iter = subset.iter().map(|i| {
                (
                    *i,
                    hash_keys.hashes.value_unchecked(*i as usize),
                    hash_keys.keys.get_unchecked(*i as usize),
                )
            });
            self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                limit,
            )
        } else {
            let iter = subset.iter().map(|i| {
                (
                    *i,
                    hash_keys.hashes.value_unchecked(*i as usize),
                    Some(hash_keys.keys.value_unchecked(*i as usize)),
                )
            });
            self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                limit,
            )
        }
    }

    fn unmarked_keys(
        &self,
        out: &mut Vec<IdxSize>,
        mut offset: IdxSize,
        limit: IdxSize,
    ) -> IdxSize {
        out.clear();

        let mut keys_processed = 0;
        if (offset as usize) < self.null_keys.len() {
            out.extend(
                self.null_keys[offset as usize..]
                    .iter()
                    .copied()
                    .take(limit as usize),
            );
            keys_processed += out.len() as IdxSize;
            offset += out.len() as IdxSize;
            if out.len() >= limit as usize {
                return keys_processed;
            }
        }

        offset -= self.null_keys.len() as IdxSize;

        while let Some((_, _, idxs)) = self.idx_map.get_index(offset) {
            let first_idx = unsafe { idxs.get_unchecked(0) };
            let first_idx_val = first_idx.load(Ordering::Acquire);
            if first_idx_val >> 63 == 0 {
                for idx in &idxs[..] {
                    out.push((idx.load(Ordering::Relaxed) & !(1 << 63)) as IdxSize);
                }
            }

            keys_processed += 1;
            offset += 1;
            if out.len() >= limit as usize {
                break;
            }
        }

        keys_processed
    }
}
