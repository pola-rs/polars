#![allow(clippy::unnecessary_cast)] // Clippy doesn't recognize that IdxSize and u64 can be different.
#![allow(unsafe_op_in_unsafe_fn)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use polars_utils::idx_map::total_idx_map::{Entry, TotalIndexMap};
use polars_utils::idx_vec::UnitVec;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::{TotalEq, TotalHash};
use polars_utils::unitvec;

use super::*;
use crate::hash_keys::HashKeys;

pub struct SingleKeyIdxTable<T: PolarsDataType> {
    // These AtomicU64s actually are IdxSizes, but we use the top bit of the
    // first index in each to mark keys during probing.
    idx_map: TotalIndexMap<T::Physical<'static>, UnitVec<AtomicU64>>,
    idx_offset: IdxSize,
    null_keys: Vec<IdxSize>,
    nulls_emitted: AtomicBool,
}

impl<T: PolarsDataType> SingleKeyIdxTable<T> {
    pub fn new() -> Self {
        Self {
            idx_map: TotalIndexMap::default(),
            idx_offset: 0,
            null_keys: Vec::new(),
            nulls_emitted: AtomicBool::new(false),
        }
    }
}

impl<T, K> SingleKeyIdxTable<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: TotalHash + TotalEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn probe_one<const MARK_MATCHES: bool>(
        &self,
        key_idx: IdxSize,
        key: &K,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
    ) -> bool {
        if let Some(idxs) = self.idx_map.get(key) {
            for idx in &idxs[..] {
                // Create matches, making sure to clear top bit.
                table_match.push((idx.load(Ordering::Relaxed) & !(1 << 63)) as IdxSize);
                probe_match.push(key_idx);
            }

            // Mark if necessary. This action is idempotent so doesn't need
            // atomic fetch_or to do it atomically.
            if MARK_MATCHES {
                let first_idx = unsafe { idxs.get_unchecked(0) };
                let first_idx_val = first_idx.load(Ordering::Relaxed);
                if first_idx_val >> 63 == 0 {
                    first_idx.store(first_idx_val | (1 << 63), Ordering::Relaxed);
                }
            }
            true
        } else {
            false
        }
    }

    fn probe_impl<
        const MARK_MATCHES: bool,
        const EMIT_UNMATCHED: bool,
        const NULL_IS_VALID: bool,
    >(
        &self,
        keys: impl Iterator<Item = (IdxSize, Option<K>)>,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        limit: IdxSize,
    ) -> IdxSize {
        let mut keys_processed = 0;
        for (key_idx, key) in keys {
            let found_match = if let Some(key) = key {
                self.probe_one::<MARK_MATCHES>(key_idx, &key, table_match, probe_match)
            } else if NULL_IS_VALID {
                for idx in &self.null_keys {
                    table_match.push(*idx);
                    probe_match.push(key_idx);
                }
                if MARK_MATCHES && !self.nulls_emitted.load(Ordering::Relaxed) {
                    self.nulls_emitted.store(true, Ordering::Relaxed);
                }
                !self.null_keys.is_empty()
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

    #[allow(clippy::too_many_arguments)]
    fn probe_dispatch(
        &self,
        keys: impl Iterator<Item = (IdxSize, Option<K>)>,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        null_is_valid: bool,
        limit: IdxSize,
    ) -> IdxSize {
        match (mark_matches, emit_unmatched, null_is_valid) {
            (false, false, false) => {
                self.probe_impl::<false, false, false>(keys, table_match, probe_match, limit)
            },
            (false, false, true) => {
                self.probe_impl::<false, false, true>(keys, table_match, probe_match, limit)
            },
            (false, true, false) => {
                self.probe_impl::<false, true, false>(keys, table_match, probe_match, limit)
            },
            (false, true, true) => {
                self.probe_impl::<false, true, true>(keys, table_match, probe_match, limit)
            },
            (true, false, false) => {
                self.probe_impl::<true, false, false>(keys, table_match, probe_match, limit)
            },
            (true, false, true) => {
                self.probe_impl::<true, false, true>(keys, table_match, probe_match, limit)
            },
            (true, true, false) => {
                self.probe_impl::<true, true, false>(keys, table_match, probe_match, limit)
            },
            (true, true, true) => {
                self.probe_impl::<true, true, true>(keys, table_match, probe_match, limit)
            },
        }
    }
}

impl<T, K> IdxTable for SingleKeyIdxTable<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: TotalHash + TotalEq + Send + Sync + 'static,
{
    fn new_empty(&self) -> Box<dyn IdxTable> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_keys(&self) -> IdxSize {
        self.idx_map.len()
    }

    fn insert_keys(&mut self, _hash_keys: &HashKeys, _track_unmatchable: bool) {
        // Isn't needed anymore, but also don't want to remove the code from the other implementations.
        unimplemented!()
    }

    unsafe fn insert_keys_subset(
        &mut self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        track_unmatchable: bool,
    ) {
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };
        let new_idx_offset = (self.idx_offset as usize)
            .checked_add(subset.len())
            .unwrap();
        assert!(
            new_idx_offset < IdxSize::MAX as usize,
            "overly large index in SingleKeyIdxTable"
        );

        let keys: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        for (i, subset_idx) in subset.iter().enumerate_idx() {
            let key = unsafe { keys.get_unchecked(*subset_idx as usize) };
            let idx = self.idx_offset + i;
            if let Some(key) = key {
                match self.idx_map.entry(key) {
                    Entry::Occupied(o) => {
                        o.into_mut().push(AtomicU64::new(idx as u64));
                    },
                    Entry::Vacant(v) => {
                        v.insert(unitvec![AtomicU64::new(idx as u64)]);
                    },
                }
            } else if track_unmatchable | hash_keys.null_is_valid {
                self.null_keys.push(idx);
            }
        }

        self.idx_offset = new_idx_offset as IdxSize;
    }

    fn probe(
        &self,
        _hash_keys: &HashKeys,
        _table_match: &mut Vec<IdxSize>,
        _probe_match: &mut Vec<IdxSize>,
        _mark_matches: bool,
        _emit_unmatched: bool,
        _limit: IdxSize,
    ) -> IdxSize {
        // Isn't needed anymore, but also don't want to remove the code from the other implementations.
        unimplemented!()
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
        let HashKeys::Single(hash_keys) = hash_keys else {
            unreachable!()
        };

        let keys: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        if keys.has_nulls() {
            let iter = subset.iter().map(|i| (*i, keys.get_unchecked(*i as usize)));
            self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                hash_keys.null_is_valid,
                limit,
            )
        } else {
            let iter = subset
                .iter()
                .map(|i| (*i, Some(keys.value_unchecked(*i as usize))));
            self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                false, // Whether or not nulls are valid doesn't matter.
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
        if !self.nulls_emitted.load(Ordering::Relaxed) {
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
        }

        while let Some((_, idxs)) = self.idx_map.get_index(offset) {
            let first_idx = unsafe { idxs.get_unchecked(0) };
            let first_idx_val = first_idx.load(Ordering::Relaxed);
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
