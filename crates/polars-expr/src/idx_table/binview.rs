#![allow(clippy::unnecessary_cast)] // Clippy doesn't recognize that IdxSize and u64 can be different.
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_arrow::array::Array;
use polars_compute::binview_index_map::{BinaryViewIndexMap, Entry};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::idx_vec::UnitVec;
use polars_utils::itertools::Itertools;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::unitvec;

use super::*;
use crate::hash_keys::{BinviewKeys, DictCodeCache, DictCodes, HashKeys, MAX_DICT_CODE_CACHE_LEN};

pub struct BinviewKeyIdxTable {
    idx_map: BinaryViewIndexMap<UnitVec<RelaxedCell<IdxSize>>>,
    idx_offset: IdxSize,
    null_keys: Vec<IdxSize>,
    nulls_marked: RelaxedCell<bool>,
    matched: MatchedRows,
    /// Identifies the table's current contents for per-thread probe caches.
    uid: u64,
}

impl BinviewKeyIdxTable {
    pub fn new() -> Self {
        Self {
            idx_map: BinaryViewIndexMap::default(),
            idx_offset: 0,
            null_keys: Vec::new(),
            nulls_marked: RelaxedCell::from(false),
            matched: MatchedRows::default(),
            uid: next_table_uid(),
        }
    }

    /// Pushes the matches of a key whose table entry is `idxs`.
    #[inline(always)]
    unsafe fn probe_one<const MARK_MATCHES: bool>(
        &self,
        key_idx: IdxSize,
        idxs: Option<&UnitVec<RelaxedCell<IdxSize>>>,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
    ) -> bool {
        if let Some(idxs) = idxs {
            for idx in &idxs[..] {
                table_match.push(idx.load() & !MARKED);
                probe_match.push(key_idx);
            }

            if MARK_MATCHES {
                unsafe { self.matched.mark_rows(idxs) };
            }
            true
        } else {
            false
        }
    }

    /// Probes keys given as their table entries; a null key is `None`.
    unsafe fn probe_impl<
        'a,
        const MARK_MATCHES: bool,
        const EMIT_UNMATCHED: bool,
        const NULL_IS_VALID: bool,
    >(
        &'a self,
        keys: impl Iterator<Item = (IdxSize, Option<Option<&'a UnitVec<RelaxedCell<IdxSize>>>>)>,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        limit: IdxSize,
    ) -> IdxSize {
        let mut keys_processed = 0;
        for (key_idx, key) in keys {
            let found_match = if let Some(idxs) = key {
                self.probe_one::<MARK_MATCHES>(key_idx, idxs, table_match, probe_match)
            } else if NULL_IS_VALID {
                for idx in &self.null_keys {
                    table_match.push(*idx);
                    probe_match.push(key_idx);
                }
                if MARK_MATCHES && !self.nulls_marked.load() {
                    self.nulls_marked.store(true);
                    for idx in &self.null_keys {
                        self.matched.mark(*idx);
                    }
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
    unsafe fn probe_dispatch<'a>(
        &'a self,
        keys: impl Iterator<Item = (IdxSize, Option<Option<&'a UnitVec<RelaxedCell<IdxSize>>>>)>,
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

    /// The table entry of each code of the keys' dictionary is looked up once, on its first
    /// valid key, and kept in a per-thread cache for this table version and dictionary.
    #[allow(clippy::too_many_arguments)]
    unsafe fn probe_subset_dict(
        &self,
        hash_keys: &BinviewKeys,
        dc: &DictCodes,
        subset: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> Option<IdxSize> {
        thread_local! {
            /// A cache per table, as a partitioned join probes a table per partition.
            static CACHES: RefCell<PlHashMap<u64, DictCodeCache>> = RefCell::new(PlHashMap::new());
        }
        /// Past this many tables, or entries in all, a thread's caches are dropped. This bounds the
        /// memory they keep once the tables are gone (they outlive the query).
        const MAX_CACHED_TABLES: usize = 64;
        const MAX_CACHED_ENTRIES: usize = 1 << 20;
        CACHES.with_borrow_mut(|caches| {
            if !caches.contains_key(&self.uid)
                && (caches.len() >= MAX_CACHED_TABLES
                    || caches.values().map(DictCodeCache::len).sum::<usize>()
                        + dc.dict.len().min(MAX_DICT_CODE_CACHE_LEN)
                        > MAX_CACHED_ENTRIES)
            {
                *caches = PlHashMap::new();
            }
            let cache = caches.entry(self.uid).or_default();
            let entries = cache.entries_for(&dc.dict, subset.len())?;
            let buffers = hash_keys.keys.data_buffers();
            let views = hash_keys.keys.views();
            let validity = hash_keys.keys.validity();
            let codes = dc.codes.as_slice();
            let iter = subset.iter().map(|i| {
                let i = *i as usize;
                if validity.is_some_and(|v| !v.get_bit_unchecked(i)) {
                    return (i as IdxSize, None);
                }
                let code = *codes.get_unchecked(i) as usize;
                let entry = entries.get_unchecked_mut(code);
                if *entry == IdxSize::MAX {
                    polars_core::frame::column::dictionary_stats::add(
                        polars_core::frame::column::dictionary_stats::Counter::ProbeCacheMisses,
                        1,
                    );
                    let hash = hash_keys.hashes.value_unchecked(i);
                    *entry = self
                        .idx_map
                        .get_index_of_view(hash, views.get_unchecked(i), buffers)
                        .unwrap_or(NO_ENTRY);
                } else {
                    polars_core::frame::column::dictionary_stats::add(
                        polars_core::frame::column::dictionary_stats::Counter::ProbeCacheHits,
                        1,
                    );
                }
                let idxs =
                    (*entry != NO_ENTRY).then(|| self.idx_map.get_index_view_unchecked(*entry).2);
                (i as IdxSize, Some(idxs))
            });
            Some(self.probe_dispatch(
                iter,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                hash_keys.null_is_valid,
                limit,
            ))
        })
    }
}

/// A dictionary code's cache entry when its value has no entry in the table.
const NO_ENTRY: IdxSize = IdxSize::MAX - 1;

static NEXT_TABLE_UID: AtomicU64 = AtomicU64::new(0);

fn next_table_uid() -> u64 {
    NEXT_TABLE_UID.fetch_add(1, Ordering::Relaxed)
}

impl IdxTable for BinviewKeyIdxTable {
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
        let HashKeys::Binview(hash_keys) = hash_keys else {
            unreachable!()
        };
        let new_idx_offset = (self.idx_offset as usize)
            .checked_add(subset.len())
            .unwrap();
        assert!(
            new_idx_offset <= MARKED as usize,
            "overly large index in BinviewKeyIdxTable"
        );
        if track_unmatchable {
            self.matched.grow(new_idx_offset);
        }
        self.uid = next_table_uid();

        unsafe {
            let buffers = hash_keys.keys.data_buffers();
            let views = hash_keys.keys.views();
            if let Some(validity) = hash_keys.keys.validity() {
                for (i, subset_idx) in subset.iter().enumerate_idx() {
                    let hash = hash_keys.hashes.value_unchecked(*subset_idx as usize);
                    let key = views.get_unchecked(*subset_idx as usize);
                    let idx = self.idx_offset + i;
                    if validity.get_bit_unchecked(*subset_idx as usize) {
                        match self.idx_map.entry_view(hash, *key, buffers) {
                            Entry::Occupied(o) => {
                                o.into_mut().push(RelaxedCell::from(idx));
                            },
                            Entry::Vacant(v) => {
                                v.insert(unitvec![RelaxedCell::from(idx)]);
                            },
                        }
                    } else if track_unmatchable | hash_keys.null_is_valid {
                        self.null_keys.push(idx);
                    }
                }
            } else {
                for (i, subset_idx) in subset.iter().enumerate_idx() {
                    let hash = hash_keys.hashes.value_unchecked(*subset_idx as usize);
                    let key = views.get_unchecked(*subset_idx as usize);
                    let idx = self.idx_offset + i;
                    match self.idx_map.entry_view(hash, *key, buffers) {
                        Entry::Occupied(o) => {
                            o.into_mut().push(RelaxedCell::from(idx));
                        },
                        Entry::Vacant(v) => {
                            v.insert(unitvec![RelaxedCell::from(idx)]);
                        },
                    }
                }
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
        let HashKeys::Binview(hash_keys) = hash_keys else {
            unreachable!()
        };

        if let Some(dc) = &hash_keys.dict_codes
            && let Some(n) = self.probe_subset_dict(
                hash_keys,
                dc,
                subset,
                table_match,
                probe_match,
                mark_matches,
                emit_unmatched,
                limit,
            )
        {
            return n;
        }

        unsafe {
            let buffers = hash_keys.keys.data_buffers();
            let views = hash_keys.keys.views();
            let lookup = |i: IdxSize| {
                let hash = hash_keys.hashes.value_unchecked(i as usize);
                self.idx_map
                    .get_view(hash, views.get_unchecked(i as usize), buffers)
            };
            if let Some(validity) = hash_keys.keys.validity() {
                let iter = subset.iter().map(|i| {
                    (
                        *i,
                        validity.get_bit_unchecked(*i as usize).then(|| lookup(*i)),
                    )
                });
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
                let iter = subset.iter().map(|i| (*i, Some(lookup(*i))));
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
    }

    fn unmarked_keys(&self, out: &mut Vec<IdxSize>, offset: IdxSize, limit: IdxSize) -> IdxSize {
        self.matched
            .unmarked_rows(self.idx_offset, out, offset, limit)
    }
}
