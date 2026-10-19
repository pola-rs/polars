#![allow(clippy::unnecessary_cast)] // Clippy doesn't recognize that IdxSize and u64 can be different.
#![allow(unsafe_op_in_unsafe_fn)]

use polars_utils::idx_vec::UnitVec;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::unitvec;

use super::*;
use crate::hash_keys::HashKeys;
use crate::key_rows::{KeyRowIndexMap, KeyRowKeys, KeyRowLayout};

const PROBE_CHUNK_SIZE: usize = 1024;

pub struct KeyRowIdxTable {
    idx_map: KeyRowIndexMap<UnitVec<RelaxedCell<IdxSize>>>,
    idx_offset: IdxSize,
    matched: MatchedRows,
}

impl KeyRowIdxTable {
    pub fn new(layout: Arc<KeyRowLayout>) -> Self {
        Self {
            idx_map: KeyRowIndexMap::new(layout),
            idx_offset: 0,
            matched: MatchedRows::default(),
        }
    }

    /// Inserts the keys `key_idxs`, where key `key_idxs[p]` gets index `idx_offset + p`.
    ///
    /// # Safety
    /// The indices must be in-bounds.
    unsafe fn insert_keys_impl(
        &mut self,
        keys: &KeyRowKeys,
        key_idxs: impl Iterator<Item = IdxSize>,
        track_unmatchable: bool,
    ) {
        let mut valid_idxs = Vec::new();
        let mut valid_pos = Vec::new();
        let mut num_keys = 0;
        for (p, key_idx) in key_idxs.enumerate() {
            num_keys += 1;
            if keys
                .validity
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(key_idx as usize))
            {
                valid_idxs.push(key_idx);
                valid_pos.push(p as IdxSize);
            }
        }

        let new_idx_offset = (self.idx_offset as usize).checked_add(num_keys).unwrap();
        assert!(
            new_idx_offset <= MARKED as usize,
            "overly large index in KeyRowIdxTable"
        );
        if track_unmatchable {
            self.matched.grow(new_idx_offset);
        }

        let idx_offset = self.idx_offset;
        let mut inserted = vec![false; valid_idxs.len()];
        let mut groups = Vec::with_capacity(valid_idxs.len());
        self.idx_map.get_or_insert_batch(
            keys,
            &valid_idxs,
            |r| {
                inserted[r] = true;
                unitvec![RelaxedCell::from(idx_offset + valid_pos[r])]
            },
            &mut groups,
        );
        for ((g, pos), inserted) in groups.iter().zip(&valid_pos).zip(&inserted) {
            if !inserted {
                self.idx_map
                    .value_unchecked_mut(*g)
                    .push(RelaxedCell::from(idx_offset + pos));
            }
        }

        self.idx_offset = new_idx_offset as IdxSize;
    }

    unsafe fn probe_impl<const MARK_MATCHES: bool, const EMIT_UNMATCHED: bool>(
        &self,
        keys: &KeyRowKeys,
        key_idxs: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        limit: IdxSize,
    ) -> IdxSize {
        let mut keys_processed = 0;
        let mut found = Vec::new();
        for chunk in key_idxs.chunks(PROBE_CHUNK_SIZE) {
            found.clear();
            self.idx_map.get_indices_of(keys, chunk, &mut found);
            for (key_idx, g) in chunk.iter().zip(&found) {
                if *g != IdxSize::MAX {
                    let idxs = self.idx_map.value_unchecked(*g);
                    for idx in &idxs[..] {
                        table_match.push(idx.load() & !MARKED);
                        probe_match.push(*key_idx);
                    }

                    if MARK_MATCHES {
                        unsafe { self.matched.mark_rows(idxs) };
                    }
                } else if EMIT_UNMATCHED {
                    table_match.push(IdxSize::MAX);
                    probe_match.push(*key_idx);
                }

                keys_processed += 1;
                if table_match.len() >= limit as usize {
                    return keys_processed;
                }
            }
        }
        keys_processed
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn probe_dispatch(
        &self,
        keys: &KeyRowKeys,
        key_idxs: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> IdxSize {
        match (mark_matches, emit_unmatched) {
            (false, false) => {
                self.probe_impl::<false, false>(keys, key_idxs, table_match, probe_match, limit)
            },
            (false, true) => {
                self.probe_impl::<false, true>(keys, key_idxs, table_match, probe_match, limit)
            },
            (true, false) => {
                self.probe_impl::<true, false>(keys, key_idxs, table_match, probe_match, limit)
            },
            (true, true) => {
                self.probe_impl::<true, true>(keys, key_idxs, table_match, probe_match, limit)
            },
        }
    }
}

impl IdxTable for KeyRowIdxTable {
    fn new_empty(&self) -> Box<dyn IdxTable> {
        Box::new(Self::new(self.idx_map.layout().clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_keys(&self) -> IdxSize {
        self.idx_map.len()
    }

    fn insert_keys(&mut self, hash_keys: &HashKeys, track_unmatchable: bool) {
        let HashKeys::KeyRows(keys) = hash_keys else {
            unreachable!()
        };
        unsafe { self.insert_keys_impl(keys, 0..keys.len() as IdxSize, track_unmatchable) };
    }

    unsafe fn insert_keys_subset(
        &mut self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        track_unmatchable: bool,
    ) {
        let HashKeys::KeyRows(keys) = hash_keys else {
            unreachable!()
        };
        self.insert_keys_impl(keys, subset.iter().copied(), track_unmatchable);
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
        let HashKeys::KeyRows(keys) = hash_keys else {
            unreachable!()
        };
        unsafe {
            let key_idxs: Vec<IdxSize> = (0..keys.len() as IdxSize).collect();
            self.probe_dispatch(
                keys,
                &key_idxs,
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
        let HashKeys::KeyRows(keys) = hash_keys else {
            unreachable!()
        };
        self.probe_dispatch(
            keys,
            subset,
            table_match,
            probe_match,
            mark_matches,
            emit_unmatched,
            limit,
        )
    }

    fn unmarked_keys(&self, out: &mut Vec<IdxSize>, offset: IdxSize, limit: IdxSize) -> IdxSize {
        self.matched
            .unmarked_rows(self.idx_offset, out, offset, limit)
    }
}
