use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::Array;
use polars_utils::idx_map::bytes_idx_map::{BytesIndexMap, Entry};
use polars_utils::idx_vec::UnitVec;
use polars_utils::itertools::Itertools;
use polars_utils::unitvec;

use super::*;
use crate::hash_keys::HashKeys;

#[derive(Default)]
pub struct RowEncodedChunkedIdxTable {
    // These AtomicU64s actually are ChunkIds, but we use the top bit of the
    // first chunk in each to mark keys during probing.
    idx_map: BytesIndexMap<UnitVec<AtomicU64>>,
    chunk_ctr: u32,
    null_keys: Vec<ChunkId<32>>,
}

impl RowEncodedChunkedIdxTable {
    pub fn new() -> Self {
        Self {
            idx_map: BytesIndexMap::new(),
            chunk_ctr: 0,
            null_keys: Vec::new(),
        }
    }
}

impl RowEncodedChunkedIdxTable {
    #[inline(always)]
    fn probe_one<const MARK_MATCHES: bool>(
        &self,
        key_idx: IdxSize,
        hash: u64,
        key: &[u8],
        table_match: &mut Vec<ChunkId<32>>,
        probe_match: &mut Vec<IdxSize>,
    ) -> bool {
        if let Some(chunk_ids) = self.idx_map.get(hash, key) {
            for chunk_id in &chunk_ids[..] {
                // Create matches, making sure to clear top bit.
                let raw_chunk_id = chunk_id.load(Ordering::Relaxed);
                let chunk_id = ChunkId::from_inner(raw_chunk_id & !(1 << 63));
                table_match.push(chunk_id);
                probe_match.push(key_idx);
            }

            // Mark if necessary. This action is idempotent so doesn't
            // need any synchronization on the load, nor does it need a
            // fetch_or to do it atomically.
            if MARK_MATCHES {
                let first_chunk_id = unsafe { chunk_ids.get_unchecked(0) };
                let first_chunk_val = first_chunk_id.load(Ordering::Relaxed);
                if first_chunk_val >> 63 == 0 {
                    first_chunk_id.store(first_chunk_val | (1 << 63), Ordering::Release);
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
        table_match: &mut Vec<ChunkId<32>>,
        probe_match: &mut Vec<IdxSize>,
        limit: IdxSize,
    ) -> IdxSize {
        table_match.clear();
        probe_match.clear();

        let mut keys_processed = 0;
        for (key_idx, hash, key) in hash_keys {
            let found_match = if let Some(key) = key {
                self.probe_one::<MARK_MATCHES>(key_idx, hash, key, table_match, probe_match)
            } else {
                false
            };

            if EMIT_UNMATCHED && !found_match {
                table_match.push(ChunkId::null());
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
        table_match: &mut Vec<ChunkId<32>>,
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

impl ChunkedIdxTable for RowEncodedChunkedIdxTable {
    fn new_empty(&self) -> Box<dyn ChunkedIdxTable> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.idx_map.reserve(additional);
    }

    fn num_keys(&self) -> IdxSize {
        self.idx_map.len()
    }

    fn insert_key_chunk(&mut self, hash_keys: HashKeys, track_unmatchable: bool) {
        let HashKeys::RowEncoded(hash_keys) = hash_keys else {
            unreachable!()
        };
        if hash_keys.keys.len() >= 1 << 31 {
            panic!("overly large chunk in RowEncodedChunkedIdxTable");
        }

        for (i, (hash, key)) in hash_keys
            .hashes
            .values_iter()
            .zip(hash_keys.keys.iter())
            .enumerate_idx()
        {
            let chunk_id = ChunkId::<32>::store(self.chunk_ctr as IdxSize, i);
            if let Some(key) = key {
                let chunk_id = AtomicU64::new(chunk_id.into_inner());
                match self.idx_map.entry(*hash, key) {
                    Entry::Occupied(o) => {
                        o.into_mut().push(chunk_id);
                    },
                    Entry::Vacant(v) => {
                        v.insert(unitvec![chunk_id]);
                    },
                }
            } else if track_unmatchable {
                self.null_keys.push(chunk_id);
            }
        }

        self.chunk_ctr = self.chunk_ctr.checked_add(1).unwrap();
    }

    fn probe(
        &self,
        hash_keys: &HashKeys,
        table_match: &mut Vec<ChunkId<32>>,
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
        table_match: &mut Vec<ChunkId<32>>,
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
        out: &mut Vec<ChunkId<32>>,
        mut offset: IdxSize,
        limit: IdxSize,
    ) -> IdxSize {
        out.clear();

        if (offset as usize) < self.null_keys.len() {
            out.extend(
                self.null_keys[offset as usize..]
                    .iter()
                    .copied()
                    .take(limit as usize),
            );
            return out.len() as IdxSize;
        }

        offset -= self.null_keys.len() as IdxSize;

        let mut keys_processed = 0;
        while let Some((_, _, chunk_ids)) = self.idx_map.get_index(offset + keys_processed) {
            let first_chunk_id = unsafe { chunk_ids.get_unchecked(0) };
            let first_chunk_val = first_chunk_id.load(Ordering::Acquire);
            if first_chunk_val >> 63 == 0 {
                for chunk_id in &chunk_ids[..] {
                    let raw_chunk_id = chunk_id.load(Ordering::Relaxed);
                    let chunk_id = ChunkId::from_inner(raw_chunk_id & !(1 << 63));
                    out.push(chunk_id);
                }
            }

            keys_processed += 1;
            if out.len() >= limit as usize {
                break;
            }
        }

        keys_processed
    }
}
