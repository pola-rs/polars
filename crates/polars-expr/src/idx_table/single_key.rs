#![allow(clippy::unnecessary_cast)] // Clippy doesn't recognize that IdxSize and u64 can be different.
#![allow(unsafe_op_in_unsafe_fn)]

use hashbrown::hash_table::{Entry, HashTable};
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use polars_core::error::constants::LENGTH_LIMIT_MSG;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};

use super::*;
use crate::hash_keys::HashKeys;

/// Keys are hashed and prefetched in blocks of this many.
const BLOCK_SIZE: usize = 256;
/// Smaller tables fit in the cache and aren't prefetched.
const MIN_PREFETCH_BUCKETS: usize = 1 << 15;

/// Set in `Slot::first` once a probe key matched the slot.
const MARKED: IdxSize = 1 << (IdxSize::BITS - 1);

struct Slot<K> {
    key: K,
    /// The first build row with this key, combined with the MARKED flag.
    first: RelaxedCell<IdxSize>,
    last: IdxSize,
}

pub struct SingleKeyIdxTable<T: PolarsDataType> {
    table: HashTable<Slot<T::Physical<'static>>>,
    /// For each build row that is not the last of its key, the next build row
    /// with that key. Empty if all keys are unique.
    next: Vec<IdxSize>,
    random_state: PlRandomState,
    idx_offset: IdxSize,
    null_keys: Vec<IdxSize>,
    nulls_emitted: RelaxedCell<bool>,
}

impl<T: PolarsDataType> SingleKeyIdxTable<T> {
    pub fn new() -> Self {
        Self {
            table: HashTable::new(),
            next: Vec::new(),
            random_state: PlRandomState::default(),
            idx_offset: 0,
            null_keys: Vec::new(),
            nulls_emitted: RelaxedCell::from(false),
        }
    }
}

impl<T, K> SingleKeyIdxTable<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: TotalHash + TotalEq + Copy + Send + Sync + 'static,
{
    #[inline(always)]
    fn should_prefetch(&self) -> bool {
        self.table.num_buckets() >= MIN_PREFETCH_BUCKETS
    }

    /// Calls `f` on each build row of a slot, in insertion order.
    #[inline(always)]
    fn for_each_row(&self, first: IdxSize, last: IdxSize, mut f: impl FnMut(IdxSize)) {
        let mut idx = first & !MARKED;
        while idx != last {
            f(idx);
            idx = unsafe { *self.next.get_unchecked(idx as usize) };
        }
        f(last);
    }

    #[inline(always)]
    fn probe_one<const MARK_MATCHES: bool>(
        &self,
        key_idx: IdxSize,
        key: &K,
        hash: u64,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
    ) -> bool {
        if let Some(slot) = self.table.find(hash, |s| s.key.tot_eq(key)) {
            let first = slot.first.load();
            self.for_each_row(first, slot.last, |idx| {
                table_match.push(idx);
                probe_match.push(key_idx);
            });

            // Mark if necessary. This action is idempotent so doesn't need
            // atomic fetch_or to do it atomically.
            if MARK_MATCHES && first & MARKED == 0 {
                slot.first.store(first | MARKED);
            }
            true
        } else {
            false
        }
    }

    /// Probes the keys in blocks, first hashing (and prefetching) the whole
    /// block so the CPU can work on the lookups of several keys at once.
    ///
    /// # Safety
    /// The subset indices must be in-bounds for `arr`, and `validity` must be
    /// `Some` if `HAS_NULLS`.
    unsafe fn probe_impl<
        const MARK_MATCHES: bool,
        const EMIT_UNMATCHED: bool,
        const HAS_NULLS: bool,
    >(
        &self,
        arr: &T::Array,
        validity: Option<&Bitmap>,
        null_is_valid: bool,
        subset: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        limit: IdxSize,
    ) -> IdxSize {
        let prefetch = self.should_prefetch();
        let mut hashes = [0; BLOCK_SIZE];
        let mut keys_processed = 0;
        for block in subset.chunks(BLOCK_SIZE) {
            for (hash, key_idx) in hashes.iter_mut().zip(block) {
                *hash = self
                    .random_state
                    .tot_hash_one(arr.value_unchecked(*key_idx as usize));
                if prefetch {
                    self.table.prefetch_get(*hash);
                }
            }

            for (hash, key_idx) in hashes.iter().zip(block) {
                let key_idx = *key_idx;
                let is_valid = !HAS_NULLS
                    || validity
                        .unwrap_unchecked()
                        .get_bit_unchecked(key_idx as usize);
                let found_match = if is_valid {
                    let key = arr.value_unchecked(key_idx as usize);
                    self.probe_one::<MARK_MATCHES>(key_idx, &key, *hash, table_match, probe_match)
                } else if null_is_valid {
                    for idx in &self.null_keys {
                        table_match.push(*idx);
                        probe_match.push(key_idx);
                    }
                    if MARK_MATCHES && !self.nulls_emitted.load() {
                        self.nulls_emitted.store(true);
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
                    return keys_processed;
                }
            }
        }
        keys_processed
    }

    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    unsafe fn probe_dispatch(
        &self,
        arr: &T::Array,
        validity: Option<&Bitmap>,
        subset: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        null_is_valid: bool,
        limit: IdxSize,
    ) -> IdxSize {
        match (mark_matches, emit_unmatched, validity.is_some()) {
            (false, false, false) => self.probe_impl::<false, false, false>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (false, false, true) => self.probe_impl::<false, false, true>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (false, true, false) => self.probe_impl::<false, true, false>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (false, true, true) => self.probe_impl::<false, true, true>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (true, false, false) => self.probe_impl::<true, false, false>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (true, false, true) => self.probe_impl::<true, false, true>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (true, true, false) => self.probe_impl::<true, true, false>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
            (true, true, true) => self.probe_impl::<true, true, true>(arr, validity, null_is_valid, subset, table_match, probe_match, limit),
        }
    }
}

impl<T, K> IdxTable for SingleKeyIdxTable<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: TotalHash + TotalEq + Copy + Send + Sync + 'static,
{
    fn new_empty(&self) -> Box<dyn IdxTable> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        let random_state = &self.random_state;
        self.table
            .reserve(additional, |s| random_state.tot_hash_one(s.key));
    }

    fn num_keys(&self) -> IdxSize {
        self.table.len() as IdxSize
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
        // The top bit of each index is reserved for MARKED.
        assert!(new_idx_offset <= MARKED as usize, "{}", LENGTH_LIMIT_MSG);

        let keys: &ChunkedArray<T> = hash_keys.keys.as_phys_any().downcast_ref().unwrap();
        let arr = keys.downcast_as_array();
        let validity = arr.validity();
        let random_state = &self.random_state;
        let hasher = |s: &Slot<K>| random_state.tot_hash_one(s.key);
        let mut hashes = [0; BLOCK_SIZE];
        let mut idx = self.idx_offset;
        for block in subset.chunks(BLOCK_SIZE) {
            // Reserve up front so the table doesn't move after prefetching.
            self.table.reserve(block.len(), hasher);
            let prefetch = self.table.num_buckets() >= MIN_PREFETCH_BUCKETS;
            for (hash, subset_idx) in hashes.iter_mut().zip(block) {
                *hash = random_state.tot_hash_one(arr.value_unchecked(*subset_idx as usize));
                if prefetch {
                    self.table.prefetch_insert(*hash);
                }
            }

            for (hash, subset_idx) in hashes.iter().zip(block) {
                let subset_idx = *subset_idx as usize;
                if validity.is_some_and(|v| !v.get_bit_unchecked(subset_idx)) {
                    if track_unmatchable | hash_keys.null_is_valid {
                        self.null_keys.push(idx);
                    }
                } else {
                    let key = arr.value_unchecked(subset_idx);
                    match self.table.entry(*hash, |s| s.key.tot_eq(&key), hasher) {
                        Entry::Occupied(o) => {
                            let slot = o.into_mut();
                            if self.next.len() < new_idx_offset {
                                self.next.resize(new_idx_offset, 0);
                            }
                            *self.next.get_unchecked_mut(slot.last as usize) = idx;
                            slot.last = idx;
                        },
                        Entry::Vacant(v) => {
                            v.insert(Slot {
                                key,
                                first: RelaxedCell::from(idx),
                                last: idx,
                            });
                        },
                    }
                }
                idx += 1;
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
        let arr = keys.downcast_as_array();
        let validity = keys.has_nulls().then(|| arr.validity()).flatten();
        self.probe_dispatch(
            arr,
            validity,
            subset,
            table_match,
            probe_match,
            mark_matches,
            emit_unmatched,
            hash_keys.null_is_valid,
            limit,
        )
    }

    fn unmarked_keys(
        &self,
        out: &mut Vec<IdxSize>,
        mut offset: IdxSize,
        limit: IdxSize,
    ) -> IdxSize {
        out.clear();

        let mut keys_processed = 0;
        if !self.nulls_emitted.load() {
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

        // The offset is a bucket index, not all buckets hold a key.
        while (offset as usize) < self.table.num_buckets() {
            if let Some(slot) = self.table.get_bucket(offset as usize) {
                let first = slot.first.load();
                if first & MARKED == 0 {
                    self.for_each_row(first, slot.last, |idx| out.push(idx));
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
