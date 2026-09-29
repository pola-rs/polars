#![allow(unsafe_op_in_unsafe_fn)]

use std::mem::MaybeUninit;

use polars_arrow::bitmap::Bitmap;
use polars_utils::mem::prefetch::prefetch_l1;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalEq, TotalHash};

use super::*;
use crate::hash_keys::HashKeys;

/// Tags are read 8 at a time. A table has at least this many slots.
const GROUP: usize = 8;
/// Keys are hashed and prefetched in blocks of this many.
const BLOCK_SIZE: usize = 256;
/// Smaller tables fit in the cache and aren't prefetched.
const MIN_PREFETCH_SLOTS: usize = 1 << 15;

/// A byte with 7 bits of the hash, never 0 as that marks an empty slot. It
/// uses the low bits, as the slot position comes from the high bits.
#[inline(always)]
fn tag(hash: u64) -> u8 {
    hash as u8 | 0x80
}

/// The first slot to look at for `hash`.
#[inline(always)]
fn home_slot(hash: u64, num_slots: usize) -> usize {
    ((hash as u128 * num_slots as u128) >> 64) as usize
}

#[inline(always)]
fn wrap(pos: usize, num_slots: usize) -> usize {
    if pos >= num_slots {
        pos - num_slots
    } else {
        pos
    }
}

/// The slots needed for `num_keys` keys, as the table is at most 3/4 full.
fn slots_for(num_keys: usize) -> usize {
    (num_keys * 4).div_ceil(3).max(GROUP)
}

const LO: u64 = u64::from_ne_bytes([0x01; 8]);
const HI: u64 = u64::from_ne_bytes([0x80; 8]);

/// The tags of 8 slots in one word, the first slot in the lowest byte.
#[derive(Clone, Copy)]
struct Group(u64);

impl Group {
    /// # Safety
    /// `pos + GROUP <= tags.len()`.
    #[inline(always)]
    unsafe fn load(tags: &[u8], pos: usize) -> Self {
        Self(u64::from_le(
            tags.as_ptr().add(pos).cast::<u64>().read_unaligned(),
        ))
    }

    /// The top bit of each byte whose slot is empty.
    #[inline(always)]
    fn empty(self) -> u64 {
        !self.0 & HI
    }

    /// The top bit of each byte whose tag may equal `tag`. Never an empty
    /// slot, but there can be false matches, which comparing the keys rules out.
    #[inline(always)]
    fn matches(self, tag: u8) -> u64 {
        let x = self.0 ^ (LO * tag as u64);
        x.wrapping_sub(LO) & !x & HI
    }
}

/// The slot of the byte of the lowest top bit in `bits`, for a group at `pos`.
#[inline(always)]
fn slot_of(bits: u64, pos: usize, num_slots: usize) -> usize {
    wrap(pos + (bits.trailing_zeros() / 8) as usize, num_slots)
}

#[derive(Clone, Copy, Default)]
struct Slot<K> {
    key: K,
    /// The first and last build row with this key.
    first: IdxSize,
    last: IdxSize,
}

/// Open addressing table with linear probing.
pub struct SingleKeyIdxTable<T: PolarsDataType> {
    slots: Vec<Slot<T::Physical<'static>>>,
    /// One tag per slot, then a copy of the first `GROUP` tags so a group can
    /// be read at any slot. Most lookups of missing keys only read these,
    /// which is much less memory than the slots.
    tags: Vec<u8>,
    /// One flag per slot, set once a probe key matched that slot.
    marked: Vec<RelaxedCell<bool>>,
    /// For each build row, the next build row with the same key. Only rows
    /// that are not the last of their key are set. Empty if all keys are unique.
    next: Vec<IdxSize>,
    num_keys: usize,
    random_state: PlRandomState,
    idx_offset: IdxSize,
    null_keys: Vec<IdxSize>,
    nulls_emitted: RelaxedCell<bool>,
}

impl<T, K> SingleKeyIdxTable<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: TotalHash + TotalEq + Copy + Default + Send + Sync + 'static,
{
    pub fn new() -> Self {
        let mut out = Self {
            slots: Vec::new(),
            tags: Vec::new(),
            marked: Vec::new(),
            next: Vec::new(),
            num_keys: 0,
            random_state: PlRandomState::default(),
            idx_offset: 0,
            null_keys: Vec::new(),
            nulls_emitted: RelaxedCell::from(false),
        };
        out.resize(GROUP);
        out
    }

    /// Clears the marks, which is fine as they are only set after building.
    fn resize(&mut self, num_slots: usize) {
        let old_slots = std::mem::replace(&mut self.slots, vec![Slot::default(); num_slots]);
        let old_tags = std::mem::replace(&mut self.tags, vec![0; num_slots + GROUP]);
        self.marked = (0..num_slots).map(|_| RelaxedCell::from(false)).collect();
        for (slot, old_tag) in old_slots.into_iter().zip(old_tags) {
            if old_tag != 0 {
                let hash = self.random_state.tot_hash_one(slot.key);
                unsafe {
                    let Err(pos) = self.find_slot(hash, &slot.key) else {
                        std::hint::unreachable_unchecked()
                    };
                    *self.slots.get_unchecked_mut(pos) = slot;
                    self.set_tag(pos, tag(hash));
                }
            }
        }
    }

    /// The slot holding `key`, or else the empty slot where it goes.
    #[inline(always)]
    unsafe fn find_slot(&self, hash: u64, key: &K) -> Result<usize, usize> {
        let num_slots = self.slots.len();
        let tag = tag(hash);
        let mut pos = home_slot(hash, num_slots);
        loop {
            let group = Group::load(&self.tags, pos);
            let mut matches = group.matches(tag);
            while matches != 0 {
                let i = slot_of(matches, pos, num_slots);
                if self.slots.get_unchecked(i).key.tot_eq(key) {
                    return Ok(i);
                }
                matches &= matches - 1;
            }
            let empty = group.empty();
            if empty != 0 {
                return Err(slot_of(empty, pos, num_slots));
            }
            pos = wrap(pos + GROUP, num_slots);
        }
    }

    #[inline(always)]
    unsafe fn set_tag(&mut self, pos: usize, tag: u8) {
        *self.tags.get_unchecked_mut(pos) = tag;
        if pos < GROUP {
            *self.tags.get_unchecked_mut(self.slots.len() + pos) = tag;
        }
    }

    /// Prefetches the first tags and slot of each hash.
    #[inline(always)]
    unsafe fn prefetch(&self, hashes: &[MaybeUninit<u64>]) {
        if self.slots.len() >= MIN_PREFETCH_SLOTS {
            for hash in hashes {
                let pos = home_slot(hash.assume_init(), self.slots.len());
                prefetch_l1(self.tags.as_ptr().add(pos));
                prefetch_l1(self.slots.as_ptr().add(pos).cast());
            }
        }
    }

    /// The build rows of the key in this slot.
    #[inline(always)]
    fn rows(&self, slot: &Slot<K>) -> impl Iterator<Item = IdxSize> + '_ {
        let (first, last) = (slot.first, slot.last);
        std::iter::successors(Some(first), move |&idx| {
            (idx != last).then(|| unsafe { *self.next.get_unchecked(idx as usize) })
        })
    }

    /// Probes the keys in blocks. Each block first hashes all its keys, so
    /// the CPU can work on the lookups of several keys at once.
    #[allow(clippy::too_many_arguments)]
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
        let limit = limit as usize;
        let mut hashes = [const { MaybeUninit::<u64>::uninit() }; BLOCK_SIZE];
        let mut keys_processed = 0;
        for block in subset.chunks(BLOCK_SIZE) {
            for (hash, key_idx) in hashes.iter_mut().zip(block) {
                let key = arr.value_unchecked(*key_idx as usize);
                hash.write(self.random_state.tot_hash_one(key));
            }
            self.prefetch(&hashes[..block.len()]);

            // Keys with one output row can only reach the limit if the whole
            // block can. Keys with more rows always check the limit.
            let mut check_each = limit.saturating_sub(table_match.len()) <= block.len();
            table_match.reserve(block.len());
            probe_match.reserve(block.len());
            // probe_match can hold earlier matches, so its length differs
            // from table_match by a fixed amount.
            let probe_offset = probe_match.len() - table_match.len();
            let mut len = table_match.len();
            for (i, (hash, key_idx)) in hashes.iter().zip(block).enumerate() {
                let key_idx = *key_idx;
                let is_valid = !HAS_NULLS
                    || validity
                        .unwrap_unchecked()
                        .get_bit_unchecked(key_idx as usize);
                // A key either writes one row directly, or pushes many rows.
                let mut single_row = None;
                let mut many_rows = false;
                if is_valid {
                    let key = arr.value_unchecked(key_idx as usize);
                    if let Ok(pos) = self.find_slot(hash.assume_init(), &key) {
                        let slot = self.slots.get_unchecked(pos);
                        if slot.first == slot.last {
                            single_row = Some(slot.first);
                        } else {
                            table_match.set_len(len);
                            probe_match.set_len(probe_offset + len);
                            for row in self.rows(slot) {
                                table_match.push(row);
                                probe_match.push(key_idx);
                            }
                            many_rows = true;
                        }

                        // Mark if necessary. This action is idempotent so doesn't need
                        // atomic fetch_or to do it atomically.
                        if MARK_MATCHES {
                            let marked = self.marked.get_unchecked(pos);
                            if !marked.load() {
                                marked.store(true);
                            }
                        }
                    }
                } else if null_is_valid {
                    if MARK_MATCHES && !self.nulls_emitted.load() {
                        self.nulls_emitted.store(true);
                    }
                    if let [row] = self.null_keys.as_slice() {
                        single_row = Some(*row);
                    } else if !self.null_keys.is_empty() {
                        table_match.set_len(len);
                        probe_match.set_len(probe_offset + len);
                        table_match.extend_from_slice(&self.null_keys);
                        probe_match.extend(self.null_keys.iter().map(|_| key_idx));
                        many_rows = true;
                    }
                }

                if many_rows {
                    // Make sure there is room for the rest of the block again.
                    table_match.reserve(block.len());
                    probe_match.reserve(block.len());
                    len = table_match.len();
                    if len >= limit {
                        return (keys_processed + i + 1) as IdxSize;
                    }
                    check_each = limit - len <= block.len();
                } else {
                    if EMIT_UNMATCHED && single_row.is_none() {
                        single_row = Some(IdxSize::MAX);
                    }
                    if let Some(row) = single_row {
                        *table_match.as_mut_ptr().add(len) = row;
                        *probe_match.as_mut_ptr().add(probe_offset + len) = key_idx;
                        len += 1;
                        if check_each && len >= limit {
                            table_match.set_len(len);
                            probe_match.set_len(probe_offset + len);
                            return (keys_processed + i + 1) as IdxSize;
                        }
                    }
                }
            }
            table_match.set_len(len);
            probe_match.set_len(probe_offset + len);
            keys_processed += block.len();
        }
        keys_processed as IdxSize
    }
}

impl<T, K> IdxTable for SingleKeyIdxTable<T>
where
    for<'a> T: PolarsDataType<Physical<'a> = K>,
    K: TotalHash + TotalEq + Copy + Default + Send + Sync + 'static,
{
    fn new_empty(&self) -> Box<dyn IdxTable> {
        Box::new(Self::new())
    }

    fn reserve(&mut self, additional: usize) {
        let num_slots = slots_for(self.num_keys + additional);
        if num_slots > self.slots.len() {
            self.resize(num_slots);
        }
    }

    fn num_keys(&self) -> IdxSize {
        self.num_keys as IdxSize
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
        let arr = keys.downcast_as_array();
        let validity = polars_arrow::array::Array::validity(arr);
        let mut hashes = [const { MaybeUninit::<u64>::uninit() }; BLOCK_SIZE];
        let mut idx = self.idx_offset;
        for block in subset.chunks(BLOCK_SIZE) {
            // Grows for the whole block at once. Nulls and duplicates don't
            // take a slot, so this can grow a bit early.
            let num_slots = slots_for(self.num_keys + block.len());
            if num_slots > self.slots.len() {
                self.resize(num_slots.max(self.slots.len() * 2));
            }
            for (hash, subset_idx) in hashes.iter_mut().zip(block) {
                let key = arr.value_unchecked(*subset_idx as usize);
                hash.write(self.random_state.tot_hash_one(key));
            }
            self.prefetch(&hashes[..block.len()]);

            for (hash, subset_idx) in hashes.iter().zip(block) {
                let subset_idx = *subset_idx as usize;
                if validity.is_some_and(|v| !v.get_bit_unchecked(subset_idx)) {
                    if track_unmatchable | hash_keys.null_is_valid {
                        self.null_keys.push(idx);
                    }
                } else {
                    let key = arr.value_unchecked(subset_idx);
                    let hash = hash.assume_init();
                    match self.find_slot(hash, &key) {
                        Ok(pos) => {
                            let slot = self.slots.get_unchecked_mut(pos);
                            let last = slot.last as usize;
                            slot.last = idx;
                            if last >= self.next.len() {
                                self.next.resize(new_idx_offset, 0);
                            }
                            *self.next.get_unchecked_mut(last) = idx;
                        },
                        Err(pos) => {
                            *self.slots.get_unchecked_mut(pos) = Slot {
                                key,
                                first: idx,
                                last: idx,
                            };
                            self.set_tag(pos, tag(hash));
                            self.num_keys += 1;
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
        let validity = polars_arrow::array::Array::validity(arr);
        let null_is_valid = hash_keys.null_is_valid;
        macro_rules! dispatch {
            ($($m:literal, $e:literal, $n:literal);*) => {
                match (mark_matches, emit_unmatched, keys.has_nulls()) {
                    $(($m, $e, $n) => self.probe_impl::<$m, $e, $n>(
                        arr,
                        validity,
                        null_is_valid,
                        subset,
                        table_match,
                        probe_match,
                        limit,
                    ),)*
                }
            };
        }
        dispatch!(
            false, false, false; false, false, true; false, true, false; false, true, true;
            true, false, false; true, false, true; true, true, false; true, true, true
        )
    }

    fn unmarked_keys(&self, out: &mut Vec<IdxSize>, mut offset: usize, limit: IdxSize) -> usize {
        out.clear();

        let mut keys_processed = 0;
        if !self.nulls_emitted.load() {
            if offset < self.null_keys.len() {
                out.extend(
                    self.null_keys[offset..]
                        .iter()
                        .copied()
                        .take(limit as usize),
                );
                keys_processed += out.len();
                offset += out.len();
                if out.len() >= limit as usize {
                    return keys_processed;
                }
            }
            offset -= self.null_keys.len();
        }

        // The offset is a slot index, not all slots hold a key.
        while offset < self.slots.len() {
            let slot = unsafe { self.slots.get_unchecked(offset) };
            let tag = unsafe { *self.tags.get_unchecked(offset) };
            let marked = unsafe { self.marked.get_unchecked(offset) };
            if tag != 0 && !marked.load() {
                out.extend(self.rows(slot));
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

#[cfg(test)]
mod tests {
    use polars_utils::aliases::PlHashMap;

    use super::*;

    #[test]
    fn grows_while_building_and_resumes_at_the_limit() {
        let n = 100_000;
        let keys: Vec<Option<i64>> = (0..n)
            .map(|i| (i % 7 != 0).then_some(if i % 256 == 128 { 0 } else { i }))
            .collect();
        let df = DataFrame::new_infer_height(vec![Column::new("k".into(), &keys)]).unwrap();
        let hash_keys = HashKeys::from_df(&df, PlRandomState::default(), false, false);
        let all: Vec<IdxSize> = (0..n as IdxSize).collect();

        // No reserve, so the table grows while it holds keys.
        let mut table = SingleKeyIdxTable::<Int64Type>::new();
        for chunk in all.chunks(10_000) {
            unsafe { table.insert_keys_subset(&hash_keys, chunk, false) };
        }

        let mut rows_per_key = PlHashMap::<i64, Vec<IdxSize>>::new();
        for (row, key) in keys.iter().enumerate() {
            if let Some(key) = key {
                rows_per_key.entry(*key).or_default().push(row as IdxSize);
            }
        }
        assert_eq!(table.num_keys() as usize, rows_per_key.len());
        let mut expected: Vec<(IdxSize, IdxSize)> = keys
            .iter()
            .enumerate()
            .filter_map(|(row, key)| Some((row as IdxSize, rows_per_key.get(key.as_ref()?)?)))
            .flat_map(|(row, rows)| rows.iter().map(move |r| (row, *r)))
            .collect();
        expected.sort_unstable();

        for limit in [IdxSize::MAX, 500] {
            let mut out = Vec::new();
            let mut offset = 0;
            while offset < all.len() {
                let mut table_match = Vec::new();
                let mut probe_match = Vec::new();
                offset += unsafe {
                    table.probe_subset(
                        &hash_keys,
                        &all[offset..],
                        &mut table_match,
                        &mut probe_match,
                        false,
                        false,
                        limit,
                    )
                } as usize;
                // Only the rows of the last key can go past the limit.
                if let Some(last) = probe_match.last() {
                    let before_last = probe_match.iter().take_while(|p| *p != last).count();
                    assert!(before_last < limit as usize);
                }
                out.extend(probe_match.into_iter().zip(table_match));
            }
            out.sort_unstable();
            assert_eq!(out, expected);
        }
    }
}
