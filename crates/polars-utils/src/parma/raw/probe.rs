/*
    To speed up probing we group table entries into groups of 8, and use
    one-byte tags to quickly skip irrelevant entries. Each tag has the following
    bit patterns:

        1000_0000 = EMPTY
        1111_1111 = DELETED
        0hhh_hhhh = OCCUPIED (with 7 bits of key hash)

    For the purposes of parallel probes, we allow false-positive probes (key
    doesn't exist or is deleted but tag is set), but not false-negative probes
    (key exists but tag is not set).
*/

use std::sync::atomic::{AtomicU64, Ordering};

const fn tag(hash: usize) -> u8 {
    (hash >> (usize::BITS - 7)) as u8
}

#[inline(always)]
const fn repeat(tag: u8) -> u64 {
    (tag as u64) * 0x01010101_01010101
}

#[repr(transparent)]
pub struct TagGroup(AtomicU64);

impl TagGroup {
    #[inline(always)]
    pub const fn all_empty() -> Self {
        Self(AtomicU64::new(repeat(0x80)))
    }

    #[inline(always)]
    pub fn all_occupied(hash: usize) -> Self {
        Self(AtomicU64::new(repeat(tag(hash))))
    }

    #[inline(always)]
    pub const fn idx_mask(num_entries: usize) -> usize {
        num_entries / 8 - 1
    }

    #[inline(always)]
    pub fn load(loc: &Self) -> Self {
        Self(AtomicU64::new(loc.0.load(Ordering::Relaxed)))
    }

    #[inline(always)]
    pub fn try_occupy(&self, cur: &mut Self, idx: usize, hash: usize) -> bool {
        let shift = 8 * idx;
        let update = ((0x80 | tag(hash)) as u64) << shift;
        let cur_val = *cur.0.get_mut();
        self.0
            .compare_exchange(
                cur_val,
                cur_val ^ update,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    #[inline(always)]
    pub fn occupy_mut(&mut self, idx: usize, hash: usize) {
        *self.0.get_mut() ^= (0x80 | tag(hash) as u64) << (8 * idx);
    }

    pub fn delete(&self, idx: usize) {
        self.0.fetch_or(0xff << (8 * idx), Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn empties(&mut self) -> TagMatches {
        let t = *self.0.get_mut();
        TagMatches(t & t.wrapping_add(repeat(1)) & repeat(0x80))
    }

    #[inline(always)]
    pub fn matches(&mut self, other: &mut TagGroup) -> TagMatches {
        let self_tags = *self.0.get_mut();
        let other_tags = *other.0.get_mut();
        TagMatches(((self_tags ^ other_tags).wrapping_sub(repeat(1))) & !self_tags & repeat(0x80))
    }
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TagMatches(u64);

impl TagMatches {
    #[inline(always)]
    pub fn has_matches(&self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub fn get(&self) -> usize {
        (self.0.trailing_zeros() / 8) as usize
    }

    pub fn has_match_at(&self, idx: usize) -> bool {
        (self.0 >> (idx * 8)) & 0x80 != 0
    }

    #[inline(always)]
    pub fn advance(&mut self) {
        self.0 &= self.0 - 1;
    }
}

impl std::ops::BitOr for TagMatches {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// A quadratic prober meant for power-of-two hash tables.
///
/// Does not reduce the index to be in-bounds, that's the responsibility of the
/// caller.
pub struct Prober {
    idx: usize,
    step: usize,
}

impl Prober {
    #[inline(always)]
    pub fn new(hash: usize) -> Self {
        Self {
            idx: hash >> 3,
            step: 0,
        }
    }

    #[inline(always)]
    pub fn get(&self) -> usize {
        self.idx
    }

    #[inline(always)]
    pub fn advance(&mut self) {
        self.step = self.step.wrapping_add(1);
        self.idx = self.idx.wrapping_add(self.step);
    }
}

pub fn max_load(num_entries: usize) -> usize {
    num_entries * 3 / 4
}

pub fn min_entries_for_load(load: usize) -> usize {
    (load * 4 / 3 + 1).next_power_of_two().max(8)
}
