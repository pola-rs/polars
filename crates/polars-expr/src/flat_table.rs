#![allow(unsafe_op_in_unsafe_fn)]

use std::mem::MaybeUninit;

use polars_utils::mem::prefetch::prefetch_l1;

/// Tags are read 8 at a time. A table has at least this many slots.
const GROUP: usize = 8;
/// Keys are hashed and prefetched in blocks of this many.
pub const BLOCK_SIZE: usize = 256;
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
pub fn slots_for(num_keys: usize) -> usize {
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

/// Open addressing table with linear probing. The caller hashes and compares
/// the keys, so a slot can hold its key or only point to it.
pub struct FlatTable<S> {
    slots: Vec<S>,
    /// One tag per slot, then a copy of the first `GROUP` tags so a group can
    /// be read at any slot. Most lookups of missing keys only read these,
    /// which is much less memory than the slots.
    tags: Vec<u8>,
}

impl<S: Copy + Default> FlatTable<S> {
    pub fn new() -> Self {
        Self {
            slots: vec![S::default(); GROUP],
            tags: vec![0; 2 * GROUP],
        }
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn resize(&mut self, num_slots: usize, hash_of: impl Fn(&S) -> u64) {
        let old_slots = std::mem::replace(&mut self.slots, vec![S::default(); num_slots]);
        let old_tags = std::mem::replace(&mut self.tags, vec![0; num_slots + GROUP]);
        for (slot, old_tag) in old_slots.into_iter().zip(old_tags) {
            if old_tag != 0 {
                let hash = hash_of(&slot);
                unsafe {
                    let pos = self.find_empty(hash);
                    self.insert(pos, hash, slot);
                }
            }
        }
    }

    /// The slot for which `eq` holds, or else the empty slot where it goes.
    #[inline(always)]
    pub unsafe fn find_slot(&self, hash: u64, eq: impl Fn(&S) -> bool) -> Result<usize, usize> {
        let num_slots = self.slots.len();
        let tag = tag(hash);
        let mut pos = home_slot(hash, num_slots);
        loop {
            let group = Group::load(&self.tags, pos);
            let mut matches = group.matches(tag);
            while matches != 0 {
                let i = slot_of(matches, pos, num_slots);
                if eq(self.slots.get_unchecked(i)) {
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

    /// The empty slot where a key that is not in the table goes.
    #[inline(always)]
    unsafe fn find_empty(&self, hash: u64) -> usize {
        let num_slots = self.slots.len();
        let mut pos = home_slot(hash, num_slots);
        loop {
            let empty = Group::load(&self.tags, pos).empty();
            if empty != 0 {
                return slot_of(empty, pos, num_slots);
            }
            pos = wrap(pos + GROUP, num_slots);
        }
    }

    /// # Safety
    /// `pos` is the empty slot `find_slot` returned for `hash`.
    #[inline(always)]
    pub unsafe fn insert(&mut self, pos: usize, hash: u64, slot: S) {
        *self.slots.get_unchecked_mut(pos) = slot;
        let tag = tag(hash);
        *self.tags.get_unchecked_mut(pos) = tag;
        if pos < GROUP {
            *self.tags.get_unchecked_mut(self.slots.len() + pos) = tag;
        }
    }

    /// # Safety
    /// `pos < self.num_slots()`.
    #[inline(always)]
    pub unsafe fn get(&self, pos: usize) -> Option<&S> {
        (*self.tags.get_unchecked(pos) != 0).then(|| self.slots.get_unchecked(pos))
    }

    /// # Safety
    /// `pos` holds a key.
    #[inline(always)]
    pub unsafe fn slot(&self, pos: usize) -> &S {
        self.slots.get_unchecked(pos)
    }

    /// # Safety
    /// `pos` holds a key.
    #[inline(always)]
    pub unsafe fn slot_mut(&mut self, pos: usize) -> &mut S {
        self.slots.get_unchecked_mut(pos)
    }

    /// Prefetches the first tags and slot of each hash.
    #[inline(always)]
    pub unsafe fn prefetch(&self, hashes: &[MaybeUninit<u64>]) {
        if self.slots.len() >= MIN_PREFETCH_SLOTS {
            for hash in hashes {
                let pos = home_slot(hash.assume_init(), self.slots.len());
                prefetch_l1(self.tags.as_ptr().add(pos));
                prefetch_l1(self.slots.as_ptr().add(pos).cast());
            }
        }
    }
}
