use polars_utils::IdxSize;
use polars_utils::select::select_unpredictable;
use polars_utils::vec::PushUnchecked;

use crate::EvictIdx;

const H2_MULT: u64 = 0xf1357aea2e62a9c5;

#[derive(Clone, Copy)]
struct Slot {
    hash: u64,
    last_access_tag: u32,
    key_index: IdxSize,
}

const EMPTY_SLOT: Slot = Slot {
    hash: 0,
    last_access_tag: u32::MAX,
    key_index: IdxSize::MAX,
};

/// A fixed-size hash table which maps keys to indices.
///
/// Instead of growing indefinitely this table will evict keys instead. It can
/// only grow by an explicit call to [`FixedIndexTable::double`].
pub struct FixedIndexTable<K> {
    slots: Vec<Slot>,
    keys: Vec<K>,
    num_filled_slots: usize, // Possibly different than keys.len() because of push_unmapped_key.
    shift: u8,
    prng: u64,
}

impl<K> FixedIndexTable<K> {
    pub fn new(num_slots: IdxSize) -> Self {
        assert!(num_slots.is_power_of_two());
        assert!(num_slots > 1);
        Self {
            slots: vec![EMPTY_SLOT; num_slots as usize],
            shift: 64 - num_slots.trailing_zeros() as u8,
            num_filled_slots: 0,
            // We add one to the capacity for the null key.
            keys: Vec::with_capacity(1 + num_slots as usize),
            prng: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Insert a key which will never be mapped to nor evicted.
    ///
    /// This is useful for permanent entries which are handled externally.
    /// Returns the key index this would have taken up.
    pub fn push_unmapped_key(&mut self, key: K) -> IdxSize {
        let idx = self.keys.len();
        self.keys.push(key);
        idx as IdxSize
    }

    /// Tries to insert a key with a given hash.
    ///
    /// A missed key is inserted even if that evicts another key when
    /// `force_insert` is set or when `next_hash`, the hash of the next key,
    /// equals `hash`.
    ///
    /// Returns Some((index, evict_old)) if successful, None otherwise.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn insert_key<Q, E, I, V>(
        &mut self,
        hash: u64,
        key: Q,
        force_insert: bool,
        next_hash: u64,
        mut eq: E,
        mut insert: I,
        mut evict_insert: V,
    ) -> Option<EvictIdx>
    where
        E: FnMut(&Q, &K) -> bool,
        I: FnMut(Q) -> K,
        V: FnMut(Q, &mut K),
    {
        let tag = hash as u32;
        let h1 = (hash >> self.shift) as usize;
        let h2 = (hash.wrapping_mul(H2_MULT) >> self.shift) as usize;

        unsafe {
            // We only want a single branch for the hot hit/miss check. This is
            // why we check both slots at once.
            let s1 = self.slots.get_unchecked(h1);
            let s2 = self.slots.get_unchecked(h2);
            let s1_delta = s1.hash ^ hash;
            let s2_delta = s2.hash ^ hash;
            // This check can have false positives (the binary AND of the deltas
            // happens to be zero by accident), but this is very unlikely and
            // harmless if it does. False negatives are impossible. If this
            // branch succeeds we almost surely have a hit, if it fails
            // we're certain we have a miss.
            if s1_delta & s2_delta == 0 {
                // We want to branchlessly select the most likely candidate
                // first to ensure no further branch mispredicts in the vast
                // majority of cases.
                let ha = select_unpredictable(s1_delta == 0, h1, h2);
                let sa = self.slots.get_unchecked_mut(ha);
                if let Some(sak) = self.keys.get(sa.key_index as usize) {
                    if eq(&key, sak) {
                        sa.last_access_tag = tag;
                        return Some(EvictIdx::new(sa.key_index, false));
                    }
                }

                // If both hashes matched we have to check the second slot too.
                if s1_delta == s2_delta {
                    let hb = h1 ^ h2 ^ ha;
                    let sb = self.slots.get_unchecked_mut(hb);
                    if let Some(sbk) = self.keys.get(sb.key_index as usize) {
                        if eq(&key, sbk) {
                            sb.last_access_tag = tag;
                            return Some(EvictIdx::new(sb.key_index, false));
                        }
                    }
                }
            }

            // Check if we can insert into an empty slot.
            let num_keys = self.keys.len() as IdxSize;
            if self.num_filled_slots < self.slots.len() {
                // Check the first slot.
                let s1 = self.slots.get_unchecked_mut(h1);
                if s1.key_index >= num_keys {
                    s1.hash = hash;
                    s1.last_access_tag = tag;
                    s1.key_index = num_keys;
                    self.keys.push_unchecked(insert(key));
                    self.num_filled_slots += 1;
                    return Some(EvictIdx::new(s1.key_index, false));
                }

                // Check the second slot.
                let s2 = self.slots.get_unchecked_mut(h2);
                if s2.key_index >= num_keys {
                    s2.hash = hash;
                    s2.last_access_tag = tag;
                    s2.key_index = num_keys;
                    self.keys.push_unchecked(insert(key));
                    self.num_filled_slots += 1;
                    return Some(EvictIdx::new(s2.key_index, false));
                }
            }

            // Randomly try to evict one of the two slots.
            let hr = select_unpredictable(self.prng >> 63 != 0, h1, h2);
            self.prng = self.prng.wrapping_add(hash);
            let slot = self.slots.get_unchecked_mut(hr);

            if (slot.last_access_tag == tag) | force_insert | (hash == next_hash) {
                slot.hash = hash;
                let evict_key = self.keys.get_unchecked_mut(slot.key_index as usize);
                evict_insert(key, evict_key);
                Some(EvictIdx::new(slot.key_index, true))
            } else {
                slot.last_access_tag = tag;
                None
            }
        }
    }

    /// Returns the slot of the given hash whose stored hash matches it, preferring the
    /// first slot, and the key index in that slot, without marking it as accessed. The
    /// key index is not below `len()` when no slot matches.
    #[inline(always)]
    pub fn find_hash(&self, hash: u64) -> (usize, IdxSize) {
        let h1 = (hash >> self.shift) as usize;
        let h2 = (hash.wrapping_mul(H2_MULT) >> self.shift) as usize;
        unsafe {
            let h = select_unpredictable(self.slots.get_unchecked(h1).hash == hash, h1, h2);
            let slot = self.slots.get_unchecked(h);
            let k = select_unpredictable(slot.hash == hash, slot.key_index, IdxSize::MAX);
            (h, k)
        }
    }

    /// Marks the key in `slot`, found for `hash`, as accessed.
    ///
    /// # Safety
    /// `slot` must be in-bounds.
    #[inline(always)]
    pub unsafe fn touch(&mut self, slot: usize, hash: u64) {
        unsafe { self.slots.get_unchecked_mut(slot).last_access_tag = hash as u32 };
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Doubles the number of slots. Every key keeps its key index and stays
    /// findable, nothing is evicted.
    pub fn double(&mut self) {
        // Both slot positions of a hash are its top bits, so after adding one
        // bit a key in slot i can always go to slot 2i or 2i + 1.
        assert!(self.shift > 1);
        let old_n = self.slots.len();
        let mut new_slots = vec![EMPTY_SLOT; 2 * old_n];
        self.shift -= 1;
        let num_keys = self.keys.len() as IdxSize;
        for (i, s) in self.slots.iter().enumerate() {
            if s.key_index >= num_keys {
                continue;
            }
            let h1 = (s.hash >> self.shift) as usize;
            let h2 = (s.hash.wrapping_mul(H2_MULT) >> self.shift) as usize;
            let dst = if h1 >> 1 == i { h1 } else { h2 };
            assert!(dst >> 1 == i);
            new_slots[dst] = *s;
        }
        self.slots = new_slots;

        // insert_key relies on this capacity, one extra for the null key.
        let need = 1 + 2 * old_n;
        if self.keys.capacity() < need {
            self.keys.reserve_exact(need - self.keys.len());
        }
    }

    pub fn keys(&self) -> &[K] {
        &self.keys
    }
}
