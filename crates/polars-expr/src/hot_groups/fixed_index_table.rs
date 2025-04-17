use polars_utils::IdxSize;
use polars_utils::select::select_unpredictable;
use polars_utils::vec::PushUnchecked;

use crate::EvictIdx;

const H2_MULT: u64 = 0xf1357aea2e62a9c5;

#[derive(Clone)]
struct Slot {
    tag: u32,
    last_access_tag: u32,
    key_index: IdxSize,
}

/// A fixed-size hash table which maps keys to indices.
///
/// Instead of growing indefinitely this table will evict keys instead.
pub struct FixedIndexTable<K> {
    slots: Vec<Slot>,
    keys: Vec<K>,
    hashes: Vec<u64>,
    shift: u8,
    prng: u64,
}

impl<K> FixedIndexTable<K> {
    pub fn new(num_slots: IdxSize) -> Self {
        assert!(num_slots.is_power_of_two());
        let empty_slot = Slot {
            tag: u32::MAX,
            last_access_tag: u32::MAX,
            key_index: IdxSize::MAX,
        };
        Self {
            slots: vec![empty_slot; num_slots as usize],
            shift: 64 - num_slots.trailing_zeros() as u8,
            keys: Vec::with_capacity(num_slots as usize),
            hashes: Vec::with_capacity(num_slots as usize),
            prng: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.key_index = IdxSize::MAX;
        }
        self.keys.clear();
        self.hashes.clear();
    }

    /// Tries to insert a key with a given hash.
    ///
    /// Returns Some((index, evict_old)) if successful, None otherwise.
    pub fn insert_key<Q, F>(&mut self, hash: u64, key: &Q, mut on_evict: F) -> Option<EvictIdx>
    where
        Q: ToOwned<Owned = K> + PartialEq<K> + ?Sized,
        F: FnMut(u64, &K),
    {
        let tag = hash as u32;
        let h1 = (hash >> self.shift) as usize;
        let h2 = (hash.wrapping_mul(H2_MULT) >> self.shift) as usize;

        unsafe {
            // We only want a single branch for the hot hit/miss check. This is
            // why we check both slots at once.
            let s1 = self.slots.get_unchecked(h1);
            let s2 = self.slots.get_unchecked(h2);
            let s1_delta = s1.tag ^ tag;
            let s2_delta = s2.tag ^ tag;
            // This check can have false positives (the binary AND of the deltas
            // happens to be zero by accident), but this is very unlikely (~1/10k)
            // and harmless if it does. False negatives are impossible. If this
            // branch succeeds we almost surely have a hit, if it fails
            // we're certain we have a miss.
            if s1_delta & s2_delta == 0 {
                // We want to branchlessly select the most likely candidate
                // first to ensure no further branch mispredicts in the vast
                // majority of cases.
                let ha = select_unpredictable(s1_delta == 0, h1, h2);
                let sa = self.slots.get_unchecked_mut(ha);
                if let Some(sak) = self.keys.get(sa.key_index as usize) {
                    if key == sak {
                        sa.last_access_tag = tag;
                        return Some(EvictIdx::new(sa.key_index, false));
                    }
                }

                // If both hashes matched we have to check the second slot too.
                if s1_delta == s2_delta {
                    let hb = h1 ^ h2 ^ ha;
                    let sb = self.slots.get_unchecked_mut(hb);
                    if let Some(sbk) = self.keys.get(sb.key_index as usize) {
                        if key == sbk {
                            sb.last_access_tag = tag;
                            return Some(EvictIdx::new(sb.key_index, false));
                        }
                    }
                }
            }

            // Check if we can insert into an empty slot.
            let num_keys = self.keys.len() as IdxSize;
            if (num_keys as usize) < self.slots.len() {
                // Check the first slot.
                let s1 = self.slots.get_unchecked_mut(h1);
                if s1.key_index >= num_keys {
                    s1.tag = tag;
                    s1.last_access_tag = tag;
                    s1.key_index = num_keys;
                    self.keys.push_unchecked(key.to_owned());
                    self.hashes.push_unchecked(hash);
                    return Some(EvictIdx::new(s1.key_index, false));
                }

                // Check the second slot.
                let s2 = self.slots.get_unchecked_mut(h2);
                if s2.key_index >= num_keys {
                    s2.tag = tag;
                    s2.last_access_tag = tag;
                    s2.key_index = num_keys;
                    self.keys.push_unchecked(key.to_owned());
                    self.hashes.push_unchecked(hash);
                    return Some(EvictIdx::new(s2.key_index, false));
                }
            }

            // Randomly try to evict one of the two slots.
            let hr = select_unpredictable(self.prng >> 63 != 0, h1, h2);
            self.prng = self.prng.wrapping_add(hash);
            let slot = self.slots.get_unchecked_mut(hr);
            if slot.last_access_tag == tag {
                slot.tag = tag;
                let evict_hash = self.hashes.get_unchecked_mut(slot.key_index as usize);
                let evict_key = self.keys.get_unchecked_mut(slot.key_index as usize);
                on_evict(*evict_hash, evict_key);
                key.clone_into(evict_key);
                *evict_hash = hash;
                Some(EvictIdx::new(slot.key_index, true))
            } else {
                slot.last_access_tag = tag;
                None
            }
        }
    }

    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }
}
