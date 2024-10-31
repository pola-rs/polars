use std::borrow::Borrow;
use std::cell::Cell;
use std::hash::Hash;
use std::mem::MaybeUninit;

use bytemuck::allocation::zeroed_vec;
use bytemuck::Zeroable;

use crate::aliases::PlRandomState;

/// A cached function that use `FastFixedCache` for access speed.
/// It is important that the key is relatively cheap to compute.
pub struct FastCachedFunc<T, R, F> {
    func: F,
    cache: FastFixedCache<T, R>,
}

impl<T, R, F> FastCachedFunc<T, R, F>
where
    F: FnMut(T) -> R,
    T: std::hash::Hash + Eq + Clone,
    R: Copy,
{
    pub fn new(func: F, size: usize) -> Self {
        Self {
            func,
            cache: FastFixedCache::new(size),
        }
    }

    pub fn eval(&mut self, x: T, use_cache: bool) -> R {
        if use_cache {
            *self
                .cache
                .get_or_insert_with(&x, |xr| (self.func)(xr.clone()))
        } else {
            (self.func)(x)
        }
    }
}

/// A fixed-size cache optimized for access speed. Does not implement LRU or use
/// a full hash table due to cost, instead we assign two pseudorandom slots
/// based on the hash of the key, and if both are full we evict the one that had
/// the older last access.
const MIN_FAST_FIXED_CACHE_SIZE: usize = 16;

#[derive(Clone)]
pub struct FastFixedCache<K, V> {
    slots: Vec<CacheSlot<K, V>>,
    access_ctr: Cell<u32>,
    shift: u32,
    random_state: PlRandomState,
}

impl<K: Hash + Eq, V> Default for FastFixedCache<K, V> {
    fn default() -> Self {
        Self::new(MIN_FAST_FIXED_CACHE_SIZE)
    }
}

impl<K: Hash + Eq, V> FastFixedCache<K, V> {
    pub fn new(n: usize) -> Self {
        let n = (n.max(MIN_FAST_FIXED_CACHE_SIZE)).next_power_of_two();
        Self {
            slots: zeroed_vec(n),
            access_ctr: Cell::new(1),
            shift: 64 - n.ilog2(),
            random_state: PlRandomState::default(),
        }
    }

    pub fn get<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        unsafe {
            // SAFETY: slot_idx from raw_get is valid and occupied.
            let slot_idx = self.raw_get(self.hash(key), key)?;
            let slot = self.slots.get_unchecked(slot_idx);
            Some(slot.value.assume_init_ref())
        }
    }

    pub fn get_mut<Q: Hash + Eq + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        unsafe {
            // SAFETY: slot_idx from raw_get is valid and occupied.
            let slot_idx = self.raw_get(self.hash(&key), key)?;
            let slot = self.slots.get_unchecked_mut(slot_idx);
            Some(slot.value.assume_init_mut())
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> &mut V {
        unsafe { self.raw_insert(self.hash(&key), key, value) }
    }

    pub fn get_or_insert_with<Q, F>(&mut self, key: &Q, f: F) -> &mut V
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
        F: FnOnce(&K) -> V,
    {
        unsafe {
            let h = self.hash(key);
            if let Some(slot_idx) = self.raw_get(self.hash(&key), key) {
                let slot = self.slots.get_unchecked_mut(slot_idx);
                return slot.value.assume_init_mut();
            }

            let key = key.to_owned();
            let val = f(&key);
            self.raw_insert(h, key, val)
        }
    }

    pub fn try_get_or_insert_with<Q, F, E>(&mut self, key: &Q, f: F) -> Result<&mut V, E>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
        F: FnOnce(&K) -> Result<V, E>,
    {
        unsafe {
            let h = self.hash(key);
            if let Some(slot_idx) = self.raw_get(self.hash(&key), key) {
                let slot = self.slots.get_unchecked_mut(slot_idx);
                return Ok(slot.value.assume_init_mut());
            }

            let key = key.to_owned();
            let val = f(&key)?;
            Ok(self.raw_insert(h, key, val))
        }
    }

    unsafe fn raw_get<Q: Eq + ?Sized>(&self, h: HashResult, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
    {
        unsafe {
            // SAFETY: we assume h is a HashResult from self.hash with valid indices
            // and we check slot.last_access != 0 before assuming the slot is initialized.
            let slot = self.slots.get_unchecked(h.i1);
            if slot.last_access.get() != 0
                && slot.hash_tag == h.tag
                && slot.key.assume_init_ref().borrow() == key
            {
                slot.last_access.set(self.new_access_ctr());
                return Some(h.i1);
            }

            let slot = self.slots.get_unchecked(h.i2);
            if slot.last_access.get() != 0
                && slot.hash_tag == h.tag
                && slot.key.assume_init_ref().borrow() == key
            {
                slot.last_access.set(self.new_access_ctr());
                return Some(h.i2);
            }
        }

        None
    }

    unsafe fn raw_insert(&mut self, h: HashResult, key: K, value: V) -> &mut V {
        let last_access = self.new_access_ctr();
        unsafe {
            // SAFETY: i1 and i2 are valid indices and older_idx returns one of them.
            let idx = self.older_idx(h.i1, h.i2);
            let slot = self.slots.get_unchecked_mut(idx);

            // Drop impl takes care of dropping old value, if occupied.
            *slot = CacheSlot {
                last_access: Cell::new(last_access),
                hash_tag: h.tag,
                key: MaybeUninit::new(key),
                value: MaybeUninit::new(value),
            };
            slot.value.assume_init_mut()
        }
    }

    /// Returns the older index based on access time, where unoccupied slots
    /// are considered infinitely old.
    unsafe fn older_idx(&mut self, i1: usize, i2: usize) -> usize {
        let age1 = self.slots.get_unchecked(i1).last_access.get();
        let age2 = self.slots.get_unchecked(i2).last_access.get();
        match (age1, age2) {
            (0, _) => i1,
            (_, 0) => i2,
            // This takes into account the wrap-around of our access_ctr.
            // We assume that the smaller value between age1.wrapping_sub(age2)
            // and age2.wrapping_sub(age1) is the true delta. Thus if
            // age1.wrapping_sub(age2) is >= 1 << 31, we know that
            // age2.wrapping_sub(age1) is smaller than it, and we also
            // immediately know that age1 is older.
            _ if age1.wrapping_sub(age2) >= (1 << 31) => i1,
            _ => i2,
        }
    }

    fn new_access_ctr(&self) -> u32 {
        // This keeps the access_ctr always odd, so we don't hit access_ctr == 0,
        // which would leak values.
        self.access_ctr.replace(self.access_ctr.get() + 2)
    }

    /// Computes the hash tag and two slot indexes for a given key.
    fn hash<Q: Hash + ?Sized>(&self, key: &Q) -> HashResult {
        // An instantiation of Dietzfelbinger's multiply-shift, see 2.3 of
        // https://arxiv.org/pdf/1504.06804.pdf.
        // The magic constants are just two randomly chosen odd 64-bit numbers.
        let h = self.random_state.hash_one(key);
        let tag = h as u32;
        let i1 = (h.wrapping_mul(0x2e623b55bc0c9073) >> self.shift) as usize;
        let i2 = (h.wrapping_mul(0x921932b06a233d39) >> self.shift) as usize;
        HashResult { tag, i1, i2 }
    }
}

struct HashResult {
    tag: u32,
    i1: usize,
    i2: usize,
}

struct CacheSlot<K, V> {
    // If last_access != 0, the rest is assumed to be initialized.
    last_access: Cell<u32>,
    hash_tag: u32,
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
}

unsafe impl<K, V> Zeroable for CacheSlot<K, V> {}

impl<K, V> Drop for CacheSlot<K, V> {
    fn drop(&mut self) {
        unsafe {
            if self.last_access.get() != 0 {
                self.key.assume_init_drop();
                self.value.assume_init_drop();
            }
        }
    }
}

impl<K: Clone, V: Clone> Clone for CacheSlot<K, V> {
    fn clone(&self) -> Self {
        unsafe {
            if self.last_access.get() != 0 {
                Self {
                    last_access: self.last_access.clone(),
                    hash_tag: self.hash_tag,
                    key: MaybeUninit::new(self.key.assume_init_ref().clone()),
                    value: MaybeUninit::new(self.value.assume_init_ref().clone()),
                }
            } else {
                Self::zeroed()
            }
        }
    }
}
