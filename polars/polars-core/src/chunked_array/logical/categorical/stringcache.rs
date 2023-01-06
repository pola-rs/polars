use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use ahash::RandomState;
use hashbrown::hash_map::RawEntryMut;
use once_cell::sync::Lazy;
use polars_utils::HashSingle;
use smartstring::{LazyCompact, SmartString};

use crate::datatypes::PlIdHashMap;
use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
use crate::prelude::InitHashMaps;

/// We use atomic reference counting
/// to determine how many threads use the string cache
/// if the refcount is zero, we may clear the string cache.
pub(crate) static USE_STRING_CACHE: AtomicU32 = AtomicU32::new(0);

/// RAII for the string cache
pub struct IUseStringCache {
    // only added so that it will never be constructed directly
    #[allow(dead_code)]
    private_zst: (),
}

impl Default for IUseStringCache {
    fn default() -> Self {
        Self::new()
    }
}

impl IUseStringCache {
    /// Hold the StringCache
    pub fn new() -> IUseStringCache {
        toggle_string_cache(true);
        IUseStringCache { private_zst: () }
    }
}

impl Drop for IUseStringCache {
    fn drop(&mut self) {
        toggle_string_cache(false)
    }
}

pub fn with_string_cache<F: FnOnce() -> T, T>(func: F) -> T {
    toggle_string_cache(true);
    let out = func();
    toggle_string_cache(false);
    out
}

/// Use a global string cache for the Categorical Types.
///
/// This is used to cache the string categories locally.
/// This allows join operations on categorical types.
pub fn toggle_string_cache(toggle: bool) {
    if toggle {
        USE_STRING_CACHE.fetch_add(1, Ordering::Release);
    } else {
        let previous = USE_STRING_CACHE.fetch_sub(1, Ordering::Release);
        if previous == 0 || previous == 1 {
            USE_STRING_CACHE.store(0, Ordering::Release);
            STRING_CACHE.clear()
        }
    }
}

/// Reset the global string cache used for the Categorical Types.
pub fn reset_string_cache() {
    USE_STRING_CACHE.store(0, Ordering::Release);
    STRING_CACHE.clear()
}

/// Check if string cache is set.
pub fn using_string_cache() -> bool {
    USE_STRING_CACHE.load(Ordering::Acquire) > 0
}

// This is the hash and the Index offset in the linear buffer
#[derive(Copy, Clone)]
struct Key {
    pub(super) hash: u64,
    pub(super) idx: u32,
}

impl Key {
    #[inline]
    pub(super) fn new(hash: u64, idx: u32) -> Self {
        Self { hash, idx }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

pub(crate) struct SCacheInner {
    map: PlIdHashMap<Key, ()>,
    pub(crate) uuid: u128,
    payloads: Vec<StrHashGlobal>,
}

impl SCacheInner {
    #[inline]
    pub(crate) unsafe fn get_unchecked(&self, cat: u32) -> &str {
        self.payloads.get_unchecked(cat as usize).as_str()
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub(crate) fn insert_from_hash(&mut self, h: u64, s: &str) -> u32 {
        let mut global_idx = self.payloads.len() as u32;
        // Note that we don't create the StrHashGlobal to search the key in the hashmap
        // as StrHashGlobal may allocate a string
        let entry = self.map.raw_entry_mut().from_hash(h, |key| {
            (key.hash == h) && {
                let pos = key.idx as usize;
                let value = unsafe { self.payloads.get_unchecked(pos) };
                s == value.as_str()
            }
        });

        match entry {
            RawEntryMut::Occupied(entry) => {
                global_idx = entry.key().idx;
            }
            RawEntryMut::Vacant(entry) => {
                let idx = self.payloads.len() as u32;
                let key = Key::new(h, idx);
                entry.insert_hashed_nocheck(h, key, ());

                // only just now we allocate the string
                self.payloads.push(s.into());
            }
        }
        global_idx
    }

    #[inline]
    pub(crate) fn get_cat(&self, s: &str) -> Option<u32> {
        let h = StringCache::get_hash_builder().hash_single(s);
        // as StrHashGlobal may allocate a string
        self.map
            .raw_entry()
            .from_hash(h, |key| {
                (key.hash == h) && {
                    let pos = key.idx as usize;
                    let value = unsafe { self.payloads.get_unchecked(pos) };
                    s == value.as_str()
                }
            })
            .map(|(k, _)| k.idx)
    }

    #[inline]
    pub(crate) fn insert(&mut self, s: &str) -> u32 {
        let h = StringCache::get_hash_builder().hash_single(s);
        self.insert_from_hash(h, s)
    }
}

impl Default for SCacheInner {
    fn default() -> Self {
        Self {
            map: PlIdHashMap::with_capacity(HASHMAP_INIT_SIZE),
            #[cfg(not(target_family = "wasm"))]
            uuid: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            #[cfg(target_family = "wasm")]
            uuid: wasm_timer::SystemTime::now()
                .duration_since(wasm_timer::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            payloads: Vec::with_capacity(HASHMAP_INIT_SIZE),
        }
    }
}

/// Used by categorical data that need to share global categories.
/// In *eager* you need to specifically toggle global string cache to have a global effect.
/// In *lazy* it is toggled on at the start of a computation run and turned of (deleted) when a
/// result is produced.
#[derive(Default)]
pub(crate) struct StringCache(pub(crate) RwLock<SCacheInner>);

impl StringCache {
    /// The global `StringCache` will always use a predictable seed. This allows local builders to mimic
    /// the hashes in case of contention.
    #[inline]
    pub(crate) fn get_hash_builder() -> RandomState {
        RandomState::with_seed(0)
    }

    /// Lock the string cache
    pub(crate) fn lock_map(&self) -> RwLockWriteGuard<SCacheInner> {
        self.0.write().unwrap()
    }

    pub(crate) fn read_map(&self) -> RwLockReadGuard<SCacheInner> {
        self.0.read().unwrap()
    }

    pub(crate) fn clear(&self) {
        let mut lock = self.lock_map();
        *lock = Default::default();
    }
}

pub(crate) static STRING_CACHE: Lazy<StringCache> = Lazy::new(Default::default);

type StrHashGlobal = SmartString<LazyCompact>;
