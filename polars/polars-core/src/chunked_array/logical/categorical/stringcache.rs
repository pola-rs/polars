use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use ahash::RandomState;
use once_cell::sync::Lazy;
use smartstring::{LazyCompact, SmartString};

use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
use crate::prelude::PlHashMap;

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

pub(crate) struct SCacheInner {
    pub(crate) map: PlHashMap<StrHashGlobal, u32>,
    pub(crate) uuid: u128,
}

impl Default for SCacheInner {
    fn default() -> Self {
        Self {
            map: PlHashMap::with_capacity_and_hasher(
                HASHMAP_INIT_SIZE,
                StringCache::get_hash_builder(),
            ),
            uuid: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        }
    }
}

/// Used by categorical data that need to share global categories.
/// In *eager* you need to specifically toggle global string cache to have a global effect.
/// In *lazy* it is toggled on at the start of a computation run and turned of (deleted) when a
/// result is produced.
pub(crate) struct StringCache(pub(crate) Mutex<SCacheInner>);

impl StringCache {
    /// The global `StringCache` will always use a predictable seed. This allows local builders to mimic
    /// the hashes in case of contention.
    pub(crate) fn get_hash_builder() -> RandomState {
        RandomState::with_seed(0)
    }

    /// Lock the string cache
    pub(crate) fn lock_map(&self) -> MutexGuard<SCacheInner> {
        self.0.lock().unwrap()
    }

    pub(crate) fn clear(&self) {
        let mut lock = self.lock_map();
        *lock = Default::default();
    }
}

impl Default for StringCache {
    fn default() -> Self {
        StringCache(Mutex::new(Default::default()))
    }
}

pub(crate) static STRING_CACHE: Lazy<StringCache> = Lazy::new(Default::default);

#[derive(Eq, Clone)]
pub struct StrHashGlobal {
    pub(crate) str: SmartString<LazyCompact>,
    pub(crate) hash: u64,
}

impl Hash for StrHashGlobal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

impl StrHashGlobal {
    pub(crate) fn new(s: SmartString<LazyCompact>, hash: u64) -> Self {
        Self { str: s, hash }
    }
}

impl PartialEq for StrHashGlobal {
    fn eq(&self, other: &Self) -> bool {
        // can be collisions in the hashtable even though the hashes are equal
        // e.g. hashtable hash = hash % n_slots
        (self.hash == other.hash) && (self.str == other.str)
    }
}

impl Borrow<str> for StrHashGlobal {
    fn borrow(&self) -> &str {
        self.str.as_str()
    }
}
