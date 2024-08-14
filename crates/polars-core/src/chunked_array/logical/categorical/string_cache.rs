use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use hashbrown::hash_map::RawEntryMut;
use once_cell::sync::Lazy;
use polars_utils::aliases::PlRandomState;
use smartstring::{LazyCompact, SmartString};

use crate::datatypes::{InitHashMaps2, PlIdHashMap};
use crate::hashing::_HASHMAP_INIT_SIZE;

/// We use atomic reference counting to determine how many threads use the
/// string cache. If the refcount is zero, we may clear the string cache.
static STRING_CACHE_REFCOUNT: Mutex<u32> = Mutex::new(0);
static STRING_CACHE_ENABLED_GLOBALLY: AtomicBool = AtomicBool::new(false);
static STRING_CACHE_UUID_CTR: AtomicU32 = AtomicU32::new(0);

/// Enable the global string cache as long as the object is alive ([RAII]).
///
/// # Examples
///
/// Enable the string cache by initializing the object:
///
/// ```
/// use polars_core::StringCacheHolder;
///
/// let _sc = StringCacheHolder::hold();
/// ```
///
/// The string cache is enabled until `handle` is dropped.
///
/// # De-allocation
///
/// Multiple threads can hold the string cache at the same time.
/// The contents of the cache will only get dropped when no thread holds it.
///
/// [RAII]: https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization
pub struct StringCacheHolder {
    // only added so that it will never be constructed directly
    #[allow(dead_code)]
    private_zst: (),
}

impl Default for StringCacheHolder {
    fn default() -> Self {
        Self::hold()
    }
}

impl StringCacheHolder {
    /// Hold the StringCache
    pub fn hold() -> StringCacheHolder {
        increment_string_cache_refcount();
        StringCacheHolder { private_zst: () }
    }
}

impl Drop for StringCacheHolder {
    fn drop(&mut self) {
        decrement_string_cache_refcount();
    }
}

fn increment_string_cache_refcount() {
    let mut refcount = STRING_CACHE_REFCOUNT.lock().unwrap();
    *refcount += 1;
}
fn decrement_string_cache_refcount() {
    let mut refcount = STRING_CACHE_REFCOUNT.lock().unwrap();
    *refcount -= 1;
    if *refcount == 0 {
        STRING_CACHE.clear()
    }
}

/// Enable the global string cache.
///
/// [`Categorical`] columns created under the same global string cache have the
/// same underlying physical value when string values are equal. This allows the
/// columns to be concatenated or used in a join operation, for example.
///
/// Note that enabling the global string cache introduces some overhead.
/// The amount of overhead depends on the number of categories in your data.
/// It is advised to enable the global string cache only when strictly necessary.
///
/// [`Categorical`]: crate::datatypes::DataType::Categorical
pub fn enable_string_cache() {
    let was_enabled = STRING_CACHE_ENABLED_GLOBALLY.swap(true, Ordering::AcqRel);
    if !was_enabled {
        increment_string_cache_refcount();
    }
}

/// Disable and clear the global string cache.
///
/// Note: Consider using [`StringCacheHolder`] for a more reliable way of
/// enabling and disabling the string cache.
pub fn disable_string_cache() {
    let was_enabled = STRING_CACHE_ENABLED_GLOBALLY.swap(false, Ordering::AcqRel);
    if was_enabled {
        decrement_string_cache_refcount();
    }
}

/// Check whether the global string cache is enabled.
pub fn using_string_cache() -> bool {
    let refcount = STRING_CACHE_REFCOUNT.lock().unwrap();
    *refcount > 0
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
    pub(crate) uuid: u32,
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
            },
            RawEntryMut::Vacant(entry) => {
                let idx = self.payloads.len() as u32;
                let key = Key::new(h, idx);
                entry.insert_hashed_nocheck(h, key, ());

                // only just now we allocate the string
                self.payloads.push(s.into());
            },
        }
        global_idx
    }

    #[inline]
    pub(crate) fn get_cat(&self, s: &str) -> Option<u32> {
        let h = StringCache::get_hash_builder().hash_one(s);
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
        let h = StringCache::get_hash_builder().hash_one(s);
        self.insert_from_hash(h, s)
    }
}

impl Default for SCacheInner {
    fn default() -> Self {
        Self {
            map: PlIdHashMap::with_capacity(_HASHMAP_INIT_SIZE),
            uuid: STRING_CACHE_UUID_CTR.fetch_add(1, Ordering::AcqRel),
            payloads: Vec::with_capacity(_HASHMAP_INIT_SIZE),
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
    pub(crate) fn get_hash_builder() -> PlRandomState {
        PlRandomState::with_seed(0)
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

    pub(crate) fn apply<F, T>(&self, fun: F) -> (u32, T)
    where
        F: FnOnce(&mut RwLockWriteGuard<SCacheInner>) -> T,
    {
        let cache = &mut crate::STRING_CACHE.lock_map();

        let result = fun(cache);

        if cache.len() > u32::MAX as usize {
            panic!("not more than {} categories supported", u32::MAX)
        };

        (cache.uuid, result)
    }
}

pub(crate) static STRING_CACHE: Lazy<StringCache> = Lazy::new(Default::default);

type StrHashGlobal = SmartString<LazyCompact>;
