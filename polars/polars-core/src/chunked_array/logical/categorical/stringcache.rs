use ahash::AHashMap;
use once_cell::sync::Lazy;
use smartstring::{LazyCompact, SmartString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) static USE_STRING_CACHE: AtomicBool = AtomicBool::new(false);

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
    USE_STRING_CACHE.store(toggle, Ordering::Release);

    if !toggle {
        STRING_CACHE.clear()
    }
}

/// Reset the global string cache used for the Categorical Types.
pub fn reset_string_cache() {
    STRING_CACHE.clear()
}

/// Check if string cache is set.
pub(crate) fn use_string_cache() -> bool {
    USE_STRING_CACHE.load(Ordering::Acquire)
}

pub(crate) struct SCacheInner {
    pub(crate) map: AHashMap<SmartString<LazyCompact>, u32>,
    pub(crate) uuid: u128,
}

impl Default for SCacheInner {
    fn default() -> Self {
        Self {
            map: Default::default(),
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
