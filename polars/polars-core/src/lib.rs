#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate core;
#[macro_use]
pub mod utils;
pub mod chunked_array;
pub mod datatypes;
#[cfg(feature = "docs")]
pub mod doc;
pub mod error;
pub mod export;
mod fmt;
pub mod frame;
pub mod functions;
mod named_from;
pub mod prelude;
pub mod schema;
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub mod serde;
pub mod series;
pub mod testing;
#[cfg(test)]
mod tests;
pub(crate) mod vector_hasher;
use once_cell::sync::Lazy;

#[cfg(any(feature = "dtype-categorical", feature = "object"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "dtype-categorical")]
use ahash::AHashMap;
use rayon::{ThreadPool, ThreadPoolBuilder};
#[cfg(feature = "dtype-categorical")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
#[cfg(feature = "dtype-categorical")]
use std::sync::MutexGuard;

#[cfg(feature = "object")]
pub(crate) static PROCESS_ID: Lazy<u128> = Lazy::new(|| {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
});

// this is re-exported in utils for polars child crates
pub static POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("POLARS_MAX_THREADS")
                .map(|s| s.parse::<usize>().expect("integer"))
                .unwrap_or_else(|_| {
                    std::thread::available_parallelism()
                        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
                        .get()
                }),
        )
        .build()
        .expect("could not spawn threads")
});

#[cfg(feature = "dtype-categorical")]
struct SCacheInner {
    map: AHashMap<String, u32>,
    uuid: u128,
}

#[cfg(feature = "dtype-categorical")]
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
#[cfg(feature = "dtype-categorical")]
pub(crate) struct StringCache(pub(crate) Mutex<SCacheInner>);

#[cfg(feature = "dtype-categorical")]
impl StringCache {
    pub(crate) fn lock_map(&self) -> MutexGuard<SCacheInner> {
        self.0.lock().unwrap()
    }

    pub(crate) fn clear(&self) {
        let mut lock = self.lock_map();
        *lock = Default::default();
    }
}

#[cfg(feature = "dtype-categorical")]
impl Default for StringCache {
    fn default() -> Self {
        StringCache(Mutex::new(Default::default()))
    }
}

#[cfg(feature = "dtype-categorical")]
pub(crate) static USE_STRING_CACHE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "dtype-categorical")]
pub(crate) static STRING_CACHE: Lazy<StringCache> = Lazy::new(Default::default);

// utility for the tests to ensure a single thread can execute
pub static SINGLE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(feature = "dtype-categorical")]
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
#[cfg(feature = "dtype-categorical")]
pub fn toggle_string_cache(toggle: bool) {
    USE_STRING_CACHE.store(toggle, Ordering::Release);

    if !toggle {
        STRING_CACHE.clear()
    }
}

/// Reset the global string cache used for the Categorical Types.
#[cfg(feature = "dtype-categorical")]
pub fn reset_string_cache() {
    STRING_CACHE.clear()
}

/// Check if string cache is set.
#[cfg(feature = "dtype-categorical")]
pub(crate) fn use_string_cache() -> bool {
    USE_STRING_CACHE.load(Ordering::Acquire)
}
