#![cfg_attr(docsrs, feature(doc_cfg))]
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
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub mod serde;
pub mod series;
pub mod testing;
#[cfg(test)]
mod tests;
#[cfg(feature = "temporal")]
pub mod time;
pub(crate) mod vector_hasher;

#[cfg(any(feature = "dtype-categorical", feature = "object"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "dtype-categorical")]
use ahash::AHashMap;
use lazy_static::lazy_static;
use rayon::{ThreadPool, ThreadPoolBuilder};
#[cfg(feature = "dtype-categorical")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "dtype-categorical")]
use std::sync::{Mutex, MutexGuard};

#[cfg(feature = "object")]
lazy_static! {
    pub(crate) static ref PROCESS_ID: u128 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
}

// this is re-exported in utils for polars child crates
lazy_static! {
    pub static ref POOL: ThreadPool = ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("POLARS_MAX_THREADS")
                .map(|s| s.parse::<usize>().expect("integer"))
                .unwrap_or_else(|_| num_cpus::get())
        )
        .build()
        .expect("could not spawn threads");
}

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
lazy_static! {
    pub(crate) static ref STRING_CACHE: StringCache = Default::default();
}

#[cfg(test)]
#[cfg(feature = "dtype-categorical")]
lazy_static! {
    // utility for the tests to ensure a single thread can execute
    pub(crate) static ref SINGLE_LOCK: Mutex<()> = Mutex::new(());
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
