#![cfg_attr(docsrs, feature(doc_cfg))]
#[macro_use]
pub mod utils;
pub mod chunked_array;
pub mod datatypes;
#[cfg(feature = "docs")]
pub mod doc;
pub mod error;
mod fmt;
pub mod frame;
pub mod functions;
pub mod prelude;
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub mod serde;
pub mod series;
pub mod testing;
pub(crate) mod vector_hasher;

use ahash::AHashMap;
use lazy_static::lazy_static;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

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

struct SCacheInner {
    map: AHashMap<String, u32>,
    uuid: u128,
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

pub(crate) static USE_STRING_CACHE: AtomicBool = AtomicBool::new(false);
lazy_static! {
    pub(crate) static ref STRING_CACHE: StringCache = Default::default();
}

#[cfg(test)]
lazy_static! {
    // utility for the tests to ensure a single thread can execute
    pub(crate) static ref SINGLE_LOCK: Mutex<()> = Mutex::new(());
}

use std::time::{SystemTime, UNIX_EPOCH};

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
