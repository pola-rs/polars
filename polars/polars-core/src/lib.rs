#[cfg_attr(docsrs, feature(doc_cfg))]
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
pub mod series;
pub mod testing;
pub(crate) mod vector_hasher;

use ahash::AHashMap;
use lazy_static::lazy_static;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

// this is re-exported in utils for polars child crates
lazy_static! {
    pub static ref POOL: ThreadPool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .expect("could not spawn threads");
}

/// Used by categorical data that need to share global categories.
/// In *eager* you need to specifically toggle global string cache to have a global effect.
/// In *lazy* it is toggled on at the start of a computation run and turned of (deleted) when a
/// result is produced.
pub(crate) struct StringCache(pub(crate) Mutex<AHashMap<String, u32>>);

impl StringCache {
    pub(crate) fn lock_map(&self) -> MutexGuard<AHashMap<String, u32>> {
        self.0.lock().unwrap()
    }

    pub(crate) fn clear(&self) {
        *self.lock_map() = AHashMap::new();
    }
}

impl Default for StringCache {
    fn default() -> Self {
        StringCache(Mutex::new(AHashMap::new()))
    }
}

thread_local! {pub(crate) static USE_STRING_CACHE: Cell<bool> = Cell::new(false)}
lazy_static! {
    static ref L_STRING_CACHE: StringCache = Default::default();
}

pub(crate) use L_STRING_CACHE as STRING_CACHE;

pub fn toggle_string_cache(toggle: bool) {
    USE_STRING_CACHE.with(|val| val.set(toggle));
    if !toggle {
        STRING_CACHE.clear()
    }
}

pub(crate) fn use_string_cache() -> bool {
    USE_STRING_CACHE.with(|val| val.get())
}
