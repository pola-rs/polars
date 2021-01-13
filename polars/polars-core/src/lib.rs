#![allow(dead_code)]
#![feature(iterator_fold_self)]
#![feature(doc_cfg)]

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
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

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
