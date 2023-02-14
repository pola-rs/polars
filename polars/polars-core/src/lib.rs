#![cfg_attr(docsrs, feature(doc_auto_cfg))]
extern crate core;

#[macro_use]
pub mod utils;
pub mod chunked_array;
pub mod cloud;
pub mod config;
pub mod datatypes;
#[cfg(feature = "docs")]
pub mod doc;
pub mod error;
pub mod export;
pub mod fmt;
pub mod frame;
pub mod functions;
mod named_from;
pub mod prelude;
pub mod schema;
#[cfg(feature = "serde")]
pub mod serde;
pub mod series;
pub mod testing;
#[cfg(test)]
mod tests;
pub(crate) mod vector_hasher;

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[cfg(feature = "dtype-categorical")]
pub use crate::chunked_array::logical::categorical::stringcache::*;

pub static PROCESS_ID: Lazy<u128> = Lazy::new(|| {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
});

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
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

#[cfg(target_family = "wasm")] // instead use this on wasm targets
pub static POOL: Lazy<polars_utils::wasm::Pool> = Lazy::new(|| polars_utils::wasm::Pool);

// utility for the tests to ensure a single thread can execute
pub static SINGLE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
