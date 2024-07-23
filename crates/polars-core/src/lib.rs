#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(ambiguous_glob_reexports)]
#![cfg_attr(feature = "nightly", allow(clippy::non_canonical_partial_ord_impl))] // remove once stable
extern crate core;

#[macro_use]
pub mod utils;
pub mod chunked_array;
pub mod config;
pub mod datatypes;
pub mod error;
pub mod export;
pub mod fmt;
pub mod frame;
pub mod functions;
pub mod hashing;
mod named_from;
pub mod prelude;
#[cfg(feature = "random")]
pub mod random;
pub mod scalar;
pub mod schema;
#[cfg(feature = "serde")]
pub mod serde;
pub mod series;
pub mod testing;
#[cfg(test)]
mod tests;

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub use hashing::IdBuildHasher;
use once_cell::sync::Lazy;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[cfg(feature = "dtype-categorical")]
pub use crate::chunked_array::logical::categorical::string_cache::*;

pub static PROCESS_ID: Lazy<u128> = Lazy::new(|| {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
});

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
pub static POOL: Lazy<ThreadPool> = Lazy::new(|| {
    let thread_name = std::env::var("POLARS_THREAD_NAME").unwrap_or_else(|_| "polars".to_string());
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
        .thread_name(move |i| format!("{}-{}", thread_name, i))
        .build()
        .expect("could not spawn threads")
});

#[cfg(target_family = "wasm")] // instead use this on wasm targets
pub static POOL: Lazy<polars_utils::wasm::Pool> = Lazy::new(|| polars_utils::wasm::Pool);

// utility for the tests to ensure a single thread can execute
pub static SINGLE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
