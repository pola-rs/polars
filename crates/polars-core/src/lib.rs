#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(ambiguous_glob_reexports)]
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
// combinations
#![cfg_attr(feature = "nightly", allow(clippy::non_canonical_partial_ord_impl))] // remove once stable
extern crate core;

#[macro_use]
pub mod utils;
pub mod chunked_array;
pub mod config;
pub mod datatypes;
pub mod error;
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

use std::cell::{Cell, RefCell};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub use datatypes::SchemaExtPl;
pub use hashing::IdBuildHasher;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub static PROCESS_ID: LazyLock<u128> = LazyLock::new(|| {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
});

// Thread locals to allow disabling threading for specific threads.
thread_local! {
    static ALLOW_THREADS: Cell<bool> = const { Cell::new(true) };
    static NOOP_POOL: RefCell<ThreadPool> = RefCell::new(
        ThreadPoolBuilder::new()
            .use_current_thread()
            .num_threads(1)
            .build()
            .expect("could not create no-op thread pool")
    );
}

pub fn with_pool<R>(op: impl FnOnce(&ThreadPool) -> R) -> R {
    if ALLOW_THREADS.get() || POOL.current_thread_index().is_some() {
        op(&POOL)
    } else {
        NOOP_POOL.with(|v| op(&v.borrow()))
    }
}

/// Try to perform the `op` without rayon threading.
pub fn without_threading<R>(op: impl FnOnce() -> R) -> R {
    // This can only be done from threads that are not in the main threadpool.
    if POOL.current_thread_index().is_some() {
        op()
    } else {
        let prev = ALLOW_THREADS.replace(false);
        // @Q? Should this catch_unwind?
        let result = op();
        ALLOW_THREADS.set(prev);
        result
    }
}

pub fn pool_num_threads() -> usize {
    if ALLOW_THREADS.get() || POOL.current_thread_index().is_some() {
        POOL.current_num_threads()
    } else {
        1
    }
}

/// Install the correct thread-pool for the current operation.
///
/// This may not actually run on multiple threads.
pub fn pool_install<OP, R>(op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    with_pool(|pool| pool.install(op))
}

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
pub static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
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
        .thread_name(move |i| format!("{thread_name}-{i}"))
        .build()
        .expect("could not spawn threads")
});

#[cfg(all(target_os = "emscripten", target_family = "wasm"))] // Use 1 rayon thread on emscripten
pub static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .expect("could not create pool")
});

#[cfg(all(not(target_os = "emscripten"), target_family = "wasm"))] // use this on other wasm targets
pub static POOL: LazyLock<polars_utils::wasm::Pool> = LazyLock::new(|| polars_utils::wasm::Pool);

// utility for the tests to ensure a single thread can execute
pub static SINGLE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Default length for a `.head()` call
pub(crate) const HEAD_DEFAULT_LENGTH: usize = 10;
/// Default length for a `.tail()` call
pub(crate) const TAIL_DEFAULT_LENGTH: usize = 10;
pub const CHEAP_SERIES_HASH_LIMIT: usize = 1000;
