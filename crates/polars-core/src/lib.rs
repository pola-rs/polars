#![cfg_attr(docsrs, feature(doc_cfg))]
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
use std::panic::AssertUnwindSafe;
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

pub struct POOL;

// Thread locals to allow disabling threading for specific threads.
#[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
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

impl POOL {
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            op()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.install(op))
        }
    }

    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.install(|| rayon::join(oper_a, oper_b))
    }

    pub fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&rayon::Scope<'scope>) -> R + Send,
        R: Send,
    {
        self.install(|| rayon::scope(op))
    }

    pub fn spawn<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static,
    {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::spawn(op)
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| {
                p.spawn(op);
                if p.current_num_threads() == 1 {
                    p.yield_now();
                }
            })
        }
    }

    pub fn spawn_fifo<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static,
    {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::spawn_fifo(op)
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| {
                p.spawn_fifo(op);
                if p.current_num_threads() == 1 {
                    p.yield_now();
                }
            })
        }
    }

    pub fn current_thread_has_pending_tasks(&self) -> Option<bool> {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            None
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.current_thread_has_pending_tasks())
        }
    }

    pub fn current_thread_index(&self) -> Option<usize> {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::current_thread_index()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.current_thread_index())
        }
    }

    pub fn current_num_threads(&self) -> usize {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::current_num_threads()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.current_num_threads())
        }
    }

    #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
    pub fn with<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&ThreadPool) -> R + Send,
        R: Send,
    {
        if ALLOW_THREADS.get() || THREAD_POOL.current_thread_index().is_some() {
            op(&THREAD_POOL)
        } else {
            NOOP_POOL.with(|v| op(&v.borrow()))
        }
    }

    pub fn without_threading<R>(&self, op: impl FnOnce() -> R) -> R {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            op()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            // This can only be done from threads that are not in the main threadpool.
            if THREAD_POOL.current_thread_index().is_some() {
                op()
            } else {
                let prev = ALLOW_THREADS.replace(false);
                // @Q? Should this catch_unwind?
                let result = std::panic::catch_unwind(AssertUnwindSafe(op));
                ALLOW_THREADS.set(prev);
                match result {
                    Ok(v) => v,
                    Err(p) => std::panic::resume_unwind(p),
                }
            }
        }
    }
}

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
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
pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .expect("could not create pool")
});

// utility for the tests to ensure a single thread can execute
pub static SINGLE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Default length for a `.head()` call
pub(crate) const HEAD_DEFAULT_LENGTH: usize = 10;
/// Default length for a `.tail()` call
pub(crate) const TAIL_DEFAULT_LENGTH: usize = 10;
pub const CHEAP_SERIES_HASH_LIMIT: usize = 1000;
