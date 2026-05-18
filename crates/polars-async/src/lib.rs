use std::sync::{Arc, LazyLock};

use polars_error::polars_warn;
use polars_utils::relaxed_cell::RelaxedCell;
use tokio::runtime::{Builder, Runtime};

use crate::executor::{THREAD_SPAWNED_BY_POLARS_EXECUTOR, is_scheduling_polars_executor_thread};

pub mod executor;
pub mod primitives;

pub struct RuntimeManager {
    rt: Runtime,
}

impl RuntimeManager {
    fn new() -> Self {
        let n_threads = std::env::var("POLARS_ASYNC_THREAD_COUNT")
            .map(|x| x.parse::<usize>().expect("integer"))
            .unwrap_or(usize::min(polars_config::config().max_threads(), 32));

        let max_blocking = std::env::var("POLARS_MAX_BLOCKING_THREAD_COUNT")
            .map(|x| x.parse::<usize>().expect("integer"))
            .unwrap_or(512);

        if polars_config::config().verbose() {
            eprintln!("async thread count: {n_threads}");
            eprintln!("blocking thread count: {max_blocking}");
        }

        let max_total_threads = n_threads + max_blocking;
        let warned = RelaxedCell::new_bool(false);
        let tokio_thread_count_start = Arc::new(RelaxedCell::new_i64(0));
        let tokio_thread_count_stop = tokio_thread_count_start.clone();

        let rt = Builder::new_multi_thread()
            .worker_threads(n_threads)
            .max_blocking_threads(max_blocking)
            .on_thread_start(move || {
                if tokio_thread_count_start.fetch_add(1) + 1 >= (max_total_threads as i64) && !warned.load() {
                    warned.store(true);
                    polars_warn!("POLARS_MAX_BLOCKING_THREAD_COUNT reached ({max_blocking}), this may indicate a deadlock");
                }
            })
            .on_thread_stop(move || { tokio_thread_count_stop.fetch_sub(1); })
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        Self { rt }
    }

    /// Runs the given function on this thread while allowing another thread to take
    /// over this thread's task execution duties.
    ///
    /// Simply directly calls f() if this thread is not an async executor thread.
    pub fn block_in_place<R, F: FnOnce() -> R>(f: F) -> R {
        if THREAD_SPAWNED_BY_POLARS_EXECUTOR.get() {
            executor::block_in_place(f)
        } else {
            tokio::task::block_in_place(f)
        }
    }

    /// Blocks this thread to evaluate the given future.
    ///
    /// This is more expensive than block_on when called from an async runtime
    /// worker thread because other async tasks scheduled to run on this thread
    /// have to be moved to a new thread.
    ///
    /// If more than POLARS_MAX_BLOCKING_THREAD_COUNT calls to this occur
    /// simultaneously a deadlock may occur.
    pub fn block_in_place_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        if THREAD_SPAWNED_BY_POLARS_EXECUTOR.get() {
            executor::block_in_place(|| self.rt.block_on(future))
        } else {
            tokio::task::block_in_place(|| self.rt.block_on(future))
        }
    }

    /// Blocks this thread to evaluate the given future.
    ///
    /// Panics if the current thread is an async runtime worker thread.
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        assert!(
            !is_scheduling_polars_executor_thread(),
            "block_on may not be called from within a polars async executor runtime worker thread"
        );
        self.rt.block_on(future)
    }

    /// Spawns a future onto the Tokio runtime (see [`tokio::runtime::Runtime::spawn`]).
    pub fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(future)
    }

    // See [`tokio::runtime::Runtime::spawn_blocking`].
    pub fn spawn_blocking<F, R>(&self, f: F) -> tokio::task::JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.rt.spawn_blocking(f)
    }
}

/// The global Polars async runtime accessor.
pub static ASYNC: LazyLock<RuntimeManager> = LazyLock::new(RuntimeManager::new);
