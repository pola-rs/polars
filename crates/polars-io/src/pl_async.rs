use std::future::Future;
use std::ops::Deref;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::RwLock;
use std::thread::ThreadId;
use std::time::Duration;

use once_cell::sync::Lazy;
use polars_core::POOL;
use polars_utils::aliases::PlHashSet;
use tokio::runtime::{Builder, Runtime};

static CONCURRENCY_BUDGET: AtomicI32 = AtomicI32::new(32);

pub async fn with_concurrency_budget<F, Fut>(requested_budget: u16, callable: F) -> Fut::Output
where
    F: FnOnce() -> Fut,
    Fut: Future,
{
    loop {
        let requested_budget = requested_budget as i32;
        let available_budget = CONCURRENCY_BUDGET.fetch_sub(requested_budget, Ordering::Relaxed);

        // Bail out, there was no budget
        if available_budget < 0 {
            CONCURRENCY_BUDGET.fetch_add(requested_budget, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(50)).await;
        } else {
            let fut = callable();
            let out = fut.await;
            CONCURRENCY_BUDGET.fetch_add(requested_budget, Ordering::Relaxed);

            return out;
        }
    }
}

/// Increase/decrease the concurrency budget and return the previous budget
pub fn increase_concurrency_budget(increase: i32) -> i32 {
    CONCURRENCY_BUDGET.fetch_add(increase, Ordering::Relaxed)
}

pub struct RuntimeManager {
    rt: Runtime,
    blocking_threads: RwLock<PlHashSet<ThreadId>>,
}

impl RuntimeManager {
    fn new() -> Self {
        let rt = Builder::new_multi_thread()
            .worker_threads(std::cmp::max(POOL.current_num_threads() / 2, 4))
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        Self {
            rt,
            blocking_threads: Default::default(),
        }
    }

    /// Keep track of rayon threads that drive the runtime. Every thread
    /// only allows a single runtime. If this thread calls block_on and this
    /// rayon thread is already driving an async execution we must start a new thread
    /// otherwise we panic. This can happen when we parallelize reads over 100s of files.
    pub fn block_on_potential_spawn<F>(&'static self, future: F) -> F::Output
    where
        F: Future + Send,
        F::Output: Send,
    {
        let thread_id = std::thread::current().id();

        if self.blocking_threads.read().unwrap().contains(&thread_id) {
            std::thread::scope(|s| s.spawn(|| self.rt.block_on(future)).join().unwrap())
        } else {
            self.blocking_threads.write().unwrap().insert(thread_id);
            let out = self.rt.block_on(future);
            self.blocking_threads.write().unwrap().remove(&thread_id);
            out
        }
    }

    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.rt.block_on(future)
    }
}

static RUNTIME: Lazy<RuntimeManager> = Lazy::new(RuntimeManager::new);

pub fn get_runtime() -> &'static RuntimeManager {
    RUNTIME.deref()
}
