use std::future::Future;
use std::ops::Deref;
use std::sync::RwLock;
use std::thread::ThreadId;

use once_cell::sync::Lazy;
use polars_core::POOL;
use polars_utils::aliases::PlHashSet;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Semaphore;

static CONCURRENCY_BUDGET: std::sync::OnceLock<(Semaphore, u32)> = std::sync::OnceLock::new();
pub(super) const MAX_BUDGET_PER_REQUEST: usize = 10;

pub async fn with_concurrency_budget<F, Fut>(requested_budget: u32, callable: F) -> Fut::Output
where
    F: FnOnce() -> Fut,
    Fut: Future,
{
    let (semaphore, initial_budget) = CONCURRENCY_BUDGET.get_or_init(|| {
        let permits = std::env::var("POLARS_CONCURRENCY_BUDGET")
            .map(|s| s.parse::<usize>().expect("integer"))
            .unwrap_or_else(|_| std::cmp::max(POOL.current_num_threads(), MAX_BUDGET_PER_REQUEST));
        (Semaphore::new(permits), permits as u32)
    });

    // This would never finish otherwise.
    assert!(requested_budget <= *initial_budget);

    // Keep permit around.
    // On drop it is returned to the semaphore.
    let _permit_acq = semaphore.acquire_many(requested_budget).await.unwrap();
    callable().await
}

pub struct RuntimeManager {
    rt: Runtime,
    blocking_threads: RwLock<PlHashSet<ThreadId>>,
}

impl RuntimeManager {
    fn new() -> Self {
        let rt = Builder::new_multi_thread()
            .worker_threads(std::cmp::max(POOL.current_num_threads(), 4))
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
