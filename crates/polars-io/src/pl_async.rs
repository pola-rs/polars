use std::collections::BTreeSet;
use std::future::Future;
use std::ops::Deref;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::RwLock;
use std::time::Duration;

use once_cell::sync::Lazy;
use polars_core::POOL;
use tokio::runtime::{Builder, Runtime};

static CONCURRENCY_BUDGET: AtomicI32 = AtomicI32::new(64);

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

pub struct RuntimeManager {
    rt: Runtime,
    blocking_rayon_threads: RwLock<BTreeSet<usize>>,
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
            blocking_rayon_threads: Default::default(),
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
        if let Some(thread_id) = POOL.current_thread_index() {
            if self
                .blocking_rayon_threads
                .read()
                .unwrap()
                .contains(&thread_id)
            {
                std::thread::scope(|s| s.spawn(|| self.rt.block_on(future)).join().unwrap())
            } else {
                self.blocking_rayon_threads
                    .write()
                    .unwrap()
                    .insert(thread_id);
                let out = self.rt.block_on(future);
                self.blocking_rayon_threads
                    .write()
                    .unwrap()
                    .remove(&thread_id);
                out
            }
        }
        // Assumption that the main thread never runs rayon tasks, so we wouldn't be rescheduled
        // on the main thread and thus we can always block.
        else {
            self.rt.block_on(future)
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
