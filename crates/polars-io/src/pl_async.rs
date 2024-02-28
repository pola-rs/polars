use std::error::Error;
use std::future::Future;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::RwLock;
use std::thread::ThreadId;

use once_cell::sync::Lazy;
use polars_core::POOL;
use polars_utils::aliases::PlHashSet;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::{Semaphore, SemaphorePermit};

static CONCURRENCY_BUDGET: std::sync::OnceLock<(Semaphore, u32)> = std::sync::OnceLock::new();
pub(super) const MAX_BUDGET_PER_REQUEST: usize = 10;

pub trait GetSize {
    fn size(&self) -> u64;
}

impl GetSize for bytes::Bytes {
    fn size(&self) -> u64 {
        self.len() as u64
    }
}

impl<T: GetSize> GetSize for Vec<T> {
    fn size(&self) -> u64 {
        self.iter().map(|v| v.size()).sum()
    }
}

impl<T: GetSize, E: Error> GetSize for Result<T, E> {
    fn size(&self) -> u64 {
        match self {
            Ok(v) => v.size(),
            Err(_) => 0,
        }
    }
}
impl GetSize for () {
    fn size(&self) -> u64 {
        0
    }
}

#[cfg(feature = "parquet")]
impl GetSize for polars_parquet::read::FileMetaData {
    fn size(&self) -> u64 {
        0
    }
}

#[cfg(feature = "reqwest")]
impl GetSize for reqwest::Response {
    fn size(&self) -> u64 {
        0
    }
}

impl GetSize for usize {
    fn size(&self) -> u64 {
        0
    }
}

struct SemaphoreTuner {
    download_speeds: Vec<u64>,
    increments: Vec<u16>,
    permit_store: Vec<SemaphorePermit<'static>>,
    last_tune: std::time::Instant,
    downloaded: AtomicU64,
    download_time: AtomicU64,
    incr_count: u16
}

impl SemaphoreTuner {
    fn new() -> Self {
        Self {
            download_speeds: vec![],
            increments: vec![],
            permit_store: vec![],
            last_tune: std::time::Instant::now(),
            downloaded: AtomicU64::new(0),
            download_time: AtomicU64::new(0),
            incr_count: 0,
        }
    }
    fn should_tune(&self) -> bool {
        self.last_tune.elapsed().as_millis() > 500
    }

    fn add_stats(&self, downloaded_bytes: u64, download_time: u64) {
        self.downloaded
            .fetch_add(downloaded_bytes, Ordering::Relaxed);
        self.download_time
            .fetch_add(download_time, Ordering::Relaxed);
    }

    fn increment(&mut self, semaphore: &Semaphore) {


        dbg!("increment");
        if self.incr_count >= 0 {
            self.last_time_increased = true;
            if self.permit_store.is_empty() {
                semaphore.add_permits(5);
            } else {
                // Drop will send them back to the semaphore.
                let _permits = self.permit_store.pop();
            }

        }
        self.incr_count += 1;
    }

    fn decrement(&mut self, semaphore: &'static Semaphore) {
        dbg!("deccrement");
        if self.incr_count > 0 {
            self.last_time_increased = false;

            // Do not acquire here, that will deadlock
            // the `write` guard cannot be a future.
            if let Ok(permits) = semaphore.try_acquire_many(5) {
                self.permit_store.push(permits);
            }
            self.incr_count -= 1;
        }
    }

    fn tune(&mut self, semaphore: &'static Semaphore) {
        self.last_tune = std::time::Instant::now();
        let download_speed = self.downloaded.fetch_add(0, Ordering::Relaxed)
            / self.download_time.fetch_add(0, Ordering::Relaxed);
        dbg!(download_speed);
        dbg!(self.permit_store.len());
        let last_download_speed = self.last_download_speed;

        self.last_download_speed = download_speed;

        if self.last_time_increased {
            if download_speed > last_download_speed {
                self.increment(semaphore)
            } else {
                self.decrement(semaphore)
            }
        } else {
            if download_speed > last_download_speed {
                self.decrement(semaphore)
            } else {
                self.increment(semaphore)
            }
        }
    }
}
static INCR: AtomicU8 = AtomicU8::new(0);
static PERMIT_STORE: std::sync::OnceLock<tokio::sync::RwLock<SemaphoreTuner>> =
    std::sync::OnceLock::new();

fn get_semaphore() -> &'static (Semaphore, u32) {
     CONCURRENCY_BUDGET.get_or_init(|| {
        let permits = std::env::var("POLARS_CONCURRENCY_BUDGET")
            .map(|s| s.parse::<usize>().expect("integer"))
            .unwrap_or_else(|_| std::cmp::max(POOL.current_num_threads(), MAX_BUDGET_PER_REQUEST));
        (Semaphore::new(permits), permits as u32)
    })

}

pub async fn tune_with_concurrency_budget<F, Fut>(requested_budget: u32, callable: F) -> Fut::Output
where
    F: FnOnce() -> Fut,
    Fut: Future,
    Fut::Output: GetSize,
{
    let (semaphore, initial_budget) = get_semaphore();

    // This would never finish otherwise.
    assert!(requested_budget <= *initial_budget);

    // Keep permit around.
    // On drop it is returned to the semaphore.
    let _permit_acq = semaphore.acquire_many(requested_budget).await.unwrap();

    let now = std::time::Instant::now();
    let res = callable().await;

    // Don't go to tuner for the types that don't implement GetSize
    if res.size() == 0 {
        return res;
    }

    let duration = now.elapsed().as_millis() as u64;
    let permit_store = PERMIT_STORE.get_or_init(|| tokio::sync::RwLock::new(SemaphoreTuner::new()));

    let tuner = permit_store.read().await;
    // Keep track of download speed
    tuner.add_stats(res.size(), duration);

    // We only tune every n ms
    if !tuner.should_tune() {
        return res;
    }
    // Drop the read tuner before trying to acquire a writer
    drop(tuner);

    // Reduce locking by letting only 1 in 5 tasks lock the tuner
    if !((INCR.fetch_add(1, Ordering::Relaxed) % 5) == 0) {
        return res;
    }
    // Never lock as we will deadlock. This can run under rayon
    let Ok(mut tuner) = permit_store.try_write() else { return res };
    dbg!("tune");
    tuner.tune(semaphore);
    dbg!("done tuning");
    drop(tuner);
    res
}

pub async fn with_concurrency_budget<F, Fut>(requested_budget: u32, callable: F) -> Fut::Output
    where
        F: FnOnce() -> Fut,
        Fut: Future,
{
    let (semaphore, initial_budget) = get_semaphore();

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
