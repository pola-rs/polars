use std::error::Error;
use std::future::Future;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};

use once_cell::sync::Lazy;
use polars_core::config::verbose;
use polars_core::POOL;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Semaphore;

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

#[cfg(feature = "cloud")]
pub(crate) struct Size(u64);

#[cfg(feature = "cloud")]
impl GetSize for Size {
    fn size(&self) -> u64 {
        self.0
    }
}
#[cfg(feature = "cloud")]
impl From<u64> for Size {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

enum Optimization {
    Step,
    Accept,
    Finished,
}

struct SemaphoreTuner {
    previous_download_speed: u64,
    last_tune: std::time::Instant,
    downloaded: AtomicU64,
    download_time: AtomicU64,
    opt_state: Optimization,
    increments: u32,
}

impl SemaphoreTuner {
    fn new() -> Self {
        Self {
            previous_download_speed: 0,
            last_tune: std::time::Instant::now(),
            downloaded: AtomicU64::new(0),
            download_time: AtomicU64::new(0),
            opt_state: Optimization::Step,
            increments: 0,
        }
    }
    fn should_tune(&self) -> bool {
        match self.opt_state {
            Optimization::Finished => false,
            _ => self.last_tune.elapsed().as_millis() > 350,
        }
    }

    fn add_stats(&self, downloaded_bytes: u64, download_time: u64) {
        self.downloaded
            .fetch_add(downloaded_bytes, Ordering::Relaxed);
        self.download_time
            .fetch_add(download_time, Ordering::Relaxed);
    }

    fn increment(&mut self, semaphore: &Semaphore) {
        semaphore.add_permits(1);
        self.increments += 1;
    }

    fn tune(&mut self, semaphore: &'static Semaphore) -> bool {
        let bytes_downloaded = self.downloaded.fetch_add(0, Ordering::Relaxed);
        let time_elapsed = self.download_time.fetch_add(0, Ordering::Relaxed);
        let download_speed = bytes_downloaded
            .checked_div(time_elapsed)
            .unwrap_or_default();

        let increased = download_speed > self.previous_download_speed;
        self.previous_download_speed = download_speed;
        match self.opt_state {
            Optimization::Step => {
                self.increment(semaphore);
                self.opt_state = Optimization::Accept
            },
            Optimization::Accept => {
                // Accept the step
                if increased {
                    // Set new step
                    self.increment(semaphore);
                    // Keep accept state to check next iteration
                }
                // Decline the step
                else {
                    self.opt_state = Optimization::Finished;
                    FINISHED_TUNING.store(true, Ordering::Relaxed);
                    if verbose() {
                        eprintln!(
                            "concurrency tuner finished after adding {} steps",
                            self.increments
                        )
                    }
                    // Finished.
                    return true;
                }
            },
            Optimization::Finished => {},
        }
        self.last_tune = std::time::Instant::now();
        // Not finished.
        false
    }
}
static INCR: AtomicU8 = AtomicU8::new(0);
static FINISHED_TUNING: AtomicBool = AtomicBool::new(false);
static PERMIT_STORE: std::sync::OnceLock<tokio::sync::RwLock<SemaphoreTuner>> =
    std::sync::OnceLock::new();

fn get_semaphore() -> &'static (Semaphore, u32) {
    CONCURRENCY_BUDGET.get_or_init(|| {
        let permits = std::env::var("POLARS_CONCURRENCY_BUDGET")
            .map(|s| {
                let budget = s.parse::<usize>().expect("integer");
                FINISHED_TUNING.store(true, Ordering::Relaxed);
                budget
            })
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

    if FINISHED_TUNING.load(Ordering::Relaxed) || res.size() == 0 {
        return res;
    }

    let duration = now.elapsed().as_millis() as u64;
    let permit_store = PERMIT_STORE.get_or_init(|| tokio::sync::RwLock::new(SemaphoreTuner::new()));

    let Ok(tuner) = permit_store.try_read() else {
        return res;
    };
    // Keep track of download speed
    tuner.add_stats(res.size(), duration);

    // We only tune every n ms
    if !tuner.should_tune() {
        return res;
    }
    // Drop the read tuner before trying to acquire a writer
    drop(tuner);

    // Reduce locking by letting only 1 in 5 tasks lock the tuner
    if (INCR.fetch_add(1, Ordering::Relaxed) % 5) != 0 {
        return res;
    }
    // Never lock as we will deadlock. This can run under rayon
    let Ok(mut tuner) = permit_store.try_write() else {
        return res;
    };
    let finished = tuner.tune(semaphore);
    if finished {
        drop(_permit_acq);
        // Undo the last step
        let undo = semaphore.acquire().await.unwrap();
        std::mem::forget(undo)
    }
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
}

impl RuntimeManager {
    fn new() -> Self {
        let rt = Builder::new_multi_thread()
            .worker_threads(std::cmp::max(POOL.current_num_threads(), 4))
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        Self { rt }
    }

    /// Keep track of rayon threads that drive the runtime. Every thread
    /// only allows a single runtime. If this thread calls block_on and this
    /// rayon thread is already driving an async execution we must start a new thread
    /// otherwise we panic. This can happen when we parallelize reads over 100s of files.
    ///
    /// # Safety
    /// The tokio runtime flavor is multi-threaded.
    pub fn block_on_potential_spawn<F>(&'static self, future: F) -> F::Output
    where
        F: Future + Send,
        F::Output: Send,
    {
        tokio::task::block_in_place(|| self.rt.block_on(future))
    }

    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
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

static RUNTIME: Lazy<RuntimeManager> = Lazy::new(RuntimeManager::new);

pub fn get_runtime() -> &'static RuntimeManager {
    RUNTIME.deref()
}
