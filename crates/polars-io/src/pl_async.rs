use std::error::Error;
use std::future::Future;
use std::sync::LazyLock;

use polars_buffer::Buffer;
use polars_core::config::{self, verbose};
use polars_core::runtime::RAYON;
use polars_utils::relaxed_cell::RelaxedCell;
use tokio::sync::Semaphore;

static CONCURRENCY_BUDGET: std::sync::OnceLock<(Semaphore, u32)> = std::sync::OnceLock::new();
pub(super) const MAX_BUDGET_PER_REQUEST: usize = 10;

/// Used to determine chunks when splitting large ranges, or combining small
/// ranges.
static DOWNLOAD_CHUNK_SIZE: LazyLock<usize> = LazyLock::new(|| {
    let v: usize = std::env::var("POLARS_DOWNLOAD_CHUNK_SIZE")
        .as_deref()
        .map(|x| x.parse().expect("integer"))
        .unwrap_or(64 * 1024 * 1024);

    if config::verbose() {
        eprintln!("async download_chunk_size: {v}")
    }

    v
});

static RANDOM_ACCESS_CHUNK_SIZE: LazyLock<usize> = LazyLock::new(|| {
    let v = std::env::var("POLARS_DOWNLOAD_CHUNK_SIZE_RANDOM_ACCESS")
        .as_deref()
        .map(|x| x.parse().expect("integer"))
        .unwrap_or(8 * 1024 * 1024);

    if config::verbose() {
        eprintln!("async download_chunk_size_random_access: {v}")
    }

    v
});

static STREAMING_CHUNK_SIZE: LazyLock<usize> = LazyLock::new(|| {
    let v = std::env::var("POLARS_DOWNLOAD_CHUNK_SIZE_STREAMING")
        .as_deref()
        .map(|x| x.parse().expect("integer"))
        .unwrap_or(32 * 1024 * 1024);

    if config::verbose() {
        eprintln!("async download_chunk_size_streaming: {v}")
    }

    v
});

pub fn get_download_chunk_size() -> usize {
    *DOWNLOAD_CHUNK_SIZE
}

pub fn get_random_access_chunk_size() -> usize {
    *RANDOM_ACCESS_CHUNK_SIZE
}
pub fn get_streaming_chunk_size() -> usize {
    *STREAMING_CHUNK_SIZE
}

static PREFETCH_MEMORY_LIMIT: LazyLock<usize> = LazyLock::new(|| {
    let v = std::env::var("POLARS_PREFETCH_MEMORY_LIMIT")
        .as_deref()
        .map(|x| x.parse().expect("integer"))
        .unwrap_or(256 * 1024 * 1024);

    if config::verbose() {
        eprintln!("async prefetch_memory_limit: {v}")
    }

    v
});

pub fn get_prefetch_memory_limit() -> usize {
    *PREFETCH_MEMORY_LIMIT
}


pub trait GetSize {
    fn size(&self) -> u64;
}

impl GetSize for Buffer<u8> {
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
    downloaded: RelaxedCell<u64>,
    download_time: RelaxedCell<u64>,
    opt_state: Optimization,
    increments: u32,
}

impl SemaphoreTuner {
    fn new() -> Self {
        Self {
            previous_download_speed: 0,
            last_tune: std::time::Instant::now(),
            downloaded: RelaxedCell::from(0),
            download_time: RelaxedCell::from(0),
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
        self.downloaded.fetch_add(downloaded_bytes);
        self.download_time.fetch_add(download_time);
    }

    fn increment(&mut self, semaphore: &Semaphore) {
        semaphore.add_permits(1);
        self.increments += 1;
    }

    fn tune(&mut self, semaphore: &'static Semaphore) -> bool {
        let bytes_downloaded = self.downloaded.load();
        let time_elapsed = self.download_time.load();
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
                    FINISHED_TUNING.store(true);
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
static INCR: RelaxedCell<u64> = RelaxedCell::new_u64(0);
static FINISHED_TUNING: RelaxedCell<bool> = RelaxedCell::new_bool(false);
static PERMIT_STORE: std::sync::OnceLock<tokio::sync::RwLock<SemaphoreTuner>> =
    std::sync::OnceLock::new();

fn get_semaphore() -> &'static (Semaphore, u32) {
    CONCURRENCY_BUDGET.get_or_init(|| {
        let permits = std::env::var("POLARS_CONCURRENCY_BUDGET")
            .map(|s| {
                let budget = s.parse::<usize>().expect("integer");
                FINISHED_TUNING.store(true);
                budget
            })
            .unwrap_or_else(|_| std::cmp::max(RAYON.current_num_threads(), MAX_BUDGET_PER_REQUEST));
        (Semaphore::new(permits), permits as u32)
    })
}

pub(crate) fn get_concurrency_limit() -> u32 {
    get_semaphore().1
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

    if FINISHED_TUNING.load() || res.size() == 0 {
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
    if !INCR.fetch_add(1).is_multiple_of(5) {
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
