use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Once};
use std::time::{Duration, Instant};

use polars_core::config;
use polars_io::cloud::concurrency_config::{FetchConfig, get_download_chunk_size};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static SHOULD_LOG_CONCURRENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_LOG_CONCURRENCY").is_ok());

/// Value of `env_var` if set; panics if it is not a positive integer.
pub(crate) fn env_nonzero_usize(env_var: &str) -> Option<usize> {
    std::env::var(env_var).ok().map(|x| {
        x.parse::<NonZeroUsize>()
            .unwrap_or_else(|_| panic!("invalid value for {env_var}: {x}"))
            .get()
    })
}

/// Default prefetch kilobyte limit for a scan pipeline, at least one download chunk.
pub(crate) fn default_prefetch_kbytes_limit(num_pipelines: usize) -> usize {
    // This should be large enough to be non-blocking, but small enough to avoid
    // excessive memory use from a run-away prefetch pipeline.
    // "Correct" formula: (a) max effective in-flight bdp-based bytes budget + (b) decode pipeline.
    // Since we do not know (a) at startup, we use  '3 * (b) + buffer' as a proxy for (a), where the
    // multiplier reflects the max gain factor for in-flight control.
    // TODO: Dynamically adapt the max memory to the observed BDP.
    // NOTE: This does not account for the decompression multiplier, so actual memory
    // usage can be substantially larger.
    let target_chunk_size_kb = FetchConfig::random_access().chunk_size.div_ceil(1024);
    (4 * num_pipelines * target_chunk_size_kb).max(get_download_chunk_size().div_ceil(1024))
}

/// Factor by which an ordered scan grows the default count limit. `PLDEV_ORDERED_WINDOW_FACTOR`
/// overrides it.
const ORDERED_WINDOW_FACTOR: usize = 2;

fn ordered_window_factor() -> usize {
    static FACTOR: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("PLDEV_ORDERED_WINDOW_FACTOR").map_or(ORDERED_WINDOW_FACTOR, |x| {
            x.parse::<NonZeroUsize>()
                .unwrap_or_else(|_| panic!("invalid value for PLDEV_ORDERED_WINDOW_FACTOR: {x}"))
                .get()
        })
    });
    *FACTOR
}

#[derive(Clone, Debug)]
pub struct PipelineBudget {
    count: Arc<Semaphore>,
    kbytes: Arc<Semaphore>,
    count_limit: Arc<AtomicUsize>,
    kbytes_limit: usize,
    /// False if the count limit was set by an env var.
    can_grow_for_ordered: bool,
    grown_for_ordered: Arc<Once>,
    last_reported: Arc<Mutex<Instant>>,
    report_interval: Duration,
}

impl PipelineBudget {
    /// Upper bound of the default count limit, also when grown for an ordered scan.
    pub(crate) const MAX_DEFAULT_COUNT_LIMIT: usize = 2048;

    pub fn new(count_limit: usize, kbytes_limit: usize, can_grow_for_ordered: bool) -> Self {
        Self {
            count: Arc::new(Semaphore::new(count_limit)),
            kbytes: Arc::new(Semaphore::new(kbytes_limit)),
            count_limit: Arc::new(AtomicUsize::new(count_limit)),
            kbytes_limit,
            can_grow_for_ordered,
            grown_for_ordered: Arc::new(Once::new()),
            last_reported: Arc::new(Mutex::new(Instant::now())),
            report_interval: Duration::from_millis(100),
        }
    }

    pub(crate) fn count_limit(&self) -> usize {
        self.count_limit.load(Ordering::Acquire)
    }

    pub(crate) fn kbytes_limit(&self) -> usize {
        self.kbytes_limit
    }

    /// Grows the count limit by `ORDERED_WINDOW_FACTOR`, once per budget (shared by clones); a
    /// no-op if the count limit was set by an env var. Call before sizing anything from
    /// `count_limit()`.
    pub(crate) fn grow_for_ordered(&self) {
        if !self.can_grow_for_ordered {
            return;
        }
        self.grown_for_ordered.call_once(|| {
            let count_limit = self.count_limit();
            let new_count_limit = count_limit
                .saturating_mul(ordered_window_factor())
                .min(Self::MAX_DEFAULT_COUNT_LIMIT);

            self.count
                .add_permits(new_count_limit.saturating_sub(count_limit));
            self.count_limit.store(new_count_limit, Ordering::Release);

            if config::verbose() {
                eprintln!(
                    "[PipelineBudget]: ordered scan: prefetch_limit: {count_limit} -> \
                    {new_count_limit}"
                );
            }
        });
    }

    /// Acquire permit for a fetch of `n_bytes`.
    ///
    /// Acquisition order is kbytes-first, then count, so that the count_in_use
    /// value is meaningful. All pipeline paths (parquet, IPC) must acquire
    /// through this method so the order can never diverge.
    ///
    /// The requested capacity is capped at the kbytes limit, so a fetch larger than the
    /// limit still proceeds, and any positive limit makes progress.
    pub(crate) async fn acquire(&self, n_bytes: usize) -> PipelinePermit {
        let n_kbytes: u32 = self.permit_kbytes(n_bytes).try_into().unwrap_or(u32::MAX);

        // Semaphores are never closed, so acquire cannot fail.
        let _kbytes = self
            .kbytes
            .clone()
            .acquire_many_owned(n_kbytes)
            .await
            .unwrap();
        let _count = self.count.clone().acquire_owned().await.unwrap();

        if *SHOULD_LOG_CONCURRENCY {
            if let Ok(mut last_log) = self.last_reported.lock() {
                let count_limit = self.count_limit();
                let kbytes_in_use = self.kbytes_limit - self.kbytes.available_permits();
                let count_in_use = count_limit.saturating_sub(self.count.available_permits());
                if last_log.elapsed() > self.report_interval {
                    eprintln!(
                        "[PipelineBudget {}] \
                        kbytes_limit={:.1} MB, \
                        kbytes_in_use={:.1} MB, \
                        kbytes_sat={:.2}, \
                        count_limit={}, \
                        count_in_use={}, \
                        count_sat={:.2}",
                        chrono::Utc::now(),
                        self.kbytes_limit as f64 / 1e3,
                        kbytes_in_use as f64 / 1e3,
                        kbytes_in_use as f64 / self.kbytes_limit as f64,
                        count_limit,
                        count_in_use,
                        count_in_use as f64 / count_limit as f64,
                    );
                    *last_log = Instant::now();
                }
            }
        }

        PipelinePermit { _count, _kbytes }
    }

    /// Kilobytes taken by the permit for a fetch of `n_bytes`.
    pub(crate) fn permit_kbytes(&self, n_bytes: usize) -> usize {
        // Capped at the limit to prevent deadlock.
        n_bytes.div_ceil(1 << 10).min(self.kbytes_limit)
    }
}

/// RAII permit pair; releases both budgets on drop.
pub(crate) struct PipelinePermit {
    _count: OwnedSemaphorePermit,
    _kbytes: OwnedSemaphorePermit,
}
