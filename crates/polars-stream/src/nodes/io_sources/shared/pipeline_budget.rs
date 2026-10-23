use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use polars_core::config;
use polars_io::cloud::concurrency::{InFlightBudget, get_inflight_request_budget};
use polars_io::cloud::concurrency_config::{FetchConfig, get_download_chunk_size};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static SHOULD_LOG_CONCURRENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_LOG_CONCURRENCY").is_ok());

/// Count limit of an ordered scan's pipeline with an in-flight budget. Its window is sized in bytes;
/// this only bounds tasks and queue slots for fetches smaller than the window / this limit.
const ORDERED_COUNT_LIMIT: usize = 8192;

/// Factor by which an ordered scan without an in-flight budget grows the default count limit.
const ORDERED_LOCAL_COUNT_FACTOR: usize = 2;

const DEFAULT_ORDERED_SLACK_MS: u64 = 100;

/// Reorder headroom of an ordered scan's window, as time at the store's delivered bandwidth
/// (`POLARS_CLOUD_PREFETCH_ORDERED_SLACK_MS`).
static ORDERED_SLACK: LazyLock<Duration> = LazyLock::new(|| {
    let env_var = "POLARS_CLOUD_PREFETCH_ORDERED_SLACK_MS";
    Duration::from_millis(
        std::env::var(env_var).map_or(DEFAULT_ORDERED_SLACK_MS, |x| {
            x.parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid value for {env_var}: {x}"))
        }),
    )
});

/// Scan nodes with a pipeline budget that are running, process-wide.
static ACTIVE_SCANS: AtomicUsize = AtomicUsize::new(0);

/// Pipeline kilobytes above the scans' default limits, process-wide: total memory / 16.
static WINDOW_POOL_KBYTES: LazyLock<AtomicUsize> = LazyLock::new(|| {
    AtomicUsize::new((polars_utils::sys::total_memory() / 16).div_ceil(1 << 10) as usize)
});

/// Value of `env_var` if set; panics if it is not a positive integer.
pub(crate) fn env_nonzero_usize(env_var: &str) -> Option<usize> {
    std::env::var(env_var).ok().map(|x| {
        x.parse::<NonZeroUsize>()
            .unwrap_or_else(|_| panic!("invalid value for {env_var}: {x}"))
            .get()
    })
}

/// Default prefetch count limit for a scan pipeline: at least the in-flight request budget, the
/// same for every running scan (only the kilobyte limit is split between them).
fn default_prefetch_limit(num_pipelines: usize) -> usize {
    num_pipelines
        .saturating_mul(2)
        .max(get_inflight_request_budget() as usize)
        .clamp(16, PipelineBudget::MAX_DEFAULT_COUNT_LIMIT)
}

/// Default prefetch kilobyte limit for a scan pipeline, split between `scans` running scans, at
/// least one download chunk.
fn default_prefetch_kbytes_limit(num_pipelines: usize, scans: usize) -> usize {
    // This should be large enough to be non-blocking, but small enough to avoid
    // excessive memory use from a run-away prefetch pipeline.
    // "Correct" formula: (a) max effective in-flight bdp-based bytes budget + (b) decode pipeline.
    // Since we do not know (a) at startup, we use  '3 * (b) + buffer' as a proxy for (a), where the
    // multiplier reflects the max gain factor for in-flight control.
    // NOTE: This does not account for the decompression multiplier, so actual memory
    // usage can be substantially larger.
    let target_chunk_size_kb = FetchConfig::random_access().chunk_size.div_ceil(1024);
    (4 * num_pipelines * target_chunk_size_kb / scans).max(get_download_chunk_size().div_ceil(1024))
}

/// A scan counted in `ACTIVE_SCANS`, with the kilobytes it holds from `WINDOW_POOL_KBYTES` (`None`
/// once released). Released once, when the scan ends or its budget is dropped.
#[derive(Debug)]
struct ScanRegistration {
    pool_kbytes: Mutex<Option<usize>>,
}

impl ScanRegistration {
    /// Registers a scan; returns it with the number of running scans, itself included.
    fn register() -> (Self, usize) {
        let scans = ACTIVE_SCANS.fetch_add(1, Ordering::Relaxed) + 1;
        let registration = Self {
            pool_kbytes: Mutex::new(Some(0)),
        };
        (registration, scans)
    }

    fn release(&self) {
        if let Some(kbytes) = self.pool_kbytes.lock().unwrap().take() {
            ACTIVE_SCANS.fetch_sub(1, Ordering::Relaxed);
            WINDOW_POOL_KBYTES.fetch_add(kbytes, Ordering::Relaxed);
        }
    }
}

impl Drop for ScanRegistration {
    fn drop(&mut self) {
        self.release();
    }
}

/// Releases the scan's registration when dropped.
pub struct ActiveScan {
    registration: Arc<ScanRegistration>,
}

impl Drop for ActiveScan {
    fn drop(&mut self) {
        self.registration.release();
    }
}

#[derive(Clone, Debug)]
pub struct PipelineBudget {
    count: Arc<Semaphore>,
    kbytes: Arc<Semaphore>,
    count_limit: Arc<AtomicUsize>,
    kbytes_limit: Arc<AtomicUsize>,
    /// False if the count limit was set by an env var.
    can_grow_count: bool,
    /// False if the kbytes limit was set by an env var.
    can_grow_kbytes: bool,
    ordered: Arc<AtomicBool>,
    num_pipelines: usize,
    /// In-flight budget of the store the scan reads from, attached by the first cloud reader.
    inflight: Arc<OnceLock<Arc<InFlightBudget>>>,
    registration: Arc<ScanRegistration>,
    last_reported: Arc<Mutex<Instant>>,
    report_interval: Duration,
}

impl PipelineBudget {
    /// Upper bound of the default count limit.
    pub(crate) const MAX_DEFAULT_COUNT_LIMIT: usize = 2048;

    /// Registers the scan as running and takes its share of the default kilobyte limit; a limit set
    /// by an env var is used as is and never grows.
    pub fn new(
        count_limit_env: Option<usize>,
        kbytes_limit_env: Option<usize>,
        num_pipelines: usize,
    ) -> Self {
        let (registration, scans) = ScanRegistration::register();
        let count_limit = count_limit_env.unwrap_or_else(|| default_prefetch_limit(num_pipelines));
        let kbytes_limit =
            kbytes_limit_env.unwrap_or_else(|| default_prefetch_kbytes_limit(num_pipelines, scans));

        Self {
            count: Arc::new(Semaphore::new(count_limit)),
            kbytes: Arc::new(Semaphore::new(kbytes_limit)),
            count_limit: Arc::new(AtomicUsize::new(count_limit)),
            kbytes_limit: Arc::new(AtomicUsize::new(kbytes_limit)),
            can_grow_count: count_limit_env.is_none(),
            can_grow_kbytes: kbytes_limit_env.is_none(),
            ordered: Arc::new(AtomicBool::new(false)),
            num_pipelines,
            inflight: Arc::new(OnceLock::new()),
            registration: Arc::new(registration),
            last_reported: Arc::new(Mutex::new(Instant::now())),
            report_interval: Duration::from_millis(100),
        }
    }

    pub(crate) fn count_limit(&self) -> usize {
        self.count_limit.load(Ordering::Acquire)
    }

    pub(crate) fn kbytes_limit(&self) -> usize {
        self.kbytes_limit.load(Ordering::Acquire)
    }

    /// Guard that releases the scan's registration when dropped, for scans that end before their
    /// budget is dropped.
    pub(crate) fn active_scan(&self) -> ActiveScan {
        ActiveScan {
            registration: self.registration.clone(),
        }
    }

    /// Per file, before sizing anything from `count_limit()`. The window grows with the first
    /// in-flight budget of the scan's store. For an ordered scan, unless set by an env var, the count
    /// limit becomes `ORDERED_COUNT_LIMIT` once a file has an in-flight budget, and grows by
    /// `ORDERED_LOCAL_COUNT_FACTOR` on the first file otherwise; the window grows by the reorder
    /// headroom.
    pub(crate) fn begin_file(&self, inflight: Option<Arc<InFlightBudget>>, maintain_order: bool) {
        if maintain_order {
            let first = !self.ordered.swap(true, Ordering::AcqRel);
            if self.can_grow_count && (first || inflight.is_some()) {
                let count_limit = if inflight.is_some() {
                    ORDERED_COUNT_LIMIT
                } else {
                    self.count_limit()
                        .saturating_mul(ORDERED_LOCAL_COUNT_FACTOR)
                        .min(Self::MAX_DEFAULT_COUNT_LIMIT)
                };
                let prev = self.count_limit.fetch_max(count_limit, Ordering::AcqRel);
                self.count.add_permits(count_limit.saturating_sub(prev));

                if config::verbose() && count_limit > prev {
                    eprintln!(
                        "[PipelineBudget]: ordered scan: prefetch_limit: {prev} -> {count_limit}"
                    );
                }
            }
        }
        if let (true, Some(inflight)) = (self.can_grow_kbytes, inflight) {
            let _ = self.inflight.set(inflight);
        }
    }

    /// Grows the kbytes limit from the window pool to the store's in-flight byte budget, plus the
    /// reorder headroom of an ordered scan, plus one fetch of `n_bytes` per decode slot. At most
    /// what the count limit holds at this fetch size. Never shrinks.
    fn grow_window(&self, n_bytes: usize) {
        let Some(inflight) = self.inflight.get() else {
            return;
        };

        let headroom = if self.ordered.load(Ordering::Acquire) {
            (inflight.delivered_bandwidth() as f64 * ORDERED_SLACK.as_secs_f64()) as usize
        } else {
            0
        };
        let target_bytes = (inflight.current_byte_budget() as usize)
            .saturating_add(headroom)
            .saturating_add((self.num_pipelines + 1).saturating_mul(n_bytes))
            .min(self.count_limit().saturating_mul(n_bytes));

        // Held while growing: concurrent fetches see the grown limit, and a release after the scan
        // ends returns all drawn kilobytes.
        let mut registration = self.registration.pool_kbytes.lock().unwrap();
        let Some(pool_kbytes) = registration.as_mut() else {
            return;
        };
        let kbytes_limit = self.kbytes_limit();
        let Some(missing) = target_bytes
            .div_ceil(1 << 10)
            .checked_sub(kbytes_limit)
            .filter(|&x| x > 0)
        else {
            return;
        };
        let free = WINDOW_POOL_KBYTES
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |free| {
                (free > 0).then(|| free - free.min(missing))
            })
            .unwrap_or(0);
        let granted = free.min(missing);
        if granted == 0 {
            return;
        }
        *pool_kbytes += granted;
        self.kbytes_limit.fetch_add(granted, Ordering::AcqRel);
        self.kbytes.add_permits(granted);

        if config::verbose() {
            eprintln!(
                "[PipelineBudget]: prefetch_kbytes_limit: {kbytes_limit} -> {}",
                kbytes_limit + granted
            );
        }
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
        if self.kbytes.available_permits() < n_bytes.div_ceil(1 << 10) {
            self.grow_window(n_bytes);
        }

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
                let kbytes_limit = self.kbytes_limit();
                let kbytes_in_use = kbytes_limit.saturating_sub(self.kbytes.available_permits());
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
                        kbytes_limit as f64 / 1e3,
                        kbytes_in_use as f64 / 1e3,
                        kbytes_in_use as f64 / kbytes_limit as f64,
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
        n_bytes.div_ceil(1 << 10).min(self.kbytes_limit())
    }
}

/// RAII permit pair; releases both budgets on drop.
pub(crate) struct PipelinePermit {
    _count: OwnedSemaphorePermit,
    _kbytes: OwnedSemaphorePermit,
}
