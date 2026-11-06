use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use polars_core::config;
use polars_io::cloud::concurrency::{InFlightBudget, get_inflight_request_budget};
use polars_io::cloud::concurrency_config::{FetchConfig, get_download_chunk_size};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

// Pipeline window: the bytes and fetches a scan holds between fetch and decode.
// - Cloud fetches: the byte limit grows toward the in-flight byte budget of the fetch's store
//   (shared by every scan on that store), plus one fetch per decode slot. It grows on demand and
//   never shrinks.
// - Ordered scans also hold completed fetches behind the head of line: the target adds the
//   delivered bandwidth times the slack, and the count limit is 8192 once a file follows a store
//   (otherwise the default x2, at most 2048).
// - Local fetches and limits set by env vars keep a fixed window.
// - Each scan starts at the default limits. Bytes above them come from one process-wide pool (a
//   fraction of total memory), first come, first served, and return when the scan ends.
// - A scan whose output is blocked (e.g. a join's probe side during the build) keeps fetching and
//   can grow toward its store's budget.

static SHOULD_LOG_CONCURRENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_LOG_CONCURRENCY").is_ok());

/// Default of `POLARS_CLOUD_PREFETCH_ORDERED_SLACK_MS`.
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

/// Count limit of an ordered scan once a file follows a store, also with the kbytes limit set by an
/// env var. The window is sized in bytes; this only bounds tasks and queue slots for small fetches.
const ORDERED_COUNT_LIMIT: usize = 8192;

/// Factor by which an ordered scan grows the default count limit when its first file does not
/// follow a store.
const ORDERED_LOCAL_COUNT_FACTOR: usize = 2;

/// Default of `POLARS_CLOUD_PREFETCH_POOL_FRACTION`.
const DEFAULT_POOL_FRACTION: f64 = 0.0625;

/// Pipeline kilobytes above the scans' default limits, process-wide: a fraction of total memory
/// (`POLARS_CLOUD_PREFETCH_POOL_FRACTION`; 0 = windows never grow).
static WINDOW_POOL_KBYTES: LazyLock<AtomicUsize> = LazyLock::new(|| {
    let env_var = "POLARS_CLOUD_PREFETCH_POOL_FRACTION";
    let fraction = std::env::var(env_var).map_or(DEFAULT_POOL_FRACTION, |x| {
        x.parse::<f64>()
            .ok()
            .filter(|f| (0.0..=1.0).contains(f))
            .unwrap_or_else(|| panic!("invalid value for {env_var}: {x}"))
    });
    let total_memory = polars_utils::sys::total_memory();
    let pool_bytes = (total_memory as f64 * fraction) as u64;

    if config::verbose() {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        eprintln!(
            "[PipelineBudget]: prefetch window pool: {:.3} GiB ({fraction} x total memory {:.3} GiB)",
            gib(pool_bytes),
            gib(total_memory)
        );
    }

    AtomicUsize::new(pool_bytes.div_ceil(1 << 10) as usize)
});

/// Interval at which a parked acquire retries growing the window.
const GROW_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// Value of `env_var` if set; panics if it is not a positive integer.
pub(crate) fn env_nonzero_usize(env_var: &str) -> Option<usize> {
    std::env::var(env_var).ok().map(|x| {
        x.parse::<NonZeroUsize>()
            .unwrap_or_else(|_| panic!("invalid value for {env_var}: {x}"))
            .get()
    })
}

/// Default prefetch count limit for a scan pipeline: max(2 x pipelines, in-flight request budget),
/// clamped to 16..=`MAX_DEFAULT_COUNT_LIMIT`.
fn default_prefetch_limit(num_pipelines: usize) -> usize {
    num_pipelines
        .saturating_mul(2)
        .max(get_inflight_request_budget() as usize)
        .clamp(16, PipelineBudget::MAX_DEFAULT_COUNT_LIMIT)
}

/// Default prefetch kilobyte limit for a scan pipeline, at least one download chunk.
fn default_prefetch_kbytes_limit(num_pipelines: usize) -> usize {
    // This should be large enough to be non-blocking, but small enough to avoid
    // excessive memory use from a run-away prefetch pipeline.
    // "Correct" formula: (a) max effective in-flight bdp-based bytes budget + (b) decode pipeline.
    // Since we do not know (a) at startup, we use  '3 * (b) + buffer' as a proxy for (a), where the
    // multiplier reflects the max gain factor for in-flight control.
    // NOTE: This does not account for the decompression multiplier, so actual memory
    // usage can be substantially larger.
    let target_chunk_size_kb = FetchConfig::random_access().chunk_size.div_ceil(1024);
    (4 * num_pipelines * target_chunk_size_kb).max(get_download_chunk_size().div_ceil(1024))
}

/// The kilobytes a scan holds from `WINDOW_POOL_KBYTES` (`None` once released). Released once, when
/// the scan ends or is aborted, or its budget is dropped. Fetch tasks that are still being aborted
/// can hold their buffers a little longer.
#[derive(Debug)]
struct ScanRegistration {
    pool_kbytes: Mutex<Option<usize>>,
}

impl ScanRegistration {
    fn new() -> Self {
        Self {
            pool_kbytes: Mutex::new(Some(0)),
        }
    }

    fn release(&self) {
        let kbytes = self
            .pool_kbytes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(kbytes) = kbytes {
            WINDOW_POOL_KBYTES.fetch_add(kbytes, Ordering::Relaxed);
        }
    }
}

impl Drop for ScanRegistration {
    fn drop(&mut self) {
        self.release();
    }
}

/// Returns the scan's pool kilobytes when dropped.
pub struct ActiveScan {
    registration: Arc<ScanRegistration>,
}

impl Drop for ActiveScan {
    fn drop(&mut self) {
        self.registration.release();
    }
}

/// What held the window at the last growth attempt, for `POLARS_LOG_CONCURRENCY`.
#[derive(Clone, Copy)]
enum GrowCap {
    /// The window reached the target.
    TargetKbytes = 1,
    /// The count limit times the fetch size is below the target.
    CountLimit = 2,
    /// The pool had less than the window needed.
    Pool = 3,
}

impl GrowCap {
    fn label(value: u8) -> &'static str {
        match value {
            1 => "target_kbytes",
            2 => "count_limit",
            3 => "pool",
            _ => "none",
        }
    }
}

/// `POLARS_LOG_CONCURRENCY` counters, since the last report.
#[derive(Debug, Default)]
struct LogStats {
    /// Acquires that waited for kilobytes: after any growth, fewer were free than they take.
    kbytes_waits: AtomicUsize,
    /// Acquires that waited for a count permit: none was free.
    count_waits: AtomicUsize,
    /// `GrowCap` of the last growth attempt; 0 before the first.
    grow_cap: AtomicU8,
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
    /// Set by the first file read in order. The window target then adds the reorder headroom.
    ordered: Arc<AtomicBool>,
    num_pipelines: usize,
    registration: Arc<ScanRegistration>,
    log_stats: Arc<LogStats>,
    last_reported: Arc<Mutex<Instant>>,
    report_interval: Duration,
}

impl PipelineBudget {
    /// Upper bound of the default count limit, also after it grows by
    /// `ORDERED_LOCAL_COUNT_FACTOR`. `ORDERED_COUNT_LIMIT` is above it.
    pub(crate) const MAX_DEFAULT_COUNT_LIMIT: usize = 2048;

    /// Starts at the default limits. A limit set by an env var is used as is and never grows.
    pub fn new(
        count_limit_env: Option<usize>,
        kbytes_limit_env: Option<usize>,
        num_pipelines: usize,
    ) -> Self {
        // Sized (and logged in verbose) at the first scan.
        LazyLock::force(&WINDOW_POOL_KBYTES);

        let count_limit = count_limit_env.unwrap_or_else(|| default_prefetch_limit(num_pipelines));
        let kbytes_limit =
            kbytes_limit_env.unwrap_or_else(|| default_prefetch_kbytes_limit(num_pipelines));

        Self {
            count: Arc::new(Semaphore::new(count_limit)),
            kbytes: Arc::new(Semaphore::new(kbytes_limit)),
            count_limit: Arc::new(AtomicUsize::new(count_limit)),
            kbytes_limit: Arc::new(AtomicUsize::new(kbytes_limit)),
            can_grow_count: count_limit_env.is_none(),
            can_grow_kbytes: kbytes_limit_env.is_none(),
            ordered: Arc::new(AtomicBool::new(false)),
            num_pipelines,
            registration: Arc::new(ScanRegistration::new()),
            log_stats: Arc::default(),
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

    /// The same budget with a kbytes limit that never grows.
    pub(crate) fn with_fixed_kbytes(mut self) -> Self {
        self.can_grow_kbytes = false;
        self
    }

    /// The initial limits, for the reader builders' verbose output.
    pub(crate) fn describe_limits(&self) -> String {
        let origin = |can_grow| if can_grow { "initial, may grow" } else { "env" };
        format!(
            "prefetch_limit: {} ({}), prefetch_kbytes_limit: {} ({})",
            self.count_limit(),
            origin(self.can_grow_count),
            self.kbytes_limit(),
            origin(self.can_grow_kbytes)
        )
    }

    /// Guard that returns the scan's pool kilobytes when dropped, for scans that end before their
    /// budget is dropped.
    pub(crate) fn active_scan(&self) -> ActiveScan {
        ActiveScan {
            registration: self.registration.clone(),
        }
    }

    /// Raises the count limit of an ordered scan: to `ORDERED_COUNT_LIMIT` when the file follows a
    /// store, else by `ORDERED_LOCAL_COUNT_FACTOR` on the first file. Not for a count set by an env
    /// var. Call per file, before sizing anything from `count_limit()`.
    pub(crate) fn begin_file(&self, follows_store: bool, maintain_order: bool) {
        if !maintain_order {
            return;
        }
        let first = !self.ordered.swap(true, Ordering::AcqRel);
        if self.can_grow_count && (first || follows_store) {
            let count_limit = if follows_store {
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

    /// Grows the kbytes limit from the pool toward the target: the in-flight byte budget of the
    /// fetch's store, plus the reorder headroom of an ordered scan, plus one fetch of `n_bytes` per
    /// decode slot. The target is at most the count limit times `n_bytes`, and growth at most what
    /// the pool has left. Never shrinks.
    fn grow_window(&self, n_bytes: usize, inflight: &InFlightBudget) {
        let headroom = if self.ordered.load(Ordering::Acquire) {
            (inflight.delivered_bandwidth() as f64 * ORDERED_SLACK.as_secs_f64()) as usize
        } else {
            0
        };
        let target_bytes = (inflight.current_byte_budget() as usize)
            .saturating_add(headroom)
            .saturating_add((self.num_pipelines + 1).saturating_mul(n_bytes));
        let cap_bytes = self.count_limit().saturating_mul(n_bytes);
        let mut grow_cap = if cap_bytes < target_bytes {
            GrowCap::CountLimit
        } else {
            GrowCap::TargetKbytes
        };
        let target_kbytes = target_bytes.min(cap_bytes).div_ceil(1 << 10);

        // Held while growing, so that a release after the scan ends returns all drawn kilobytes.
        let mut registration = self
            .registration
            .pool_kbytes
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(pool_kbytes) = registration.as_mut() else {
            return;
        };
        let kbytes_limit = self.kbytes_limit();
        let missing = target_kbytes.saturating_sub(kbytes_limit);
        if missing > 0 {
            let free = WINDOW_POOL_KBYTES
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |free| {
                    (free > 0).then(|| free - free.min(missing))
                })
                .unwrap_or(0);
            let granted = free.min(missing);
            if granted < missing {
                grow_cap = GrowCap::Pool;
            }
            if granted > 0 {
                *pool_kbytes += granted;
                self.kbytes_limit.fetch_add(granted, Ordering::AcqRel);
                self.kbytes.add_permits(granted);

                if config::verbose() {
                    eprintln!(
                        "[PipelineBudget]: prefetch_kbytes_limit: {kbytes_limit} -> {} (target: \
                        {target_kbytes}, pool free: {})",
                        kbytes_limit + granted,
                        free - granted
                    );
                }
            }
        }

        if *SHOULD_LOG_CONCURRENCY {
            self.log_stats
                .grow_cap
                .store(grow_cap as u8, Ordering::Relaxed);
        }
    }

    /// Acquire permit for a fetch of `n_bytes`. `inflight`: the in-flight budget of the fetch's
    /// store if its window follows it (`DynByteSource::window_inflight_budget`); the window then
    /// grows when the fetch finds it short, and again every `GROW_RETRY_INTERVAL` while it waits.
    ///
    /// Acquisition order is kbytes-first, then count, so that the count_in_use
    /// value is meaningful. All pipeline paths (parquet, IPC) must acquire
    /// through this method so the order can never diverge.
    ///
    /// The requested capacity is capped at the kbytes limit, so a fetch larger than the
    /// limit still proceeds, and any positive limit makes progress. Such a fetch keeps the charge
    /// of the limit it was sized under, also when the window grows later.
    pub(crate) async fn acquire(
        &self,
        n_bytes: usize,
        inflight: Option<&InFlightBudget>,
    ) -> PipelinePermit {
        let inflight = inflight.filter(|_| self.can_grow_kbytes);
        if let Some(inflight) = inflight
            && self.kbytes.available_permits() < n_bytes.div_ceil(1 << 10)
        {
            self.grow_window(n_bytes, inflight);
        }

        let n_kbytes: u32 = self.permit_kbytes(n_bytes).try_into().unwrap_or(u32::MAX);

        if *SHOULD_LOG_CONCURRENCY && self.kbytes.available_permits() < n_kbytes as usize {
            self.log_stats.kbytes_waits.fetch_add(1, Ordering::Relaxed);
        }

        // Semaphores are never closed, so acquire cannot fail.
        let kbytes = self.kbytes.clone().acquire_many_owned(n_kbytes);
        let _kbytes = match inflight {
            Some(inflight) => {
                tokio::pin!(kbytes);
                loop {
                    tokio::select! {
                        permit = &mut kbytes => break permit,
                        _ = tokio::time::sleep(GROW_RETRY_INTERVAL) => {
                            self.grow_window(n_bytes, inflight)
                        },
                    }
                }
            },
            None => kbytes.await,
        }
        .unwrap();

        if *SHOULD_LOG_CONCURRENCY && self.count.available_permits() == 0 {
            self.log_stats.count_waits.fetch_add(1, Ordering::Relaxed);
        }
        let _count = self.count.clone().acquire_owned().await.unwrap();

        if *SHOULD_LOG_CONCURRENCY {
            if let Ok(mut last_log) = self.last_reported.lock() {
                let count_limit = self.count_limit();
                let kbytes_limit = self.kbytes_limit();
                let kbytes_in_use = kbytes_limit.saturating_sub(self.kbytes.available_permits());
                let count_in_use = count_limit.saturating_sub(self.count.available_permits());
                if last_log.elapsed() > self.report_interval {
                    let mb = |kbytes: usize| kbytes as f64 * 1024.0 / 1e6;
                    let kbytes_waits = self.log_stats.kbytes_waits.swap(0, Ordering::Relaxed);
                    let count_waits = self.log_stats.count_waits.swap(0, Ordering::Relaxed);
                    let bound = match (kbytes_waits > 0, count_waits > 0) {
                        (true, true) => "kbytes+count",
                        (true, false) => "kbytes",
                        (false, true) => "count",
                        (false, false) => "none",
                    };
                    eprintln!(
                        "[PipelineBudget {}] \
                        kbytes_limit={:.1} MB, \
                        kbytes_in_use={:.1} MB, \
                        kbytes_sat={:.2}, \
                        count_limit={}, \
                        count_in_use={}, \
                        count_sat={:.2}, \
                        bound={bound}, \
                        grow_cap={}, \
                        pool_free={:.1} MB",
                        chrono::Utc::now(),
                        mb(kbytes_limit),
                        mb(kbytes_in_use),
                        kbytes_in_use as f64 / kbytes_limit as f64,
                        count_limit,
                        count_in_use,
                        count_in_use as f64 / count_limit as f64,
                        GrowCap::label(self.log_stats.grow_cap.load(Ordering::Relaxed)),
                        mb(WINDOW_POOL_KBYTES.load(Ordering::Relaxed)),
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
