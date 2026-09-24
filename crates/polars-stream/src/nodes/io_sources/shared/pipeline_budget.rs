use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use polars_io::cloud::concurrency_config::{FetchConfig, get_download_chunk_size};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static SHOULD_LOG_CONCURRENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_LOG_CONCURRENCY").is_ok());

/// Prefetch kilobyte limit for a scan pipeline.
///
/// A value set in `env_var` is used as given. The default is at least one download chunk.
pub(crate) fn prefetch_kbytes_limit_from_env_or_default(
    env_var: &str,
    num_pipelines: usize,
) -> usize {
    if let Ok(x) = std::env::var(env_var) {
        return x
            .parse::<NonZeroUsize>()
            .unwrap_or_else(|_| panic!("invalid value for {env_var}: {x}"))
            .get();
    }

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

#[derive(Clone, Debug)]
pub(crate) struct PipelineBudget {
    count: Arc<Semaphore>,
    kbytes: Arc<Semaphore>,
    count_limit: usize,
    kbytes_limit: usize,
    last_reported: Arc<Mutex<Instant>>,
    report_interval: Duration,
}

impl PipelineBudget {
    pub(crate) fn new(count_limit: usize, kbytes_limit: usize) -> Self {
        Self {
            count: Arc::new(Semaphore::new(count_limit)),
            kbytes: Arc::new(Semaphore::new(kbytes_limit)),
            count_limit,
            kbytes_limit,
            last_reported: Arc::new(Mutex::new(Instant::now())),
            report_interval: Duration::from_millis(100),
        }
    }

    pub(crate) fn count_limit(&self) -> usize {
        self.count_limit
    }

    #[allow(unused)]
    pub(crate) fn kbytes_limit(&self) -> usize {
        self.kbytes_limit
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
        // Prevent deadlock.
        let n_kbytes: u32 = n_bytes
            .div_ceil(1 << 10)
            .min(self.kbytes_limit)
            .try_into()
            .unwrap_or(u32::MAX);

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
                let kbytes_in_use = self.kbytes_limit - self.kbytes.available_permits();
                let count_in_use = self.count_limit - self.count.available_permits();
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
                        self.count_limit,
                        count_in_use,
                        count_in_use as f64 / self.count_limit as f64,
                    );
                    *last_log = Instant::now();
                }
            }
        }

        PipelinePermit { _count, _kbytes }
    }
}

/// RAII permit pair; releases both budgets on drop.
pub(crate) struct PipelinePermit {
    _count: OwnedSemaphorePermit,
    _kbytes: OwnedSemaphorePermit,
}
