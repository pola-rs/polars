use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static SHOULD_LOG_CONCURRENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_LOG_CONCURRENCY").is_ok());

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
    /// The requested capacity is cap'ped to avoid deadlock, at the expense of
    /// relaxing the memory management.
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
