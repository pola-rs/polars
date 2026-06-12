use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Debug)]
pub(crate) struct PipelineBudget {
    count: Arc<Semaphore>,
    kbytes: Arc<Semaphore>,
    count_limit: usize,
    kbytes_limit: usize,
}

impl PipelineBudget {
    pub(crate) fn new(count_limit: usize, kbytes_limit: usize) -> Self {
        Self {
            count: Arc::new(Semaphore::new(count_limit)),
            kbytes: Arc::new(Semaphore::new(kbytes_limit)),
            count_limit,
            kbytes_limit,
        }
    }

    pub(crate) fn count_limit(&self) -> usize {
        self.count_limit
    }

    #[allow(unused)]
    pub(crate) fn kbytes_limit(&self) -> usize {
        self.kbytes_limit
    }

    /// Acquire permits for a fetch of `n_bytes`.
    ///
    /// Acquisition order is kbytes-first, then count. All pipeline
    /// paths (parquet, IPC) must acquire through this method so the
    /// order can never diverge.
    pub(crate) async fn acquire(&self, n_bytes: usize) -> PipelinePermit {
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

        PipelinePermit { _count, _kbytes }
    }
}

/// RAII permit pair; releases both budgets on drop.
pub(crate) struct PipelinePermit {
    _count: OwnedSemaphorePermit,
    _kbytes: OwnedSemaphorePermit,
}
