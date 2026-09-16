use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
pub(crate) struct PipelinePermit {
    _count: OwnedSemaphorePermit,
    _kbytes: OwnedSemaphorePermit,
}

impl PipelinePermit {
    /// Acquire pipeline permits in canonical order (kbytes, then count).
    /// `n_bytes` is clamped to the semaphore capacity so oversized
    /// requests serialize rather than deadlock.
    pub(crate) async fn acquire(
        count_semaphore: Arc<Semaphore>,
        kbytes_semaphore: Arc<Semaphore>,
        kbytes_limit: usize,
        n_bytes: usize,
    ) -> Self {
        let n_kbytes = n_bytes
            .div_ceil(1 << 10)
            .min(kbytes_limit)
            .try_into()
            .unwrap_or(u32::MAX);

        let _kbytes = kbytes_semaphore.acquire_many_owned(n_kbytes).await.unwrap();
        let _count = count_semaphore.acquire_owned().await.unwrap();

        Self { _count, _kbytes }
    }
}
