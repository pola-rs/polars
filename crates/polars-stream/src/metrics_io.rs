use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, atomic};
use std::time::Instant;

use futures::FutureExt;
use polars_utils::relaxed_cell::RelaxedCell;

pub struct ActiveIOMetrics {
    base_instant: Instant,
    active_io_count: RelaxedCell<u64>,
    /// Offset against `base_instant`.
    active_io_offset_ns: AtomicU64,
    active_io_total_ns: RelaxedCell<u64>,
}

impl Default for ActiveIOMetrics {
    fn default() -> Self {
        Self {
            base_instant: Instant::now(),
            active_io_count: RelaxedCell::default(),
            active_io_offset_ns: AtomicU64::new(0),
            active_io_total_ns: RelaxedCell::new_u64(0),
        }
    }
}

impl ActiveIOMetrics {
    pub fn active_io_total_ns(&self) -> u64 {
        self.active_io_total_ns.load()
    }

    pub async fn record_active_io_time<F, O>(&self, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        if self.active_io_count.fetch_add(1) == 0 {
            self.active_io_offset_ns.store(
                Instant::now()
                    .saturating_duration_since(self.base_instant)
                    .as_nanos() as _,
                atomic::Ordering::Release,
            );
        }

        let out = AssertUnwindSafe(fut).catch_unwind().await;

        let elapsed = u64::saturating_sub(
            Instant::now()
                .saturating_duration_since(self.base_instant)
                .as_nanos() as _,
            self.active_io_offset_ns.load(atomic::Ordering::Acquire),
        );

        if self.active_io_count.fetch_sub(1) == 1 {
            self.active_io_total_ns.fetch_add(elapsed);
        }

        match out {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

#[derive(Default, Clone)]
pub struct IOMetrics {
    pub(super) active_io_time_metrics: Arc<ActiveIOMetrics>,
    pub(super) bytes_requested: RelaxedCell<u64>,
    pub(super) bytes_received: RelaxedCell<u64>,
}

impl IOMetrics {
    pub async fn record_download<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        self.bytes_requested.fetch_add(num_bytes);

        let out = self.active_io_time_metrics.record_active_io_time(fut).await;

        self.bytes_received.fetch_add(num_bytes);

        out
    }
}

#[derive(Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub async fn record_download<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        if let Some(v) = self.0.clone() {
            v.record_download(num_bytes, fut).await
        } else {
            fut.await
        }
    }
}
