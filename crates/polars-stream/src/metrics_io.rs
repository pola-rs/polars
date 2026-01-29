use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, atomic};
use std::time::Instant;

use futures::FutureExt;
use polars_utils::relaxed_cell::RelaxedCell;

#[derive(Default)]
struct ActiveIOMetricsInner {
    active_io_count: AtomicU64,
    /// Offset against `base_instant`.
    active_io_offset_ns: RelaxedCell<u64>,
    active_io_total_ns: RelaxedCell<u64>,
}

pub struct ActiveIOMetrics {
    base_instant: Instant,
    inner: parking_lot::RwLock<ActiveIOMetricsInner>,
}

impl Default for ActiveIOMetrics {
    fn default() -> Self {
        Self {
            base_instant: Instant::now(),
            inner: Default::default(),
        }
    }
}

impl ActiveIOMetrics {
    fn ns_since_base_instant(&self) -> u64 {
        Instant::now()
            .saturating_duration_since(self.base_instant)
            .as_nanos() as _
    }

    fn get_or_start_session(&self) -> IOSession<'_> {
        let inner = self.inner.read();

        if inner
            .active_io_count
            .fetch_add(1, atomic::Ordering::Relaxed)
            == 0
        {
            inner
                .active_io_offset_ns
                .store(self.ns_since_base_instant());
        }

        drop(inner);

        IOSession {
            active_io_metrics: self,
            finished: false,
        }
    }

    pub fn active_io_total_ns(&self) -> u64 {
        let ns_since_base_instant = self.ns_since_base_instant();

        // Take an exclusive lock to wait for in-flight commits to finish and block new commits.
        let inner = self.inner.write();
        let active_io_count = inner.active_io_count.load(atomic::Ordering::Acquire);
        let active_io_offset_ns = inner.active_io_offset_ns.load();
        let active_io_total_ns = inner.active_io_total_ns.load();
        drop(inner);

        let elapsed = u64::saturating_sub(ns_since_base_instant, active_io_offset_ns);

        if active_io_count > 0 {
            active_io_total_ns + elapsed
        } else {
            active_io_total_ns
        }
    }

    pub async fn record_active_io_time<F, O>(&self, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        let session = self.get_or_start_session();

        let out = AssertUnwindSafe(fut).catch_unwind().await;

        drop(session);

        match out {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

/// Has Drop impl that ensures atomic counter is decremented.
struct IOSession<'a> {
    active_io_metrics: &'a ActiveIOMetrics,
    finished: bool,
}

impl IOSession<'_> {
    fn finish_by_ref(&mut self) {
        assert!(!self.finished);
        self.finished = true;

        let ns_since_base_instant = self.active_io_metrics.ns_since_base_instant();
        let active_io_metrics = self.active_io_metrics.inner.read();

        let elapsed = u64::saturating_sub(
            ns_since_base_instant,
            active_io_metrics.active_io_offset_ns.load(),
        );

        if active_io_metrics
            .active_io_count
            .fetch_sub(1, atomic::Ordering::Release)
            == 1
        {
            atomic::fence(atomic::Ordering::Acquire);
            active_io_metrics.active_io_total_ns.fetch_add(elapsed);
        }
    }
}

impl Drop for IOSession<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.finish_by_ref();
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
