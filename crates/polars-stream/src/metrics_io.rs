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

/// Has Drop impl that ensures atomic counter is decremented.
struct IOSession<'a> {
    active_io_metrics: &'a ActiveIOMetrics,
    decremented: bool,
}

impl IOSession<'_> {
    fn finish(&mut self, ns_since_base_instant: u64) -> bool {
        assert!(!self.decremented);
        self.decremented = true;

        let active_io_metrics = self.active_io_metrics;

        let elapsed = u64::saturating_sub(
            ns_since_base_instant,
            active_io_metrics
                .active_io_offset_ns
                .load(atomic::Ordering::Acquire),
        );

        let ended_by_this_call = active_io_metrics.active_io_count.fetch_sub(1) == 1;

        if ended_by_this_call {
            active_io_metrics.active_io_total_ns.fetch_add(elapsed);
        }

        ended_by_this_call
    }
}

impl Drop for IOSession<'_> {
    fn drop(&mut self) {
        if !self.decremented {
            self.finish(
                Instant::now()
                    .saturating_duration_since(self.active_io_metrics.base_instant)
                    .as_nanos() as _,
            );
        }
    }
}

impl ActiveIOMetrics {
    fn start_io_session(&self) -> (IOSession<'_>, bool) {
        let started_by_this_call = self.active_io_count.fetch_add(1) == 0;

        if started_by_this_call {
            self.active_io_offset_ns.store(
                Instant::now()
                    .saturating_duration_since(self.base_instant)
                    .as_nanos() as _,
                atomic::Ordering::Release,
            );
        }

        (
            IOSession {
                active_io_metrics: self,
                decremented: false,
            },
            started_by_this_call,
        )
    }

    pub fn active_io_total_ns(&self) -> u64 {
        // If there are active I/O tasks, we would like to add the current elapsed
        // time to the `active_io_total_ns` counter.
        // For us to do so we need to ensure no other threads will concurrently
        // update the counter - we do this by starting a session here.
        let (mut session_ref, started_by_this_call) = self.start_io_session();

        let active_io_total_ns = self.active_io_total_ns.load();

        let ns_since_base_instant = Instant::now()
            .saturating_duration_since(self.base_instant)
            .as_nanos() as _;
        let elapsed = u64::saturating_sub(
            ns_since_base_instant,
            self.active_io_offset_ns.load(atomic::Ordering::Acquire),
        );

        session_ref.finish(if started_by_this_call {
            0
        } else {
            ns_since_base_instant
        });

        if started_by_this_call {
            active_io_total_ns
        } else {
            active_io_total_ns + elapsed
        }
    }

    pub async fn record_active_io_time<F, O>(&self, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        let (mut session_ref, _) = self.start_io_session();

        let out = AssertUnwindSafe(fut).catch_unwind().await;

        session_ref.finish(
            Instant::now()
                .saturating_duration_since(self.base_instant)
                .as_nanos() as _,
        );

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
