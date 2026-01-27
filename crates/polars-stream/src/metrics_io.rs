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
    /// MUST have an associated `end_io_session()`.
    fn start_io_session(&self) -> bool {
        let started_by_this_call = self.active_io_count.fetch_add(1) == 0;

        if started_by_this_call {
            self.active_io_offset_ns.store(
                Instant::now()
                    .saturating_duration_since(self.base_instant)
                    .as_nanos() as _,
                atomic::Ordering::Release,
            );
        }

        started_by_this_call
    }

    /// MUST have an associated `start_io_session()`.
    fn end_io_session(&self, ns_since_base_instant: u64) -> bool {
        let elapsed = u64::saturating_sub(
            ns_since_base_instant,
            self.active_io_offset_ns.load(atomic::Ordering::Acquire),
        );

        let ended_by_this_call = self.active_io_count.fetch_sub(1) == 1;

        if ended_by_this_call {
            self.active_io_total_ns.fetch_add(elapsed);
        }

        ended_by_this_call
    }

    pub fn active_io_total_ns(&self) -> u64 {
        // If there are active I/O tasks, we would like to add the current elapsed
        // time to the `active_io_total_ns` counter.
        // To do this, we `start_io_session()` to hold a refcount to ensure
        // `active_io_total_ns` isn't updated by another thread.
        let started_by_this_call = self.start_io_session();

        let active_io_total_ns = self.active_io_total_ns.load();

        let ns_since_base_instant = Instant::now()
            .saturating_duration_since(self.base_instant)
            .as_nanos() as _;
        let elapsed = u64::saturating_sub(
            ns_since_base_instant,
            self.active_io_offset_ns.load(atomic::Ordering::Acquire),
        );

        self.end_io_session(if started_by_this_call {
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
        self.start_io_session();

        let out = AssertUnwindSafe(fut).catch_unwind().await;

        self.end_io_session(
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
