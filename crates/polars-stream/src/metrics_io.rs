use std::sync::Arc;

use polars_utils::active_timer::{ActiveTimer, ActiveTimerSessionGuard};
use polars_utils::relaxed_cell::RelaxedCell;

#[derive(Default, Clone)]
pub struct IOMetrics {
    pub(super) active_io_timer: Arc<ActiveTimer>,
    pub(super) bytes_requested: RelaxedCell<u64>,
    pub(super) bytes_received: RelaxedCell<u64>,
}

impl IOMetrics {
    pub async fn record_download<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        self.bytes_requested.fetch_add(num_bytes);

        let session_guard = self.active_io_timer.register_session();
        let out = fut.await;
        drop(session_guard);

        self.bytes_received.fetch_add(num_bytes);

        out
    }
}

#[derive(Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub fn add_bytes_requested(&self, bytes_requested: u64) {
        self.0
            .as_ref()
            .map(|x| x.bytes_requested.fetch_add(bytes_requested));
    }

    pub fn add_bytes_received(&self, bytes_received: u64) {
        self.0
            .as_ref()
            .map(|x| x.bytes_received.fetch_add(bytes_received));
    }

    pub fn new_io_session_guard(&self) -> Option<ActiveTimerSessionGuard<'_>> {
        self.0
            .as_ref()
            .map(|x| x.active_io_timer.register_session())
    }

    pub async fn record_download<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        self.add_bytes_requested(num_bytes);

        let guard = self.new_io_session_guard();

        let out = fut.await;

        drop(guard);

        self.add_bytes_received(num_bytes);

        out
    }
}
