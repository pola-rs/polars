use std::sync::Arc;

use polars_utils::live_timer::{LiveTimer, LiveTimerSession};
use polars_utils::relaxed_cell::RelaxedCell;

#[derive(Debug, Default, Clone)]
pub struct IOMetrics {
    pub io_timer: LiveTimer,
    pub bytes_requested: RelaxedCell<u64>,
    pub bytes_received: RelaxedCell<u64>,
    pub bytes_sent: RelaxedCell<u64>,
}

impl IOMetrics {
    pub async fn record_download<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        self.bytes_requested.fetch_add(num_bytes);

        let session_guard = self.io_timer.start_session();
        let out = fut.await;
        drop(session_guard);

        self.bytes_received.fetch_add(num_bytes);

        out
    }
}

#[derive(Debug, Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub fn new_io_session_guard(&self) -> Option<LiveTimerSession> {
        self.0.as_ref().map(|x| x.io_timer.start_session())
    }

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

    pub fn add_bytes_sent(&self, bytes_sent: u64) {
        self.0.as_ref().map(|x| x.bytes_sent.fetch_add(bytes_sent));
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
