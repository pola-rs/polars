use std::sync::Arc;

use polars_utils::live_timer::{LiveTimer, LiveTimerSession};
use polars_utils::relaxed_cell::RelaxedCell;

pub const HEAD_RESPONSE_SIZE_ESTIMATE: u64 = 1;

#[derive(Debug, Default, Clone)]
pub struct IOMetrics {
    pub io_timer: LiveTimer,
    pub bytes_requested: RelaxedCell<u64>,
    pub bytes_received: RelaxedCell<u64>,
    pub bytes_sent: RelaxedCell<u64>,
}

#[derive(Debug, Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub fn start_io_session(&self) -> Option<LiveTimerSession> {
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

    pub async fn record_io_read<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        self.add_bytes_requested(num_bytes);

        let io_session = self.start_io_session();

        let out = fut.await;

        drop(io_session);

        self.add_bytes_received(num_bytes);

        out
    }

    pub async fn record_bytes_tx<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        let io_session = self.start_io_session();

        let out = fut.await;

        drop(io_session);

        self.add_bytes_sent(num_bytes);

        out
    }
}
