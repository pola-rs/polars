use std::sync::Arc;

use polars_utils::active_timer::{ActiveTimer, ActiveTimerSessionGuard};
use polars_utils::relaxed_cell::RelaxedCell;

#[derive(Debug, Default, Clone)]
pub struct IOMetrics {
    pub active_io_timer: Arc<ActiveTimer>,
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

        let session_guard = self.active_io_timer.register_session();
        let out = fut.await;
        drop(session_guard);

        self.bytes_received.fetch_add(num_bytes);

        out
    }
}

#[derive(Debug, Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub fn new_io_session_guard(&self) -> Option<ActiveTimerSessionGuard<'_>> {
        self.0
            .as_ref()
            .map(|x| x.active_io_timer.register_session())
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

pub struct MetricsWriteWrap<'a>(pub &'a mut (dyn std::io::Write + Send), pub OptIOMetrics);

impl std::io::Write for MetricsWriteWrap<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let guard = self.1.new_io_session_guard();
        let n = self.0.write(buf)?;
        drop(guard);
        self.1.add_bytes_sent(n as u64);

        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let guard = self.1.new_io_session_guard();
        self.0.flush()?;
        drop(guard);

        Ok(())
    }
}
