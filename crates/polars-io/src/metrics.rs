use std::sync::Arc;

use polars_utils::live_timer::{LiveTimer, LiveTimerSession};
use polars_utils::relaxed_cell::RelaxedCell;

#[derive(Debug, Default, Clone)]
pub struct IOMetrics {
    pub io_timer: LiveTimer,
    pub query_io_timers: Option<QueryIOTimers>,
    pub bytes_requested: RelaxedCell<u64>,
    pub bytes_received: RelaxedCell<u64>,
    pub bytes_sent: RelaxedCell<u64>,
}

#[derive(Debug, Default, Clone)]
pub struct QueryIOTimers {
    pub total: LiveTimer,
    pub rx: LiveTimer,
    pub tx: LiveTimer,
}

pub struct IOSession {
    _node: LiveTimerSession,
    _query: Option<QueryIOSession>,
}

struct QueryIOSession {
    _direction: LiveTimerSession,
    _total: LiveTimerSession,
}

#[derive(Clone, Copy)]
enum IODirection {
    Rx,
    Tx,
}

impl IOMetrics {
    fn start_session(&self, direction: IODirection) -> IOSession {
        let query = self.query_io_timers.as_ref().map(|timers| {
            let total = timers.total.start_session();
            let direction_timer = match direction {
                IODirection::Rx => &timers.rx,
                IODirection::Tx => &timers.tx,
            };
            QueryIOSession {
                _direction: direction_timer.start_session(),
                _total: total,
            }
        });

        IOSession {
            _node: self.io_timer.start_session(),
            _query: query,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub fn start_rx_session(&self) -> Option<IOSession> {
        self.0.as_ref().map(|x| x.start_session(IODirection::Rx))
    }

    pub fn start_tx_session(&self) -> Option<IOSession> {
        self.0.as_ref().map(|x| x.start_session(IODirection::Tx))
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

        let io_session = self.start_rx_session();

        let out = fut.await;

        drop(io_session);

        self.add_bytes_received(num_bytes);

        out
    }

    /// [`Self::record_io_read`] for a read done on the calling thread.
    pub fn record_io_read_blocking<F, O>(&self, num_bytes: u64, f: F) -> O
    where
        F: FnOnce() -> O,
    {
        self.add_bytes_requested(num_bytes);

        let io_session = self.start_rx_session();

        let out = f();

        drop(io_session);

        self.add_bytes_received(num_bytes);

        out
    }

    pub async fn record_bytes_tx<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        let io_session = self.start_tx_session();

        let out = fut.await;

        drop(io_session);

        self.add_bytes_sent(num_bytes);

        out
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use super::*;

    const IO_TIME: Duration = Duration::from_millis(10);

    fn tracked(timers: &QueryIOTimers) -> OptIOMetrics {
        OptIOMetrics(Some(Arc::new(IOMetrics {
            query_io_timers: Some(timers.clone()),
            ..Default::default()
        })))
    }

    fn node_ns(metrics: &OptIOMetrics) -> u64 {
        metrics.0.as_ref().unwrap().io_timer.total_time_live_ns()
    }

    /// Polls a future that completes without waiting.
    fn run_ready<F: Future>(fut: F) -> F::Output {
        match pin!(fut).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(out) => out,
            Poll::Pending => unreachable!(),
        }
    }

    #[test]
    fn an_rx_session_ticks_the_node_rx_and_total_timers() {
        let timers = QueryIOTimers::default();
        let metrics = tracked(&timers);

        let session = metrics.start_rx_session();
        std::thread::sleep(IO_TIME);
        drop(session);

        let node = node_ns(&metrics);
        assert!(node > 0);
        assert!(timers.rx.total_time_live_ns() >= node);
        assert!(timers.total.total_time_live_ns() >= timers.rx.total_time_live_ns());
        assert_eq!(timers.tx.total_time_live_ns(), 0);
    }

    #[test]
    fn a_tx_session_ticks_the_node_tx_and_total_timers() {
        let timers = QueryIOTimers::default();
        let metrics = tracked(&timers);

        let session = metrics.start_tx_session();
        std::thread::sleep(IO_TIME);
        drop(session);

        let node = node_ns(&metrics);
        assert!(node > 0);
        assert!(timers.tx.total_time_live_ns() >= node);
        assert!(timers.total.total_time_live_ns() >= timers.tx.total_time_live_ns());
        assert_eq!(timers.rx.total_time_live_ns(), 0);
    }

    #[test]
    fn reads_are_timed_as_rx() {
        let timers = QueryIOTimers::default();
        let metrics = tracked(&timers);

        metrics.record_io_read_blocking(1, || std::thread::sleep(IO_TIME));
        run_ready(metrics.record_io_read(1, async { std::thread::sleep(IO_TIME) }));

        assert!(timers.rx.total_time_live_ns() >= 2 * IO_TIME.as_nanos() as u64);
        assert_eq!(timers.tx.total_time_live_ns(), 0);
    }

    #[test]
    fn sends_are_timed_as_tx() {
        let timers = QueryIOTimers::default();
        let metrics = tracked(&timers);

        run_ready(metrics.record_bytes_tx(1, async { std::thread::sleep(IO_TIME) }));

        assert!(timers.tx.total_time_live_ns() >= IO_TIME.as_nanos() as u64);
        assert_eq!(timers.rx.total_time_live_ns(), 0);
    }

    #[test]
    fn without_query_timers_only_the_node_timer_ticks() {
        let metrics = OptIOMetrics(Some(Arc::default()));

        let session = metrics.start_rx_session();
        std::thread::sleep(IO_TIME);
        drop(session);

        assert!(node_ns(&metrics) > 0);
    }

    #[test]
    fn untracked_io_starts_no_session() {
        let metrics = OptIOMetrics(None);

        assert!(metrics.start_rx_session().is_none());
        assert!(metrics.start_tx_session().is_none());
    }
}
