//! The plugin's own async runtime. Host storage futures are awaited on it; the host IO itself
//! runs on the Polars IO runtime, so CPU work on these workers cannot stall IO.
use std::sync::OnceLock;

const MAX_WORKER_THREADS: usize = 8;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn runtime(max_threads_hint: Option<usize>) -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let n_threads = max_threads_hint
            .unwrap_or(available)
            .min(available)
            .clamp(1, MAX_WORKER_THREADS);

        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(n_threads)
            .thread_name("polars-delta")
            .build()
            .expect("failed to build polars-delta runtime")
    })
}

/// Block the calling (host) thread on `fut`. Called concurrently by several host threads when a
/// query contains several scans.
pub fn block_on<F: Future>(max_threads_hint: Option<u64>, fut: F) -> F::Output {
    runtime(max_threads_hint.map(|n| n as usize)).block_on(fut)
}
