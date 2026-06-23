//! Adaptive in-flight concurrency controller for cloud IO (Input only).
//!
//! Admission control for concurrency uses two budgets:
//! - A (primary) bytes-based budget to model the bandwidth-delay product (BDP)
//! - A (secondary) count-based budget to limit the number of in-flight requests
//!
//! The bytes-based budget models the BDP as
//!   BDP = BW_max * TTFB_est
//!
//! Three components cooperate:
//! - Model: records IO observations and models the network (BW_max, TTFB_est, BDP)
//! - Regime: state machine driving the admission (Init / RampUp / Stable / ProbeUp)
//! - Admission: admission control, enforces byte-based + request-based budgets

// Loosely based on BBR: Congestion-Based Congestion Control
// see https://queue.acm.org/detail.cfm?id=3022184

mod admission;
mod model;
mod regime;

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use admission::{InFlightBudget, InFlightPermit, InFlightStats};
use crossbeam_queue::ArrayQueue;
pub use model::Model;
use polars_utils::relaxed_cell::RelaxedCell;
pub use regime::{Regime, RegimeState};

// Number of samples in the queue, which gets drained on every tick.
// At 50k requests per second and 100 ms tick window, we need 5k.
// kdn TODO TEST & TUNE: Refactor to a run-time config-based value.
const SAMPLE_QUEUE_CAPACITY: usize = 8192;

use crate::cloud::concurrency_config::get_random_access_chunk_size;

#[derive(Clone, Copy, Debug)]
pub struct IoSample {
    pub n_bytes: u64,
    // Time-to-first-byte.
    pub ttfb: Duration,
    // TODO: Factor out as we only care about per-tick_window stats.
    pub completion_time: Instant,
}

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    // Sliding window over which the most recent round-trip-time (RTT) and bandwidth (BW)
    // will be calculated. Also acts as the retention window.
    window: Duration,
    // Byte-based budget during the Init phase, and as the base for the RampUp phase.
    init_byte_budget: u64,
    // Lower limit for the byte-based budget - needed to avoid deadlock.
    floor_byte_budget: u64,
    // Count-based request budget.
    request_budget: u32,
    // Controller regime update frequency.
    control_interval: Duration,
    // Total budget only resizes if the relative changes exceeds this threshold
    budget_resize_threshold: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        // Only used for bytes-based budget.
        let target_chunk_size = get_random_access_chunk_size() as u64;
        Self {
            window: Duration::from_millis(1000),

            // Byte-based budget during the ramp-up phase.
            // Starting too low results in lost opportunity (time) during ramp-up.
            // Starting too high results in early congestion, delayed completion of the first chunk,
            // and inflated bandwidth estimation.
            //
            // Some BDP numbers for reference:
            //   1 Gbps x 20 ms = 2.5 MB
            //   1 Gbps x 50 ms = 6.25 MB
            //   10 Gbps x 50 ms = 62.5 MB
            //   100 Gbps x 50 ms = 625 MB
            init_byte_budget: get_init_byte_budget(target_chunk_size),

            // Byte-based budget floor.
            // Must be >=larger than target_chunk_size to avoid potential deadlock.
            floor_byte_budget: target_chunk_size,

            // Count-based budget, currently fixed
            request_budget: get_request_budget(),
            control_interval: Duration::from_millis(100),
            budget_resize_threshold: 0.05,
        }
    }
}

/// Max number of bytes concurrently in flight during the init and start of rampup phase.
fn get_init_byte_budget(target_chunk_size: u64) -> u64 {
    let init_byte_budget = std::env::var("POLARS_INFLIGHT_INIT_BYTE_BUDGET")
        .map(|x| {
            x.parse::<NonZeroU64>()
                .unwrap_or_else(|_| {
                    panic!("invalid value for POLARS_INFLIGHT_INIT_BYTE_BUDGET: {x}")
                })
                .get()
        })
        .unwrap_or_else(|_| {
            // This should be lower than the expected BDP so it can ramp-up, but
            // too low a value delays the transition to stable.
            // Heuristic: higher bandwidth is expected on larger instances.
            let n = polars_config::config().max_threads() as u64;
            n.div_ceil(8).max(4) * target_chunk_size
        })
        .max(1);

    if init_byte_budget < target_chunk_size {
        panic!("in-flight byte budget init must be larger than the target_chunk_size");
    }

    init_byte_budget
}

/// Maximum number of requests concurrently in flight.
pub fn get_request_budget() -> u32 {
    // Since object_store/reqwest use HTTP/1 with a connection pool, this value controls the
    // max concurrent TCP sessions to S3 for the pipeline.
    // When modifying this value, consider the max_thread count configuration(s), the OS limitations
    // (e.g., ulimit -n), and any cloud infrastructure limitations.
    std::env::var("POLARS_INFLIGHT_REQUEST_BUDGET")
        .map(|x| {
            x.parse::<NonZeroU32>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_INFLIGHT_REQUEST_BUDGET: {x}"))
                .get()
        })
        .unwrap_or(512)
        .max(1)
}

#[derive(Debug)]
pub struct ConcurrencyController {
    config: ControllerConfig,
    sample_queue: Arc<ArrayQueue<IoSample>>,
    samples_dropped: Arc<RelaxedCell<u64>>,
    inflight_budget: Arc<InFlightBudget>,
    _control_task: tokio::task::JoinHandle<()>,
}

impl ConcurrencyController {
    pub fn new(config: ControllerConfig) -> Self {
        let sample_queue = Arc::new(ArrayQueue::new(SAMPLE_QUEUE_CAPACITY));
        let samples_dropped = Arc::new(RelaxedCell::new_u64(0));

        let inflight_budget = Arc::new(InFlightBudget::new(
            config.init_byte_budget,
            config.floor_byte_budget,
            config.request_budget,
        ));

        let control_task = Self::spawn_control_loop(
            sample_queue.clone(),
            samples_dropped.clone(),
            inflight_budget.clone(),
            config.clone(),
        );

        Self {
            config,
            sample_queue,
            samples_dropped,
            inflight_budget,
            _control_task: control_task,
        }
    }

    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }

    /// Record a completed IO. Hot path.
    pub fn record_io(&self, sample: IoSample) {
        if self.sample_queue.push(sample).is_err() {
            // Queue full: drop. Samples are statistics is considered acceptable.
            self.samples_dropped.fetch_add(1);
        }
    }

    pub fn inflight_budget(&self) -> &Arc<InFlightBudget> {
        &self.inflight_budget
    }

    pub async fn acquire(&self, bytes: u64) -> InFlightPermit {
        self.inflight_budget.acquire(bytes).await
    }

    fn spawn_control_loop(
        sample_queue: Arc<ArrayQueue<IoSample>>,
        samples_dropped: Arc<RelaxedCell<u64>>,
        admission: Arc<InFlightBudget>,
        config: ControllerConfig,
    ) -> tokio::task::JoinHandle<()> {
        if polars_config::config().verbose() {
            eprintln!(
                "[InFlightConcurrency]: spawn control loop: control_interval: {}ms",
                config.control_interval.as_millis()
            );
        }
        tokio::spawn(async move {
            let mut model = Model::new(config.window);
            let mut regime = Regime::new(Instant::now());

            let mut ticker = tokio::time::interval(config.control_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                ticker.tick().await;
                let now = Instant::now();

                // Update model statistics and step regime.
                let (state, signal, dropped, bw_hwm_held) = {
                    for _ in 0..SAMPLE_QUEUE_CAPACITY {
                        let Some(s) = sample_queue.pop() else { break };
                        model.record(s);
                    }
                    let dropped = samples_dropped.swap(0);
                    model.update(now);
                    let signal = model.signal();
                    let state = regime.step(signal, now);
                    let bw_hwm_held = model.bw_hwm_bps();
                    (state, signal, dropped, bw_hwm_held)
                };

                if !matches!(state, RegimeState::WarmIdle { .. }) {
                    // Compute base BDP
                    let base_budget = match (state, signal) {
                        (RegimeState::Init, _) | (_, None) => config.init_byte_budget,
                        (_, Some(signal)) => signal.bdp_bytes().max(config.init_byte_budget),
                    };

                    // Compute target BDP using the gain multiplier. This is similar to BBR cwnd_gain.
                    let gain = match state {
                        RegimeState::Init => 1.0,
                        RegimeState::RampUp { .. } => 2.0,
                        // NOTE: >> 1.0 for the purpose of absorbing environment noise.
                        RegimeState::Stable => 2.0,
                        RegimeState::ProbeUp { .. } => 3.0,
                        // Unreachable.
                        RegimeState::WarmIdle { .. } => 1.0,
                    };
                    let target_budget = (base_budget as f64 * gain) as u64;

                    // Resize if needed.
                    let current_byte_budget = admission.current_byte_budget();
                    let threshold = config.budget_resize_threshold;
                    let should_resize = match current_byte_budget {
                        0 => target_budget > 0,
                        current => {
                            let ratio = target_budget as f64 / current as f64;
                            ratio < (1.0 - threshold) || ratio > (1.0 + threshold)
                        },
                    };

                    if should_resize {
                        admission.resize_byte_budget(target_budget);
                    }
                }

                // Log snapshot.
                if std::env::var("POLARS_LOG_CONCURRENCY").is_ok() {
                    let stats = admission.stats();
                    eprintln!(
                        "[InFlightConcurrency {}] regime={}, \
                        bw_hwm={:.1} MB/s, \
                        bw_avg={:.1} MB/s, \
                        rtt_min={:.1} ms, \
                        rtt_avg={:.1} ms, \
                        bdp_obs={:.1} MB, \
                        bytes_budget={:.1} MB, \
                        bytes_in_use={:.1} MB, \
                        bytes_sat={:.2}, \
                        req_budget={}, \
                        req_in_use={}, \
                        req_sat={:.2}",
                        chrono::Utc::now(),
                        state.label(),
                        signal.map(|s| s.bw_hwm_bps).or(bw_hwm_held).unwrap_or(0.0) / 1e6,
                        signal.map_or(0.0, |s| s.bw_avg_bps) / 1e6,
                        signal.map_or(0, |s| s.ttfb_min.as_millis()),
                        signal.map_or(0, |s| s.ttfb_avg.as_millis()),
                        signal.map_or(0, |s| s.bdp_bytes()) as f64 / 1e6,
                        stats.bytes_budget as f64 / 1e6,
                        stats.bytes_in_use as f64 / 1e6,
                        stats.bytes_saturation,
                        stats.request_budget,
                        stats.requests_in_use,
                        stats.requests_saturation
                    );
                    if dropped > 0 {
                        eprintln!(
                            "[InFlightConcurrency] WARN: {dropped} samples dropped (queue full)"
                        );
                    }
                }
            }
        })
    }
}

impl Drop for ConcurrencyController {
    fn drop(&mut self) {
        self._control_task.abort();
    }
}
