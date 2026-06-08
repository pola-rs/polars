//! Adaptive in-flight concurrency controller for cloud IO.
//!
//! Admission control for concurrency uses two budgets:
//! - A (primary) bytes-based budget to model the bandwidth-delay product (BDP)
//! - A (secondary) count-based budget to limit the number of in-flight requests
//!
//! The bytes-based budget models the BDP as
//!   BDP = BW_max * TTFB_min
//!
//! Three components cooperate:
//! - Model: records IO observations and models the network (BW_max, TTFB_min, BDP)
//! - Regime: state machine (Init?RampUp/Stable/ProbeUp) driving the admission
//! - Admission: admission control, enforces byte + request budgets via semaphores
//!
//! Loosely based on BBR: Congestion-Based Congestion Control
//! see https://queue.acm.org/detail.cfm?id=3022184

mod admission;
mod model;
mod regime;

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use admission::{InFlightBudget, InFlightPermit, InFlightStats};
pub use model::Model;
pub use regime::{Regime, RegimeState};

use crate::pl_async::get_random_access_chunk_size;

#[derive(Clone, Copy, Debug)]
pub struct IoSample {
    pub nbytes: u64,
    // Time-to-first-byte
    pub ttfb: Duration,
    // Time-to-last-byte
    pub ttlb: Duration,
    pub completion_time: Instant,
}

// kdn TODO TUNE: make window a multiplier of control_interval?

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    // Retention time for keeping IOSamples.
    // kdn TODO: no value atm beyond window
    retention: Duration,
    // Sliding window over which the most recent round-trip-time (RTT) and bandwidth (BW)
    // will be calculated.
    window: Duration,
    // Time-to-first-byte default in case there is no sample available yet.
    ttfb_default: Duration,
    // Byte-based budget during the init and start of ramp-up phase. Used as a substitute
    // for the to-e modeled Bandwidth-Delay Product (BDP).
    init_byte_budget: u64,
    floor_byte_budget: u64,
    // Count-based request budget.
    request_budget: u32,
    // Controller update frequency.
    control_interval: Duration,
    // Total budget only resizes if the relative changes exceeds this threshold
    budget_resize_threshold: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        // Only used for bytes-based budget.
        let max_chunk_size = get_random_access_chunk_size() as u64;
        Self {
            retention: Duration::from_secs(1000),
            window: Duration::from_millis(1000),
            ttfb_default: Duration::from_millis(50), // AWS intra-region S3 TTFB range is 20-80 ms

            // kdn TODO TUNE: size by nr of vCPUs
            // Byte-based budget during the ramp-up phase.
            // Starting too low results in lost opportunity during ramp-up.
            // Starting too high results in early congestion, delayed completion of the first chunk,
            // and inflated bandwidth estimation.
            //
            // Some BDP numbers for reference:
            //   1 Gbps x 20 ms = 2.5 MB
            //   1 Gbps x 50 ms = 6.25 MB
            //   10 Gbps x 50 ms = 62.5 MB
            //   100 Gbps x 50 ms = 625 MB
            init_byte_budget: get_init_byte_budget(max_chunk_size),

            // Byte-based budget floor.
            // Must be larger than max chunk_size to avoid potential deadlock.
            floor_byte_budget: get_floor_byte_budget(max_chunk_size),

            // Count-based budget, currently fixed
            request_budget: get_request_budget(),
            control_interval: Duration::from_millis(100),
            budget_resize_threshold: 0.05,
        }
    }
}

fn get_init_byte_budget(max_chunk_size: u64) -> u64 {
    // Maximum number of bytes concurrently in flight during the init and start of rampup phase.
    let init_byte_budget = std::env::var("POLARS_INFLIGHT_INIT_BYTE_BUDGET")
        .map(|x| {
            x.parse::<NonZeroU64>()
                .unwrap_or_else(|_| {
                    panic!("invalid value for POLARS_INFLIGHT_INIT_BYTE_BUDGET: {x}")
                })
                .get()
        })
        //kdn TODO INIT probably 2* ??
        .unwrap_or(4 * max_chunk_size)
        .max(1);

    if init_byte_budget < max_chunk_size {
        panic!("in-flight byte budget init must be larger than the max_chunk_size");
    }

    init_byte_budget
}

fn get_floor_byte_budget(max_chunk_size: u64) -> u64 {
    // Floor number of bytes concurrently in flight.
    let floor_byte_budget = std::env::var("POLARS_INFLIGHT_FLOOR_BYTE_BUDGET")
        .map(|x| {
            x.parse::<NonZeroU64>()
                .unwrap_or_else(|_| {
                    panic!("invalid value for POLARS_INFLIGHT_FLOOR_BYTE_BUDGET: {x}")
                })
                .get()
        })
        .unwrap_or(2 * max_chunk_size)
        .max(1);

    if floor_byte_budget < max_chunk_size {
        panic!("in-flight byte budget floor must be larger than the max_chunk_size");
    }

    floor_byte_budget
}

pub fn get_request_budget() -> u32 {
    // Maximum number of concurrent in-flight requests. 
    // Since object_store/reqwest use HTTP/1 with a connection pool, this value controls the 
    // max concurrent TCP sessions to S3 for the pipeline.
    // Consider the tokio max_thread count, OS limitations, and object_store back-end limitations
    // when modifying this value.
    let request_budget = std::env::var("POLARS_INFLIGHT_REQUEST_BUDGET")
        .map(|x| {
            x.parse::<NonZeroU32>()
                .unwrap_or_else(|_| panic!("invalid value for POLARS_INFLIGHT_REQUEST_BUDGET: {x}"))
                .get()
        })
        .unwrap_or(512)
        .max(1);
    request_budget
}

#[derive(Debug)]
pub struct ConcurrencyController {
    config: ControllerConfig,
    model: Arc<Mutex<Model>>,
    _regime: Arc<Mutex<Regime>>,
    inflight_budget: Arc<InFlightBudget>,
    _control_task: tokio::task::JoinHandle<()>,
}

impl ConcurrencyController {
    pub fn new(config: ControllerConfig) -> Self {
        let now = Instant::now();

        let model = Arc::new(Mutex::new(Model::new(
            config.retention,
            config.window,
            config.ttfb_default,
        )));
        let regime = Arc::new(Mutex::new(Regime::new(now)));
        let inflight_budget = Arc::new(InFlightBudget::new(
            config.init_byte_budget,
            config.floor_byte_budget,
            config.request_budget,
        ));

        let control_task = Self::spawn_control_loop(
            model.clone(),
            regime.clone(),
            inflight_budget.clone(),
            config.clone(),
        );

        Self {
            config,
            model,
            _regime: regime,
            inflight_budget,
            _control_task: control_task,
        }
    }

    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }

    /// Record a completed IO. Hot path.
    pub fn record_io(&self, sample: IoSample) {
        // #kdn TODO PERF (AI): Mutex contended between hot-path and control loop.
        // Consider lockless ring buffer or per-thread batching if measured hot.
        if let Ok(mut e) = self.model.lock() {
            e.record(sample);
        }
    }

    /// Record a TTFB. Possibly hot path
    pub fn record_ttfb(&self, ttfb: Duration) {
        // #kdn TODO PERF (AI)): Mutex contended between hot-path and control loop.
        // Consider lockless ring buffer or per-thread batching if measured hot.

        if polars_config::config().verbose() {
            eprintln!("[AsyncIO] observed RTT: {:?}", ttfb)
        }

        if let Ok(mut e) = self.model.lock() {
            e.record_ttfb(ttfb);
        }
    }

    pub fn inflight_budget(&self) -> &Arc<InFlightBudget> {
        &self.inflight_budget
    }

    pub async fn acquire(&self, bytes: u64) -> InFlightPermit {
        self.inflight_budget.acquire(bytes).await
    }

    fn spawn_control_loop(
        model: Arc<Mutex<Model>>,
        regime: Arc<Mutex<Regime>>,
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
            let mut ticker = tokio::time::interval(config.control_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                ticker.tick().await;
                let now = Instant::now();

                // 1. Recompute estimates
                // Update bw_max, rtt_min, bw_last
                let (model_updated, bw_last, ttfb_min_last, ttfb_avg_last) = {
                    let mut model = model.lock().unwrap();
                    model.recompute(now)
                };

                // 2. Step regime through its state machine
                let (observed_bdp, gain, state, state_label) = {
                    let model = model.lock().unwrap();
                    let mut regime = regime.lock().unwrap();
                    regime.step(&model, now);
                    (
                        model.bdp_bytes(),
                        regime.gain_factor(),
                        regime.state().clone(),
                        regime.state_label(),
                    )
                };

                // 3. Compute target BDP and maybe resize
                let base_budget = match state {
                    RegimeState::Init => config.init_byte_budget,
                    RegimeState::RampUp { .. } => observed_bdp.max(config.init_byte_budget),
                    RegimeState::Stable | RegimeState::ProbeUp { .. } => {
                        //kdn TEST - hold on to init
                        observed_bdp.max(config.init_byte_budget)
                    },
                };

                let target_budget = (base_budget as f64 * gain) as u64;

                let current_byte_budget = admission.current_byte_budget();
                if should_resize(
                    target_budget,
                    current_byte_budget,
                    config.budget_resize_threshold,
                ) {
                    //kdn TODO - what if this takes longer than a tick
                    admission.resize_byte_budget(target_budget).await;
                }

                // 4. Log snapshot
                if (polars_config::config().verbose() && model_updated)
                    || std::env::var("POLARS_LOG_CONCURRENCY").is_ok()
                {
                    let model = model.lock().unwrap();
                    let stats = admission.stats();
                    eprintln!(
                        "[concurrency {}] regime={}, \
                        bw_max={:.1} MB/s, \
                        bw_last={:.1} MB/s, \
                        rtt_min={:.1} ms, \
                        rtt_avg={:.1} ms, \
                        rtt_ema={:.1} ms, \
                        bdp_obs={:.1} MB, \
                        budget={:.1} MB, \
                        in_use={:.1} MB, \
                        bytes_sat={:.2}, \
                        budget={}, \
                        in_use={}, \
                        req_sat={:.2}",
                        chrono::Utc::now(),
                        state_label,
                        model.bw_max_bps() / 1e6,
                        bw_last.unwrap_or(0.0) / 1e6,
                        model.ttfb_min().as_millis(),
                        model.ttfb_avg().as_millis(),
                        model.ttfb_ema_as_millis(),
                        model.bdp_bytes() as f64 / 1e6,
                        stats.byte_budget as f64 / 1e6,
                        stats.bytes_in_use as f64 / 1e6,
                        stats.bytes_saturation,
                        stats.request_budget,
                        stats.requests_in_use,
                        stats.requests_saturation
                    );
                }
            }
        })
    }
}

fn should_resize(target: u64, current: u64, deadband: f64) -> bool {
    if current == 0 {
        return target > 0;
    }
    let ratio = target as f64 / current as f64;
    ratio < (1.0 - deadband) || ratio > (1.0 + deadband)
}
