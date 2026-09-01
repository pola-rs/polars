// Additive Increase / Multiplicative Decrease (AIMD) adaptive rate-limiter at
// the HttpService with JIT-pricing.
//
// Components:
// - Rate-limiter: responsible for config-based rate-limiting, including adapting
//   to the observed success rate.
// - Pacer (lock-free hot path): responsible for pacing the requests on `admit()`.
//   The pacer will deny requests that have an estimated wait that is too far out.
//   Holds an AtomicU64 f64 bits representation of the learned rate.
// - PacerSignal (lock-free warm path): collects metrics from the pacer and holds
//   time window boundaries to protect the cold path Mutex.
// - AimdState (cold path): actuator, adapts the rate based on the observed
//   HTTP success or failure rate. Authoritative for the atomic (learned) rate,
//   shared real-time lock-free with the pacer and inflight concurrency controller.
//
// Integration points:
// - The inflight concurrency controller adjusts its concurrency based on the
//   learned rate. The `RateCell` contains the shared information.
// - The object_store is responsible for retries, acting as a shock absorber
//   for when the pipeline pushes above the rate-limit.
// - The `InitPolicy` facilitates persistent learning when an object_store
//   is rebuilt on error.
//
// TODO:
// - Rate-limit is one per object_store per verb. AWS S3 scaled out its
//   rate-limit per-prefix when hot. This can be used to refine the rate-limiter.
// - The bridge between rate-limiter and inflight concurrency controller allows
//   for a more reliable RTT_min signal than the clamped one currently in use,
//   since it sits below the retry layer.

use std::collections::VecDeque;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use object_store::ClientOptions;
use object_store::client::{
    HttpClient, HttpConnector, HttpError, HttpErrorKind, HttpRequest, HttpResponse, HttpService,
};
use polars_core::runtime::ASYNC;
use tokio::sync::oneshot;

use super::token_bucket::TokenBucket;
use crate::cloud::{CloudDirectionalRateLimitConfig, CloudRateLimitConfig};

// For tracing purposes.
static CONTROLLER_ID: AtomicU32 = AtomicU32::new(0);

// Verbose.
static LOG_HTTP_RATE_LIMIT: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_LOG_HTTP_RATE_LIMIT").is_ok());

// Request/s rate init and boundaries.
const DEFAULT_INIT_RATE: f64 = 1000.0;
const DEFAULT_FLOOR_RATE: f64 = 10.0;
const DEFAULT_CEILING_RATE: f64 = 50_000.0;

// Increase / decrease parameters.
// Cold-start multiplicative increase, per tick.
const FAST_RAMP_FACTOR: f64 = 2.0;
// Additive step increase (relative to max), per tick.
const PROBE_FRACTION: f64 = 0.1;
const PROBE_MIN: f64 = 5.0;
// Multiplicative decrease (on cut), with signal
const BETA: f64 = 0.7;
// Multiplicative decrease (on cut), no signal available (e.g. cold start)
const BETA_NO_SIGNAL: f64 = 0.5;

// Timing parameters.
// Target queue depth as communicated to the concurrency controller.
const DEFAULT_RATE_HORIZON_MS: u64 = 200;
// Maximum queue depth, or admission denial ceiling: fail-fast rather than park
// when the estimated wait exceeds this.
// Note. Under nominal conditions the concurrency-driven population cap
// sets depth below this value. Kicks in as buffer on rate collapse.
// TODO: Dynamically size based on CloudRetryConfig and Init/Floor values.
// Indicative sizing: to avoid object store erroring out, the rate-limiter must have
// sufficient capacity to handle a drop-off from Init to Floor. This comes down
// (MAX_WAIT * FLOOR) >= (INIT * HORIZON). In addition, the combined system
// must be able to absorb a retry-storm, the size of which is unknown.
const DEFAULT_RATE_MAX_WAIT_MS: u64 = 10_000;
// Settle frequency, where AIMD updates its state based on the observed traffic.
const TICK_INTERVAL: Duration = Duration::from_secs(1);
// Period during which additional cuts will be suppressed.
const REFRACTORY: Duration = Duration::from_secs(1);
// Period after last cut where growth beyond last_max is deferred.
const PROBE_QUIET_PERIOD: Duration = Duration::from_secs(2);

// Load and freshness parameters.
// Utilization bound for growth.
const SATURATION_THRESHOLD: f64 = 0.8;
// EWMA weight of fresh goodput sample.
const SUCCESS_SMOOTH: f64 = 0.8;
// Decay multiplier towards init on idle, per tick.
const IDLE_DECAY: f64 = 0.9;

// Wake-latency tick. Bounds how long a parked waiter oversleeps past its
// arithmetic due-time; does NOT affect pacing accuracy.
pub(crate) const HTTP_RATE_LIMIT_WAKE_TICK: Duration = Duration::from_millis(1);

// Used for sharing values between concurrency controller and rate-limiter.
// Captures the f64 value in bits.
pub type RateCell = Arc<AtomicU64>;

/// What happens to the learned rate state when a store inits. Important
/// when an object store rebuilds on error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(unused)]
pub(crate) enum InitPolicy {
    /// Reset to init values: a rebuild is a deliberate clean slate.
    SetToInit,
    /// Reset to floor values: a rebuild falls back to a conservative start.
    SetToFloor,
    /// Leave the rate cells untouched: inherit the learned rate, if any.
    Inherit,
}

impl InitPolicy {
    fn target_rate(self, config: &DirectionalRateLimitConfig) -> Option<f64> {
        match self {
            Self::SetToInit => Some(config.init_rate),
            Self::SetToFloor => Some(config.floor_rate),
            Self::Inherit => None,
        }
    }
}

/// Rate-limit parameters.
#[derive(Debug, Clone)]
pub(crate) struct DirectionalRateLimitConfig {
    // HTTP requests per second (rps).
    pub(crate) init_rate: f64,
    pub(crate) floor_rate: f64,
    pub(crate) ceiling_rate: f64,
    pub(crate) horizon: Duration,
    pub(crate) max_wait: Duration,
    pub(crate) init_policy: InitPolicy,
}

/// Rate-limit parameters.
#[derive(Debug, Clone)]
pub(crate) struct RateLimitConfig {
    pub(crate) read: DirectionalRateLimitConfig,
    pub(crate) write: DirectionalRateLimitConfig,
}

impl From<CloudRateLimitConfig> for RateLimitConfig {
    fn from(value: CloudRateLimitConfig) -> Self {
        fn to_rate_limit_config(
            config: &CloudDirectionalRateLimitConfig,
        ) -> DirectionalRateLimitConfig {
            DirectionalRateLimitConfig {
                init_rate: config.init_rate.map_or(DEFAULTS.init_rate, |r| r as f64),
                floor_rate: config.floor_rate.map_or(DEFAULTS.floor_rate, |r| r as f64),
                ceiling_rate: config
                    .ceiling_rate
                    .map_or(DEFAULTS.ceiling_rate, |r| r as f64),
                horizon: DEFAULTS.horizon,
                max_wait: DEFAULTS.max_wait,
                init_policy: DEFAULTS.init_policy,
            }
        }

        let read_config = to_rate_limit_config(&value.read);
        let write_config = to_rate_limit_config(&value.write);

        return RateLimitConfig {
            read: read_config,
            write: write_config,
        };

        static DEFAULTS: LazyLock<DirectionalRateLimitConfig> =
            LazyLock::new(|| DirectionalRateLimitConfig {
                init_rate: parse_env_var(DEFAULT_INIT_RATE, "POLARS_CLOUD_INIT_RATE"),
                floor_rate: {
                    let floor_rate = parse_env_var(DEFAULT_FLOOR_RATE, "POLARS_CLOUD_FLOOR_RATE");
                    assert!(floor_rate > 0.0);
                    floor_rate
                },
                ceiling_rate: parse_env_var(DEFAULT_CEILING_RATE, "POLARS_CLOUD_CEILING_RATE"),
                horizon: Duration::from_millis(parse_env_var(
                    DEFAULT_RATE_HORIZON_MS,
                    "POLARS_CLOUD_RATE_HORIZON_MS",
                )),
                max_wait: Duration::from_millis(parse_env_var(
                    DEFAULT_RATE_MAX_WAIT_MS,
                    "POLARS_CLOUD_RATE_MAX_WAIT_MS",
                )),
                init_policy: InitPolicy::Inherit,
            });

        fn parse_env_var<T: FromStr>(default: T, name: &'static str) -> T {
            std::env::var(name).map_or(default, |x| {
                x.parse::<T>()
                    .ok()
                    .unwrap_or_else(|| panic!("invalid value for {name}: {x}"))
            })
        }
    }
}

/// Read-only view of the pacing budget including the learned rate-limit signal
/// for internal consumers (e.g., ConcurrencyController).
/// The cells contains learned rates (f64 bits in an AtomicU64), and are updated
/// by the AIMD loop only.
/// Valid for the process lifetime: object store rebuilds can rotate behind the
/// cells rather than replacing them.
#[derive(Debug, Clone)]
pub struct PacingBudget {
    rate_bits: RateCell,
    horizon: Duration,
}

impl PacingBudget {
    pub fn rate(&self) -> f64 {
        f64::from_bits(self.rate_bits.load(Relaxed))
    }

    pub fn horizon(&self) -> Duration {
        self.horizon
    }

    /// Helper so downstream doesn't have to re-implement the formula
    pub fn request_budget(&self, bdp: f64) -> f64 {
        bdp.min(self.rate() * self.horizon().as_secs_f64())
    }
}
// Persist rate-limit state when rebuilding an object_store with InitPolicy::Inherit.
#[derive(Debug, Clone)]
pub(crate) struct RateState {
    pub rate_bits: RateCell,
    pub max_bits: RateCell, // NaN represents None
}

impl RateState {
    pub fn new(init_rate: f64) -> Self {
        Self {
            rate_bits: Arc::new(AtomicU64::new(init_rate.to_bits())),
            max_bits: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
        }
    }

    #[inline]
    pub fn get_rate(&self) -> f64 {
        f64::from_bits(self.rate_bits.load(Relaxed))
    }

    #[inline]
    pub fn set_rate(&self, rate: f64) {
        self.rate_bits.store(rate.to_bits(), Relaxed);
    }

    #[inline]
    pub fn get_last_max(&self) -> Option<f64> {
        // NaN means no value.
        let value = f64::from_bits(self.max_bits.load(Relaxed));
        (!value.is_nan()).then_some(value)
    }

    #[inline]
    pub fn set_last_max(&self, max: Option<f64>) {
        let bits = max.unwrap_or(f64::NAN).to_bits();
        self.max_bits.store(bits, Relaxed);
    }

    pub fn reset(&self, init_rate: f64) {
        self.set_rate(init_rate);
        self.set_last_max(None);
    }
}

/// Builder-owned. State that may survive object store rebuilds.
#[derive(Debug)]
struct RateLimitState {
    read_state: RateState,
    write_state: RateState,
}

impl RateLimitState {
    pub(crate) fn new(config: &RateLimitConfig) -> Self {
        Self {
            read_state: RateState::new(config.read.init_rate),
            write_state: RateState::new(config.write.init_rate),
        }
    }

    /// Initialize the cell values from prior state on object store rebuild.
    pub(crate) fn apply_init_policy(&self, config: &RateLimitConfig) {
        fn apply_one(state: &RateState, config: &DirectionalRateLimitConfig) {
            if let Some(rate) = config.init_policy.target_rate(config) {
                state.reset(rate);
            }
        }

        apply_one(&self.read_state, &config.read);
        apply_one(&self.write_state, &config.write);
    }
}

#[derive(Debug)]
pub(crate) struct RateLimiter {
    pub(crate) config: RateLimitConfig,
    state: RateLimitState,
}

impl RateLimiter {
    pub(crate) fn new(config: RateLimitConfig) -> Self {
        let state = RateLimitState::new(&config);
        Self { config, state }
    }

    /// Initialize the cell values. Called at every store (re)build, before
    /// constructing the new PacedHttpConnector.
    pub(crate) fn apply_init_policy(&self) {
        self.state.apply_init_policy(&self.config);
    }

    // Read-only view of pacing budget for 'read' based on learned rate.
    // Targeted at upstream consumers.
    pub(crate) fn read_budget(&self) -> PacingBudget {
        PacingBudget {
            rate_bits: Arc::clone(&self.state.read_state.rate_bits),
            horizon: self.config.read.horizon,
        }
    }

    // Read-only view of pacing budget for 'write' based on learned rate.
    // Targeted at upstream consumers.
    #[allow(unused)]
    pub(crate) fn write_budget(&self) -> PacingBudget {
        PacingBudget {
            rate_bits: Arc::clone(&self.state.write_state.rate_bits),
            horizon: self.config.write.horizon,
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum Regime {
    // No knowledge about ceiling or cut. Fast-ramp.
    Search,
    // Recover from a congestion event.
    Recover { until: Instant, anchor: f64 },
    // Track towards a learned ceiling and probe above.
    Track { anchor: f64 },
}

// Adaptive Increase Multiplicative Decrease (AIMD) state.
// Note: the initial fast_ramp is multiplicative, not additive.
// Cold path: state gets updated every tick interval.
#[derive(Debug)]
struct AimdState {
    // How to drive the rate based on what has been observed.
    regime: Regime,

    // Source of truth for the learned request rates in requests per second (rps).
    // Represented as f64 in bits and shared as Atomic cells.
    shared: RateState,
    // Exponentially weighted moving average (EWMA) of the success rate.
    success_rate: Option<f64>,
    // Start of window, where window is the interval between 2 ticks.
    window_start: Option<Instant>,
    // Last cut time.
    last_cut_time: Option<Instant>,

    // Pacer statistics at last tick.
    prev_admitted: u64,
    prev_denied: u64,
    prev_resp_succeeded: u64,
    prev_resp_throttled: u64,
}

impl AimdState {
    #[inline]
    pub fn rate(&self) -> f64 {
        self.shared.get_rate()
    }

    #[inline]
    pub fn set_rate(&mut self, rate: f64) {
        self.shared.set_rate(rate);
    }

    #[inline]
    pub fn last_max(&self) -> Option<f64> {
        self.shared.get_last_max()
    }

    #[inline]
    pub fn set_last_max(&mut self, max: Option<f64>) {
        self.shared.set_last_max(max);
    }

    /// The success_rate is the observed 'goodput' signal.
    fn update_success_rate(&mut self, successes: u64, elapsed_s: f64) {
        // Too short to be statistically meaningful (also guards div-by-zero).
        if elapsed_s < 0.5 * TICK_INTERVAL.as_secs_f64() {
            return;
        }

        if successes == 0 {
            return;
        }

        let success_rate = successes as f64 / elapsed_s;
        self.success_rate = Some(match self.success_rate {
            None => success_rate,
            Some(rate) => SUCCESS_SMOOTH * success_rate + (1.0 - SUCCESS_SMOOTH) * rate,
        });
    }
}

// Lock-free metrics and time window filter to guard the Mutex, warm path.
#[derive(Debug)]
pub(crate) struct PacerSignal {
    // Fast-path view of window end.
    window_end_ns: AtomicU64,
    // Advisory refractory pre-check (0 = never).
    last_cut_ns: AtomicU64,
    // Cumulative HTTP response counters, since epoch.
    resp_succeeded: AtomicU64,
    resp_throttled: AtomicU64,
}

#[derive(Debug)]
pub(crate) struct AdaptiveRateController {
    epoch: Instant,
    // Hot path, lock-free atomics.
    //  TBD - do we need an Arc?
    pacer: Arc<Pacer>,
    // Warm signal path, lock-free atomics.
    signal: PacerSignal,
    // Cold path, change state and rate.
    state: Mutex<AimdState>,
    label: &'static str,
    id: u32,
    config: DirectionalRateLimitConfig,
}

impl AdaptiveRateController {
    fn new(
        label: &'static str,
        id: u32,
        shared: RateState,
        config: DirectionalRateLimitConfig,
    ) -> Self {
        let epoch = Instant::now();

        let regime = match shared.get_last_max() {
            Some(anchor) => Regime::Track { anchor },
            None => Regime::Search,
        };

        let token_bucket = Arc::new(TokenBucket::new(shared.rate_bits.clone()));
        let pacer = Pacer::start(token_bucket, config.max_wait);

        let signal = PacerSignal {
            window_end_ns: AtomicU64::new(u64::MAX),
            last_cut_ns: AtomicU64::new(0),
            resp_succeeded: AtomicU64::new(0),
            resp_throttled: AtomicU64::new(0),
        };

        let state = Mutex::new(AimdState {
            regime,
            shared,
            success_rate: None,
            window_start: None,
            last_cut_time: None,
            prev_admitted: 0,
            prev_denied: 0,
            prev_resp_succeeded: 0,
            prev_resp_throttled: 0,
        });

        Self {
            epoch,
            pacer,
            signal,
            state,
            label,
            id,
            config,
        }
    }

    #[inline]
    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    #[inline]
    fn mark_first_traffic(&self) {
        // The window origin is FIRST TRAFFIC, not construction: the pipeline
        // takes an unknown time to spin up, and a window that spans the idle
        // prologue measures emptiness (4 successes / 533ms at cold start).
        if self.signal.window_end_ns.load(Relaxed) == u64::MAX {
            let now_ns = self.now_ns();
            self.signal
                .window_end_ns
                .store(now_ns + TICK_INTERVAL.as_nanos() as u64, Relaxed);
            self.state.lock().unwrap().window_start = Some(Instant::now());
        }
    }

    fn on_congestion(&self) {
        self.signal.resp_throttled.fetch_add(1, Relaxed);

        self.mark_first_traffic();
        self.maybe_settle();

        // Lock-free fast-path (1): advisory refractory pre-check, avoid Mutex storm.
        let now_ns = self.now_ns();
        let last = self.signal.last_cut_ns.load(Relaxed);
        if last != 0 && now_ns.saturating_sub(last) < REFRACTORY.as_nanos() as u64 {
            return;
        }

        // Locking fast-path (2): check the authoritative `last_cut` in AimdState.
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        if state
            .last_cut_time
            .is_some_and(|t| now.duration_since(t) < REFRACTORY)
        {
            return;
        }

        // Calculate anchor rate, which is our (conservative) estimate for the unknown rate-limit enforced
        // by the back-end.
        let (anchor, beta) = match state.success_rate {
            None => (state.rate(), BETA_NO_SIGNAL),
            Some(success_rate) => (
                state.rate().min(success_rate.max(self.config.floor_rate)),
                BETA,
            ),
        };

        // Activate new rate and update state.
        state.set_rate((anchor * beta).max(self.config.floor_rate));
        state.last_cut_time = Some(now);
        self.signal.last_cut_ns.store(now_ns, Relaxed);

        // Unconditionally move into Recover on every cut.
        state.regime = Regime::Recover {
            until: now.checked_add(PROBE_QUIET_PERIOD).unwrap(),
            anchor,
        };

        // Log.
        if *LOG_HTTP_RATE_LIMIT {
            eprintln!(
                "[http rate_limit #{}_{} {}] ..cut (anchored): rate: {:.1}, success_rate: {:.1}, last_max: {:.1}",
                self.id,
                self.label,
                Utc::now(),
                state.rate(),
                state.success_rate.unwrap_or_default(),
                state.last_max().unwrap_or_default(),
            );
        }
    }

    fn on_success(&self) {
        self.signal.resp_succeeded.fetch_add(1, Relaxed);

        self.mark_first_traffic();
        self.maybe_settle();
    }

    fn on_other(&self) {
        self.mark_first_traffic();
        self.maybe_settle();
    }

    #[inline]
    fn maybe_settle(&self) {
        // Lock-free fast path when settlement is not due.

        if self.now_ns() >= self.signal.window_end_ns.load(Relaxed) {
            self.settle_lazy_tick();
        }
    }

    // Settle the rate-limiter after every tick interval, on the first success response.
    fn settle_lazy_tick(&self) {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        let now_ns = self.now_ns();

        // Another success event may have settled while we waited on the lock.
        let elapsed = now.duration_since(state.window_start.unwrap());
        let ticks = (elapsed.as_secs_f64() / TICK_INTERVAL.as_secs_f64()).floor();
        if ticks < 1.0 {
            return;
        }

        // Update AIMD stats.
        let admitted = self.pacer.admitted();
        let win_admitted = admitted.saturating_sub(state.prev_admitted);
        state.prev_admitted = admitted;

        let denied = self.pacer.denied.load(Relaxed);
        let win_denied = denied.saturating_sub(state.prev_denied);
        state.prev_denied = denied;

        let succeeded = self.signal.resp_succeeded.load(Relaxed);
        let win_succeeded = succeeded.saturating_sub(state.prev_resp_succeeded);
        state.prev_resp_succeeded = succeeded;

        let throttled = self.signal.resp_throttled.load(Relaxed);
        let win_throttled = throttled.saturating_sub(state.prev_resp_throttled);
        state.prev_resp_throttled = throttled;

        // Growth conditions.
        let elapsed_s = elapsed.as_secs_f64();
        let queued = self.pacer.queue_depth() > 0;
        let util_bound = win_admitted as f64 >= SATURATION_THRESHOLD * state.rate() * elapsed_s;

        // Update signal, if any.
        state.update_success_rate(win_succeeded, elapsed_s);

        // Goodput is a always a lower bound on capacity, so an observation may always
        // increase. It may only decrease when the system was pressure-tested (i.e., demand saturated).
        if let Some(observed) = state.success_rate {
            let pressure_tested =
                win_admitted as f64 / elapsed_s >= state.last_max().unwrap_or(0.0);
            let new_max = match state.last_max() {
                Some(prev) if !pressure_tested => prev.max(observed),
                _ => observed,
            };
            state.set_last_max(Some(new_max.max(self.config.floor_rate)));
        }

        // Evaluate and apply rate rate and regime changes.
        let verdict = {
            let rate = state.rate();
            let init_rate = self.config.init_rate;
            let floor_rate = self.config.floor_rate;
            let ceiling_rate = self.config.ceiling_rate;

            if let Regime::Recover { until, anchor } = state.regime
                && now > until
            {
                state.regime = Regime::Track { anchor }
            };

            if win_admitted + win_denied == 0 {
                // No demand - decay toward seed only if we are currently above it.
                if rate > init_rate {
                    let rate = init_rate + (state.rate() - init_rate) * IDLE_DECAY.powf(ticks);
                    state.set_rate(rate.clamp(floor_rate, ceiling_rate));
                }
                // Note: we leave last_max as-is which is not always right.
                "decay (idle)"
            } else {
                match state.regime {
                    Regime::Search => {
                        // Cold start -> aggressive exponential growth, if earned
                        // Note. One may expect `.powf(ticks)`; this is deliberately omitted.
                        if util_bound && win_throttled == 0 {
                            state.set_rate((rate * FAST_RAMP_FACTOR).min(ceiling_rate));
                            "increase (fast_ramp)"
                        } else {
                            "hold (app-limited)"
                        }
                    },
                    Regime::Recover { .. } => {
                        // No-op, rate changes were handled in `on_congestion`
                        "hold (recover)"
                    },
                    Regime::Track { anchor } => {
                        // Below anchor + backlogged/saturated -> fast reclaim up to our known-good ceiling
                        if rate < anchor && util_bound {
                            state.set_rate((rate * FAST_RAMP_FACTOR).min(anchor).min(ceiling_rate));
                            "increase (reclaim)"
                        } else
                        // Below anchor but not hitting capacity bounds -> wait for more traffic
                        if rate < anchor {
                            "hold (app-limited)"
                        } else
                        // At or above anchor + saturated + past quiet period -> cautious additive probing
                        if util_bound {
                            let delta = (PROBE_FRACTION * anchor).max(PROBE_MIN);
                            state.set_rate((rate + delta).min(ceiling_rate));
                            "increase (probe)"
                        } else
                        // At/above anchor and backlog is draining out.
                        if queued {
                            "hold (draining)"
                        } else {
                            // Fallback for unutilized capacity above the anchor.
                            "hold (app-limited)"
                        }
                    },
                }
            }
        };

        // Update state.
        state.window_start = Some(now);
        self.signal
            .window_end_ns
            .store(now_ns + TICK_INTERVAL.as_nanos() as u64, Relaxed);

        // Logging.
        if *LOG_HTTP_RATE_LIMIT {
            eprintln!(
                "[http rate_limit #{}_{} {}] {}: rate: {:.1}, success_rate_ewma: {:.1}, last_max: {:.1}, \
                    elapsed: {:.1}s, win_admit: {}, win_deny: {}, win_success: {}, win_throttle: {},  q_depth: {}",
                self.id,
                self.label,
                Utc::now(),
                verdict,
                state.rate(),
                state.success_rate.unwrap_or_default(),
                state.last_max().unwrap_or_default(),
                elapsed_s,
                win_admitted,
                win_denied,
                win_succeeded,
                win_throttled,
                self.pacer.queue_depth(),
            );
        }
    }
}

/// Grouped rate-limit controller as used by a logical object_store instance.
#[derive(Debug)]
pub(crate) struct RateController {
    // GET, HEAD
    pub read: AdaptiveRateController,
    // PUT, POST, DELETE
    pub write: AdaptiveRateController,
}

impl RateController {
    pub(crate) fn new(rate_limiter: &RateLimiter) -> Arc<Self> {
        // NOTE: A rebuilt object_store can share one rate state.
        // Acceptable transient situation.
        let id = CONTROLLER_ID.fetch_add(1, Relaxed);
        Arc::new(Self {
            read: AdaptiveRateController::new(
                "read",
                id,
                rate_limiter.state.read_state.clone(),
                rate_limiter.config.read.clone(),
            ),
            write: AdaptiveRateController::new(
                "write",
                id,
                rate_limiter.state.write_state.clone(),
                rate_limiter.config.write.clone(),
            ),
        })
    }

    fn class(&self, req: &HttpRequest) -> &AdaptiveRateController {
        match req.method().as_str() {
            "GET" | "HEAD" => &self.read,
            _ => &self.write,
        }
    }
}

#[derive(Debug)]
struct PacerBusy {
    est_wait_ms: u64,
}
impl fmt::Display for PacerBusy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "internal pacer denied: estimated wait {}ms exceeds bound",
            self.est_wait_ms
        )
    }
}
impl std::error::Error for PacerBusy {}

// PACER INCL WAITER_QUEUE

// The pacer is responsible for admissin/denial into the `TokenBucket` annex
// `WaiterQueue`. Once admitted, it paces the requests to the prescribed rate.

#[derive(Debug, Default)]
struct WaiterQueue {
    parked: Mutex<VecDeque<oneshot::Sender<()>>>,
    /// Lock-free mirror of parked.len(): read by the fast path (anti-barge),
    /// the population coupling, and telemetry.
    depth: AtomicU64,
}

impl WaiterQueue {
    fn park(&self) -> oneshot::Receiver<()> {
        let (tx, rx) = oneshot::channel();
        let mut q = self.parked.lock().unwrap();
        q.push_back(tx);
        self.depth.store(q.len() as u64, Relaxed);
        rx
    }

    /// Pop the head and deliver a grant. Returns false when the queue is
    /// empty. A dead (cancelled) head consumes no grant: we skip it and keep
    /// the token for the next live waiter.
    fn grant_one(&self) -> bool {
        let mut q = self.parked.lock().unwrap();
        while let Some(tx) = q.pop_front() {
            self.depth.store(q.len() as u64, Relaxed);
            if tx.send(()).is_ok() {
                return true; // grant delivered
            }
            // Cancelled waiter: token still in hand, try the next.
        }
        false
    }

    fn depth(&self) -> u64 {
        self.depth.load(Relaxed)
    }
}

/// Lock-free hot path enforcing the rate-limit by pacing requests through the token bucket.
#[derive(Debug)]
pub struct Pacer {
    bucket: Arc<TokenBucket>,
    queue: Arc<WaiterQueue>,
    max_wait: Duration,
    // Decided to admit.
    admitted: AtomicU64,
    // Decided not to park.
    denied: AtomicU64,
}

impl Pacer {
    /// Construct and spawn the wake tick. The tick is per-pacer.
    pub fn start(bucket: Arc<TokenBucket>, max_wait: Duration) -> Arc<Self> {
        let pacer = Arc::new(Self {
            bucket,
            queue: Arc::new(WaiterQueue::default()),
            max_wait,
            admitted: AtomicU64::new(0),
            denied: AtomicU64::new(0),
        });
        Self::spawn_wake_tick(Arc::downgrade(&pacer));
        pacer
    }

    /// Admit a request to the pacer, which may get queued internally. Returns a
    /// PacerBusy Error when the estimated wait exceeds the wait bound.
    pub async fn admit(&self) -> Result<(), HttpError> {
        // Anti-barge plus fast path.
        if self.queue.depth() == 0 && self.bucket.try_acquire().is_ok() {
            self.admitted.fetch_add(1, Relaxed);
            return Ok(());
        }

        // JIT rejection: price the line, before parking.
        let est_wait =
            Duration::from_secs_f64((self.queue.depth() + 1) as f64 / self.bucket.rate());
        if est_wait > self.max_wait {
            self.denied.fetch_add(1, Relaxed);
            return Err(HttpError::new(
                HttpErrorKind::Timeout, // retryable by object_store
                PacerBusy {
                    est_wait_ms: est_wait.as_millis() as u64,
                },
            ));
        }

        // Park. Wake when granted.
        let rx = self.queue.park();

        // Wait. On Err, allow through unpaced when WaiterQueue/Pacer is torn down.
        let _ = rx.await;
        self.admitted.fetch_add(1, Relaxed);
        Ok(())
    }

    pub fn queue_depth(&self) -> u64 {
        self.queue.depth()
    }

    pub fn admitted(&self) -> u64 {
        self.admitted.load(Relaxed)
    }

    pub fn bucket(&self) -> &Arc<TokenBucket> {
        &self.bucket
    }

    /// The single timer in the design. Each tick: while waiters exist AND the
    /// bucket yields a token, hand grants to the FIFO head. Weak handle so a
    /// dropped pacer (store rebuild) tears its tick down with it.
    fn spawn_wake_tick(gate: std::sync::Weak<Self>) {
        ASYNC.spawn(async move {
            let mut tick = tokio::time::interval(HTTP_RATE_LIMIT_WAKE_TICK);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let Some(g) = gate.upgrade() else { return };

                while g.queue.depth() > 0 {
                    if g.bucket.try_acquire().is_err() {
                        break;
                    }
                    if !g.queue.grant_one() {
                        // Queue drained between the depth check and the grant
                        // (or all-cancelled): one token over-taken.
                        break;
                    }
                }
            }
        });
    }
}

// OBJECT_STORE HTTP_SERVICE MIDDLEWARE

#[derive(Debug)]
pub(crate) struct PacedHttpConnector {
    inner: Box<dyn HttpConnector>,
    controller: Arc<RateController>,
}

impl PacedHttpConnector {
    pub(crate) fn new(inner: Box<dyn HttpConnector>, rate_limiter: &RateLimiter) -> Self {
        // The HTTP Connector may re-use its learned request rate.
        rate_limiter.apply_init_policy();

        Self {
            inner,
            controller: RateController::new(rate_limiter),
        }
    }
}

impl HttpConnector for PacedHttpConnector {
    fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
        let client = self.inner.connect(options)?;
        Ok(HttpClient::new(PacedHttpService {
            inner: client,
            controller: Arc::clone(&self.controller),
        }))
    }
}

#[derive(Debug)]
pub(crate) struct PacedHttpService {
    inner: HttpClient,
    controller: Arc<RateController>,
}

#[async_trait]
impl HttpService for PacedHttpService {
    async fn call(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        let verb_pacer = self.controller.class(&req);

        // Enforce pacing: on admission, wait; on denied, raise an error.
        // Requests get denied when the estimated wait is too high (aka 'shed').
        // The object_store is responsible for retry.
        verb_pacer.pacer.admit().await?;

        let response = self.inner.execute(req).await;
        match &response {
            Ok(r) if r.status().as_u16() == 429 || r.status().as_u16() == 503 => {
                verb_pacer.on_congestion()
            },
            Ok(r) if r.status().is_success() => verb_pacer.on_success(),
            _ => verb_pacer.on_other(),
        }

        response
    }
}
