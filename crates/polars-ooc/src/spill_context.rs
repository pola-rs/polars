use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Instant;

use polars_utils::relaxed_cell::RelaxedCell;
use rand::RngExt;
use rand::rngs::ThreadRng;
use thread_local::ThreadLocal;

use crate::{DynSpillToken, SpillToken, Spillable, memory_manager};

#[derive(Default)]
struct LocalSpillQueue {
    tokens: VecDeque<(Weak<dyn DynSpillToken>, u64)>,
    retain_amort: usize,
}

impl LocalSpillQueue {
    pub fn push_back(&mut self, token: &Arc<dyn DynSpillToken>, id: u64) {
        self.gc();
        if token.current_registration_id() == id {
            self.tokens.push_front((Arc::downgrade(token), id));
        }
    }

    pub fn push_front(&mut self, token: &Arc<dyn DynSpillToken>, id: u64) {
        self.gc();
        if token.current_registration_id() == id {
            self.tokens.push_front((Arc::downgrade(token), id));
        }
    }

    pub fn pop_front(&mut self) -> Option<(Arc<dyn DynSpillToken>, u64)> {
        loop {
            let (weak, id) = self.tokens.pop_front()?;
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some((token, id));
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<(Arc<dyn DynSpillToken>, u64)> {
        loop {
            let (weak, id) = self.tokens.pop_back()?;
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some((token, id));
            }
        }
    }

    pub fn pop_random(&mut self, rng: &mut ThreadRng) -> Option<(Arc<dyn DynSpillToken>, u64)> {
        while !self.tokens.is_empty() {
            let idx = rng.random_range(0..self.tokens.len());
            let (weak, id) = self.tokens.swap_remove_back(idx).unwrap();
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some((token, id));
            }
        }
        None
    }

    fn gc(&mut self) {
        self.retain_amort += 2; // Grows twice as fast as push.
        if self.retain_amort >= self.tokens.len() {
            self.retain_amort = 0;
            self.tokens.retain(|(token, id)| {
                token
                    .upgrade()
                    .is_some_and(|t| t.current_registration_id() == *id)
            });
        }
    }
}

pub trait SpillContext: Send + Sync + 'static {
    fn stats(&self) -> &Arc<SpillContextStatistics>;
    fn pop(&self) -> Vec<(Arc<dyn DynSpillToken>, u64)>;
    fn reinsert(&self, token: &Arc<dyn DynSpillToken>, id: u64);
}

pub trait ParameterFreeSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
        Self: Sized;
}

/// A context that spills the most-recently registered spillable when asked.
pub struct MostRecentSpillContext {
    local: ThreadLocal<RwLock<LocalSpillQueue>>,
    stats: Arc<SpillContextStatistics>,
}

impl MostRecentSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
            stats: Arc::default(),
        });
        memory_manager().register_ctx(&slf);
        slf
    }
}

impl SpillContext for MostRecentSpillContext {
    fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }

    fn pop(&self) -> Vec<(Arc<dyn DynSpillToken>, u64)> {
        let mut out = Vec::new();
        for local_lock in self.local.iter() {
            if let Ok(mut local) = local_lock.try_write() {
                out.extend(local.pop_back());
            }
        }
        out
    }

    fn reinsert(&self, token: &Arc<dyn DynSpillToken>, id: u64) {
        // Reinsertions always act like least recent, so we use push_front.
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_front(token, id);
    }
}

impl ParameterFreeSpillContext for MostRecentSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.new_registration_id());
    }
}

impl Debug for MostRecentSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MostRecentSpillContext").finish()
    }
}

/// A context that spills the least-recently registered spillable when asked.
pub struct LeastRecentSpillContext {
    local: ThreadLocal<RwLock<LocalSpillQueue>>,
    stats: Arc<SpillContextStatistics>,
}

impl LeastRecentSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
            stats: Arc::default(),
        });
        memory_manager().register_ctx(&slf);
        slf
    }
}

impl SpillContext for LeastRecentSpillContext {
    fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }

    fn pop(&self) -> Vec<(Arc<dyn DynSpillToken>, u64)> {
        let mut out = Vec::new();
        for local_lock in self.local.iter() {
            if let Ok(mut local) = local_lock.try_write() {
                out.extend(local.pop_front());
            }
        }
        out
    }

    fn reinsert(&self, token: &Arc<dyn DynSpillToken>, id: u64) {
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(token, id);
    }
}

impl ParameterFreeSpillContext for LeastRecentSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.new_registration_id());
    }
}

impl Debug for LeastRecentSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeastRecentSpillContext").finish()
    }
}

/// A context that spills a random registered spillable when asked.
pub struct RandomSpillContext {
    local: ThreadLocal<RwLock<LocalSpillQueue>>,
    stats: Arc<SpillContextStatistics>,
}

impl RandomSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
            stats: Arc::default(),
        });
        memory_manager().register_ctx(&slf);
        slf
    }
}

impl SpillContext for RandomSpillContext {
    fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }

    fn pop(&self) -> Vec<(Arc<dyn DynSpillToken>, u64)> {
        let mut out = Vec::new();
        let mut rng = rand::rng();
        for local_lock in self.local.iter() {
            if let Ok(mut local) = local_lock.try_write() {
                out.extend(local.pop_random(&mut rng));
            }
        }
        out
    }

    fn reinsert(&self, token: &Arc<dyn DynSpillToken>, id: u64) {
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(token, id);
    }
}

impl ParameterFreeSpillContext for RandomSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.new_registration_id());
    }
}

impl Debug for RandomSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSpillContext").finish()
    }
}

fn f32_pair_to_u64(a: f32, b: f32) -> u64 {
    ((a.to_bits() as u64) << 32) | b.to_bits() as u64
}

fn u64_to_f32_pair(u: u64) -> (f32, f32) {
    (f32::from_bits((u >> 32) as u32), f32::from_bits(u as u32))
}

// Used to normalize divisor to avoid absurdly high scores. Set to 1us.
const BASE_IO_TIME: f64 = 1e-6;
const UNEXPLORED_SCORE: f32 = 1e10_f32;
const UNEXPLORED_VARIANCE: f32 = 1e5_f32;
const STABLE_WEIGHT_THRESHOLD: f64 = 0.1; // Weights under this are considered unreliable (division).
const EXPLORED_WEIGHT_THRESHOLD: f64 = 1.0 + STABLE_WEIGHT_THRESHOLD; // Just over 1 so sample variance correction is stable.
const UNSPILL_EVENT_HALF_LIFE_SEC: f64 = 5.0;
const NANOSECONDS_IN_SECOND: f64 = 1e9;

pub struct SpillContextStatistics {
    score_cache: RelaxedCell<u64>,
    stats: Mutex<Statistics>,
}

impl Default for SpillContextStatistics {
    fn default() -> Self {
        Self {
            // TODO: starting score based on context.
            score_cache: RelaxedCell::new_u64(f32_pair_to_u64(
                UNEXPLORED_SCORE,
                UNEXPLORED_VARIANCE,
            )),
            stats: Mutex::default(),
        }
    }
}

impl SpillContextStatistics {
    // Returns the discounted mean and discounted variance of the 'score' of
    // this context. The score is the number of spilled byte-seconds divided by
    // the IO time in seconds. The discount is a time-based exponential decay
    // which assigns lower weight to older spill events (with a half-life of
    // UNSPILL_EVENT_HALF_LIFE_SEC), and full weight to current spills.
    pub fn score(&self) -> (f32, f32) {
        // Try to re-compute, if lock is taken just take cached value.
        if let Ok(mut stats) = self.stats.try_lock() {
            let (mean, var) = stats.score();
            self.score_cache.store(f32_pair_to_u64(mean, var));
            (mean, var)
        } else {
            u64_to_f32_pair(self.score_cache.load())
        }
    }

    pub fn add_failed_spill(&self, spill_start: Instant) {
        let mut stats = self.stats.lock().unwrap();
        let spill_time_sec = spill_start.elapsed().as_secs_f64();
        stats.spill_time += spill_time_sec;
        stats.failed_spills += 1;
        stats.add_bandit_event(0, spill_time_sec, 0.0, 0.0);
    }

    /// Returns the number of nanoseconds the spilling took, as well as the
    /// Instant from which we consider this value to be spilled. Both must be
    /// given back as arguments to `add_unspill`, to prevent accidental
    /// statistics drift and keep the accumulators precise.
    pub fn add_successful_spill(&self, n_bytes: usize, spill_start: Instant) -> (u64, Instant) {
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        stats.step_time(now); // Important: step time before mutating.

        let spill_time = now - spill_start;
        let spill_time_ns = spill_time.as_nanos() as u64;
        stats.spill_time += spill_time.as_secs_f64();
        stats.successful_spills += 1;
        stats.active_spills += 1;
        stats.bytes_currently_spilled += n_bytes as u64;
        stats.active_spills_io_time_ns += spill_time_ns;
        (spill_time_ns, now)
    }

    pub fn add_unspill(
        &self,
        n_bytes: usize,
        spill_time_ns: u64,
        spilled_start: Instant,
        unspill_start: Instant,
    ) {
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        stats.step_time(now); // Important: step time before mutating.

        let spilled_time = unspill_start - spilled_start;
        let unspill_time = now - unspill_start;
        stats.unspill_time += unspill_time.as_secs_f64();
        stats.active_spills -= 1;
        stats.bytes_currently_spilled -= n_bytes as u64;
        stats.active_spills_io_time_ns -= spill_time_ns;

        // The unspill time was also included in active_spilled_byte_nanoseconds
        // due to our consistent stepping of time, so to cancel everything out
        // we have to count from spilled_start until now. Otherwise
        // active_spilled_byte_nanoseconds would slowly drift upward over time.
        stats.active_spilled_byte_nanoseconds -= n_bytes as u128 * (now - spilled_start).as_nanos();

        stats.add_bandit_event(
            n_bytes as u64,
            spill_time_ns as f64 / NANOSECONDS_IN_SECOND,
            spilled_time.as_secs_f64(),
            unspill_time.as_secs_f64(),
        );
    }
}

struct Statistics {
    // Total stats.
    spilled_byte_seconds: f64,
    spill_time: f64,
    unspill_time: f64,
    successful_spills: u64,
    failed_spills: u64,

    // Stats of currently active spills. These are tracked as integers so they
    // remain exact.
    active_spills: u64,
    active_spilled_byte_nanoseconds: u128,
    active_spills_io_time_ns: u64,
    bytes_currently_spilled: u64,

    // Historical stats for the multi-armed bandit algorithm. These are
    // discounted statistics (meaning they decay over time), where weight is the
    // (decaying) number of observations, and each observation is of the form
    // spilled_byte_seconds / spill_time.
    //
    // The multi-armed bandit algorithm in use is adapted from
    // "Discounted Thompson Sampling for Non-Stationary Bandit Problems" https://arxiv.org/pdf/2305.10718
    // Adaptations made: use direct sample mean/variance, discount based on time
    // rather than arm pulls.
    bandit_weight: f64,
    bandit_reward: f64,
    bandit_sq_reward: f64,

    last_update: Instant,
}

impl Statistics {
    fn step_time(&mut self, now: Instant) {
        let dt = now - self.last_update;
        let dt_s = dt.as_secs_f64();
        let dt_ns = dt.as_nanos();
        self.active_spilled_byte_nanoseconds += self.bytes_currently_spilled as u128 * dt_ns;
        self.spilled_byte_seconds += self.bytes_currently_spilled as f64 * dt_s;

        // Exponentially decay old bandit events.
        let mult = -f64::ln(2.0) / UNSPILL_EVENT_HALF_LIFE_SEC;
        let decay_factor = f64::exp(mult * dt_s);
        self.bandit_weight *= decay_factor;
        self.bandit_reward *= decay_factor;
        self.bandit_sq_reward *= decay_factor;

        self.last_update = now;
    }

    // An event to update the multi-armed bandit algorithm.
    fn add_bandit_event(
        &mut self,
        n_bytes: u64,
        spill_time: f64,
        spilled_time: f64,
        unspill_time: f64,
    ) {
        let spilled_byte_seconds = n_bytes as f64 * spilled_time;
        let io_time = spill_time + unspill_time;
        let r = spilled_byte_seconds / (BASE_IO_TIME + io_time);
        self.bandit_weight += 1.0;
        self.bandit_reward += r;
        self.bandit_sq_reward += r * r;
    }

    fn score(&mut self) -> (f32, f32) {
        self.step_time(Instant::now());

        let mut reward_weight = 0.0;
        let mut reward_sum = 0.0;
        let mut reward_sum_of_sq = 0.0;

        let active_reward_weight = self.active_spills as f64;
        if active_reward_weight >= STABLE_WEIGHT_THRESHOLD {
            // We don't keep track of the variance of active spills, so we
            // just assume it has no variance, that is, each active spill is
            // accounted for as the mean active spill.
            let active_spill_byte_seconds =
                self.active_spilled_byte_nanoseconds as f64 / NANOSECONDS_IN_SECOND;
            let active_spill_io_time = self.active_spills_io_time_ns as f64 / NANOSECONDS_IN_SECOND;
            let active_reward_mean = active_spill_byte_seconds
                / (BASE_IO_TIME * active_reward_weight + active_spill_io_time);
            let active_reward_sum = active_reward_mean * active_reward_weight;

            reward_weight += active_reward_weight;
            reward_sum += active_reward_sum;
            reward_sum_of_sq += active_reward_mean * active_reward_sum;
        }

        if self.bandit_weight >= STABLE_WEIGHT_THRESHOLD {
            reward_weight += self.bandit_weight;
            reward_sum += self.bandit_reward;
            reward_sum_of_sq += self.bandit_sq_reward;
        }

        if reward_weight < EXPLORED_WEIGHT_THRESHOLD {
            return (UNEXPLORED_SCORE, UNEXPLORED_VARIANCE);
        }

        let mean = reward_sum / reward_weight;
        let var = reward_sum_of_sq / reward_weight - mean * mean;
        let corr = reward_weight / (reward_weight - 1.0);
        (mean as f32, (var * corr) as f32)
    }
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            spilled_byte_seconds: 0.0,
            spill_time: 0.0,
            unspill_time: 0.0,
            successful_spills: 0,
            failed_spills: 0,
            last_update: Instant::now(),

            active_spills: 0,
            active_spills_io_time_ns: 0,
            active_spilled_byte_nanoseconds: 0,
            bytes_currently_spilled: 0,

            bandit_weight: 0.0,
            bandit_reward: 0.0,
            bandit_sq_reward: 0.0,
        }
    }
}
