use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};

use polars_utils::pl_str::PlSmallStr;
use polars_utils::relaxed_cell::RelaxedCell;
use rand::rngs::ThreadRng;
use rand::{Rng, RngExt};
use rand_distr::Distribution;
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
            self.tokens.push_back((Arc::downgrade(token), id));
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
    pub fn new(name: PlSmallStr) -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
            stats: Arc::new(SpillContextStatistics::new(name)),
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
    pub fn new(name: PlSmallStr) -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
            stats: Arc::new(SpillContextStatistics::new(name)),
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
    pub fn new(name: PlSmallStr) -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
            stats: Arc::new(SpillContextStatistics::new(name)),
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

// Used to normalize divisor to avoid absurdly high scores. Set to 1us.
const BASE_IO_TIME: f64 = 1e-6;
pub(crate) const UNEXPLORED_SCORE: f64 = 1e30_f64;
const EXPLORED_WEIGHT_THRESHOLD: f64 = 1.1; // Just over 1 so sample variance correction is stable.
const UNSPILL_EVENT_HALF_LIFE_SEC: f64 = 5.0;
const NANOSECONDS_IN_SECOND: f64 = 1e9;

pub struct SpillContextStatistics {
    score_cache: RelaxedCell<u64>,
    stats: Mutex<Statistics>,
    name: PlSmallStr,
}

impl SpillContextStatistics {
    fn new(name: PlSmallStr) -> Self {
        Self {
            // TODO: starting score based on context.
            score_cache: RelaxedCell::new_u64(UNEXPLORED_SCORE.to_bits()),
            stats: Mutex::default(),
            name,
        }
    }
}

impl Drop for SpillContextStatistics {
    fn drop(&mut self) {
        if polars_config::config().ooc_log_metrics() {
            let name = &self.name;
            let stats = self.stats.get_mut().unwrap();
            let relief = stats.spilled_byte_seconds / (1000.0 * 1000.0);
            let spill_io = Duration::from_secs_f64(stats.spill_time);
            let unspill_io = Duration::from_secs_f64(stats.spill_time);
            let spills_tot = stats.successful_spills + stats.failed_spills;
            let spills_succ = 100.0 * stats.successful_spills as f64 / spills_tot as f64;
            let explore_tot = stats.total_explorations;
            let explore_succ = 100.0 * stats.successful_explorations as f64 / explore_tot as f64;

            eprintln!(
                "spill_stats({name}): \
                relief_mb_s({relief:.2}), \
                io(spill={spill_io:.2?}, unspill={unspill_io:.2?}), \
                spill(succ={spills_succ:.1}%, n={spills_tot}), \
                explore(succ={explore_succ:.1}%, n={explore_tot})"
            )
        }
    }
}

struct Statistics {
    // Total stats.
    spilled_byte_seconds: f64,
    spill_time: f64,
    unspill_time: f64,
    successful_spills: u64,
    failed_spills: u64,
    total_explorations: u64,
    successful_explorations: u64,

    // Historical stats for the multi-armed bandit algorithm. These are
    // discounted statistics (meaning they decay over time), where weight is the
    // (decaying) number of observations.
    //
    // Each observation consists of r (relief in byte-seconds) and t (io time).
    // After sampling from this bivariate distribution the score is r / t.
    //
    // The multi-armed bandit algorithm in use is adapted from
    // "Discounted Thompson Sampling for Non-Stationary Bandit Problems" https://arxiv.org/pdf/2305.10718
    // Adaptations made: use direct sample mean/variance, discount based on time
    // rather than arm pulls.
    bandit_weight: f64,
    bandit_r_sum: f64,
    bandit_rr_sum: f64,
    bandit_t_sum: f64,
    bandit_tt_sum: f64,
    bandit_rt_sum: f64,

    bandit_explore_weight: f64,
    bandit_explore_success: f64,

    // Stats of currently active spills. We prefer integers here for accuracy,
    // but have to use floats for the bigger accumulators. Those are only used
    // for variance, so some drift is acceptable.
    active_spills: u64,
    active_spills_bytes: u64,      // sum(bytes)
    active_spills_bytes_ns: u128,  // sum(bytes * ns)
    active_spills_io_time_ns: u64, // sum(io_time)

    // These two are Welford-style accumulators (co-moments) for variance.
    // dp = product of deviations, see moment.rs in polars-compute.
    // r_i = b_i * s_i (relief in byte-seconds).
    active_spills_dp_rr: f64, // sum((r_i - mean(r))^2)
    active_spills_dp_bb: f64, // sum((b_i - mean(b))^2)
    active_spills_dp_rb: f64, // sum((r_i - mean(r)) * (b_i - mean(b)))

    last_update: Instant,
}

impl SpillContextStatistics {
    pub fn name(&self) -> &str {
        &self.name
    }

    // Returns a sample of the expected performance of this context, discounting
    // older data. The score is the number of spilled byte-seconds divided by
    // the IO time in seconds. The discount is a time-based exponential decay
    // which assigns lower weight to older spill events (with a half-life of
    // UNSPILL_EVENT_HALF_LIFE_SEC), and full weight to current spills.
    pub fn sample_score<R: Rng>(&self, rng: &mut R) -> f64 {
        // Try to re-compute, if lock is taken just take cached value.
        if let Ok(mut stats) = self.stats.try_lock() {
            let score = stats.sample_score(rng);
            self.score_cache.store(score.to_bits());
            score
        } else {
            f64::from_bits(self.score_cache.load())
        }
    }

    pub fn start_exploration_event(&self) {
        // We don't bother stepping time here as it's already done in score sampling.
        let mut stats = self.stats.lock().unwrap();
        stats.total_explorations += 1;
        stats.bandit_explore_weight += 1.0;
    }

    pub fn finish_exploration_event(&self, success: bool) {
        // We don't bother stepping time here as it's already done in score sampling.
        let mut stats = self.stats.lock().unwrap();
        stats.successful_explorations += success as u64;
        stats.bandit_explore_success += success as u64 as f64;
    }

    pub fn add_failed_spill(&self, spill_start: Instant) {
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        stats.step_time(now); // Important: step time before mutating.

        let spill_time_sec = (now - spill_start).as_secs_f64();
        stats.spill_time += spill_time_sec;
        stats.failed_spills += 1;
        stats.add_bandit_spill_event(0, spill_time_sec, 0.0, 0.0);
    }

    /// Returns the number of nanoseconds the spilling took, as well as the
    /// Instant from which we consider this value to be spilled. Both must be
    /// given back as arguments to `add_unspill`, to prevent accidental
    /// statistics drift and keep the accumulators precise.
    pub fn add_successful_spill(&self, n_bytes: usize, spill_start: Instant) -> (u64, Instant) {
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        stats.step_time(now); // Important: step time before mutating.

        let mean_b_before = stats.active_spills_bytes as f64 / stats.active_spills.max(1) as f64;
        let mean_r_before = stats.active_spills_bytes_ns as f64
            / (stats.active_spills.max(1) as f64 * NANOSECONDS_IN_SECOND);

        let spill_time = now - spill_start;
        let spill_time_ns = spill_time.as_nanos() as u64;
        stats.spill_time += spill_time.as_secs_f64();
        stats.successful_spills += 1;
        stats.active_spills += 1;
        stats.active_spills_bytes += n_bytes as u64;
        stats.active_spills_io_time_ns += spill_time_ns;

        let mean_b_after = stats.active_spills_bytes as f64 / stats.active_spills as f64;
        let mean_r_after = stats.active_spills_bytes_ns as f64
            / (stats.active_spills as f64 * NANOSECONDS_IN_SECOND);

        let delta_r = -0.0;
        let delta_b = n_bytes as f64;
        stats.active_spills_dp_rr += (delta_r - mean_r_before) * (delta_r - mean_r_after);
        stats.active_spills_dp_rb += (delta_r - mean_r_before) * (delta_b - mean_b_after);
        stats.active_spills_dp_bb += (delta_b - mean_b_before) * (delta_b - mean_b_after);
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

        let mean_b_before = stats.active_spills_bytes as f64 / stats.active_spills as f64;
        let mean_r_before = stats.active_spills_bytes_ns as f64
            / (stats.active_spills as f64 * NANOSECONDS_IN_SECOND);

        let spilled_time = unspill_start - spilled_start;
        let unspill_time = now - unspill_start;
        stats.unspill_time += unspill_time.as_secs_f64();
        stats.active_spills -= 1;
        stats.active_spills_bytes -= n_bytes as u64;
        stats.active_spills_io_time_ns -= spill_time_ns;

        // The unspill time was also included in active_spilled_byte_nanoseconds
        // due to our consistent stepping of time, so to cancel everything out
        // we have to count from spilled_start until now. Otherwise
        // active_spilled_byte_nanoseconds would slowly drift upward over time.
        let elapsed = now - spilled_start;
        let elapsed_s = elapsed.as_secs_f64();

        stats.active_spills_bytes_ns -= n_bytes as u128 * elapsed.as_nanos();

        if stats.active_spills == 0 {
            stats.active_spills_dp_rr = 0.0;
            stats.active_spills_dp_bb = 0.0;
            stats.active_spills_dp_rb = 0.0;
        } else {
            let delta_b = n_bytes as f64;
            let delta_r = delta_b * elapsed_s;
            let mean_b_after = stats.active_spills_bytes as f64 / stats.active_spills as f64;
            let mean_r_after = stats.active_spills_bytes_ns as f64
                / (stats.active_spills as f64 * NANOSECONDS_IN_SECOND);
            stats.active_spills_dp_rr -= (delta_r - mean_r_before) * (delta_r - mean_r_after);
            stats.active_spills_dp_rb -= (delta_r - mean_r_before) * (delta_b - mean_b_after);
            stats.active_spills_dp_bb -= (delta_b - mean_b_before) * (delta_b - mean_b_after);
        }

        stats.add_bandit_spill_event(
            n_bytes as u64,
            spill_time_ns as f64 / NANOSECONDS_IN_SECOND,
            spilled_time.as_secs_f64(),
            unspill_time.as_secs_f64(),
        );
    }
}

impl Statistics {
    fn step_time(&mut self, now: Instant) {
        let dt = now - self.last_update;
        let dt_s = dt.as_secs_f64();
        let dt_ns = dt.as_nanos();
        self.active_spills_bytes_ns += self.active_spills_bytes as u128 * dt_ns;
        self.spilled_byte_seconds += self.active_spills_bytes as f64 * dt_s;

        /*
            This part is rather tricky. Remember that r_i = b_i * s_i.

            We have these two deviations:
                dv_r_i = r_i - mean(r)
                dv_b_i = b_i - mean(b)

            Note that in this update step:
                b_i is unchanged
                r_i increases by b_i * dt
                mean(r) increases by mean(b) * dt
                dv_r_i increases by dv_b_i * dt

            Thus:
                dp_rr_new = sum((dv_r_i + dv_b_i * dt)^2)
                dp_rr_new = sum(dv_r_i^2) + 2 * sum(dv_r_i * dv_b_i) * dt + sum((dv_b_i)^2) * dt^2
                dp_rr_new = dp_rr_old + 2 * dp_rb_old * dt + dp_bb * dt * dt
        */

        self.active_spills_dp_rr +=
            2.0 * self.active_spills_dp_rb * dt_s + self.active_spills_dp_bb * dt_s * dt_s;
        self.active_spills_dp_rb += self.active_spills_dp_bb * dt_s;

        // Exponentially decay old bandit events.
        let mult = -f64::ln(2.0) / UNSPILL_EVENT_HALF_LIFE_SEC;
        let decay_factor = f64::exp(mult * dt_s);
        self.bandit_weight *= decay_factor;
        self.bandit_r_sum *= decay_factor;
        self.bandit_rr_sum *= decay_factor;
        self.bandit_t_sum *= decay_factor;
        self.bandit_tt_sum *= decay_factor;
        self.bandit_rt_sum *= decay_factor;

        self.bandit_explore_success *= decay_factor;
        self.bandit_explore_weight *= decay_factor;

        self.last_update = now;
    }

    // A spill event to update the multi-armed bandit algorithm.
    fn add_bandit_spill_event(
        &mut self,
        n_bytes: u64,
        spill_time: f64,
        spilled_time: f64,
        unspill_time: f64,
    ) {
        let r = n_bytes as f64 * spilled_time;
        let t = spill_time + unspill_time;

        self.bandit_weight += 1.0;
        self.bandit_r_sum += r;
        self.bandit_rr_sum += r * r;
        self.bandit_t_sum += t;
        self.bandit_tt_sum += t * t;
        self.bandit_rt_sum += r * t;
    }

    fn sample_score<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.step_time(Instant::now());

        // Take into account how often we've tried to inspect this context and
        // didn't find anything to spill.
        if self.bandit_explore_weight >= EXPLORED_WEIGHT_THRESHOLD {
            // Bernoulli Thompson sampling of a probability.
            let alpha = 1.0 + self.bandit_explore_success.max(0.0);
            let beta = 1.0 + (self.bandit_explore_weight - self.bandit_explore_success).max(0.0);
            let p = rand_distr::Beta::new(alpha, beta).unwrap().sample(rng);
            if !rng.random_bool(p) {
                return 0.0;
            }
        }

        let mut weight = self.bandit_weight;
        let mut r_sum = self.bandit_r_sum;
        let mut rr_sum = self.bandit_rr_sum;
        let mut t_sum = self.bandit_t_sum;
        let mut tt_sum = self.bandit_tt_sum;
        let mut rt_sum = self.bandit_rt_sum;

        if self.active_spills > 0 {
            // We have active spills but there are three problems:
            //
            //  1. We don't know how long these spills will go on.
            //  2. We don't know the correlation between spill time and relief.
            //  3. We have no idea how long unspilling will take, as that hasn't happened yet.
            //
            // We will assume that these spills get unspilled right now, and make the assumption
            // that IO time is 1:1 correlated with size, using the total spill time so far as
            // multiplier.

            let w = self.active_spills as f64;
            let b = self.active_spills_bytes as f64;
            let r = self.active_spills_bytes_ns as f64 / NANOSECONDS_IN_SECOND;
            let t = self.active_spills_io_time_ns as f64 / NANOSECONDS_IN_SECOND;
            let io_sec_per_byte = t / b;

            let rr = self.active_spills_dp_rr + r * (r / w);
            let bb = self.active_spills_dp_bb + b * (b / w);
            let rb = self.active_spills_dp_rb + r * (b / w);

            weight += w;
            r_sum += r;
            rr_sum += rr;
            t_sum += t;
            tt_sum += io_sec_per_byte * io_sec_per_byte * bb;
            rt_sum += io_sec_per_byte * rb;
        }

        if weight < EXPLORED_WEIGHT_THRESHOLD {
            return UNEXPLORED_SCORE;
        }

        // Calculate means and covariance matrix.
        let inv_weight = 1.0 / weight;
        let bessel = weight / (weight - 1.0);
        let mean_r = r_sum * inv_weight;
        let mean_t = t_sum * inv_weight;
        let var_r = bessel * (rr_sum * inv_weight - mean_r * mean_r).max(0.0);
        let var_t = bessel * (tt_sum * inv_weight - mean_t * mean_t).max(0.0);
        let cov_rt = bessel * (rt_sum * inv_weight - mean_r * mean_t);
        let std_r = var_r.sqrt();
        let std_t = var_t.sqrt();

        loop {
            let z1: f64 = rand_distr::StandardNormal.sample(rng);
            let z2: f64 = rand_distr::StandardNormal.sample(rng);

            let (r, t);
            if std_r > 0.0 && std_t > 0.0 {
                // Sample from bivariate distribution.
                let rho = (cov_rt / (std_r * std_t)).clamp(-1.0, 1.0);
                r = mean_r + std_r * z1;
                t = mean_t + std_t * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            } else {
                // At least one variance is zero, just independent sampling.
                r = mean_r + std_r * z1;
                t = mean_t + std_t * z2;
            }

            if r >= 0.0 && t >= 0.0 {
                return r / (BASE_IO_TIME + t);
            }
        }
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
            successful_explorations: 0,
            total_explorations: 0,
            last_update: Instant::now(),

            bandit_weight: 0.0,
            bandit_r_sum: 0.0,
            bandit_rr_sum: 0.0,
            bandit_t_sum: 0.0,
            bandit_tt_sum: 0.0,
            bandit_rt_sum: 0.0,

            bandit_explore_weight: 0.0,
            bandit_explore_success: 0.0,

            active_spills: 0,
            active_spills_io_time_ns: 0,
            active_spills_bytes_ns: 0,
            active_spills_bytes: 0,

            active_spills_dp_rr: 0.0,
            active_spills_dp_bb: 0.0,
            active_spills_dp_rb: 0.0,
        }
    }
}
