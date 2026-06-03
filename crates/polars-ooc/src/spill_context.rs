use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Instant;

use polars_utils::relaxed_cell::RelaxedCell;
use thread_local::ThreadLocal;

use crate::{DynSpillToken, SpillToken, SpillTokenInner, Spillable, memory_manager};

#[derive(Default)]
struct LocalSpillQueue {
    tokens: VecDeque<(Weak<dyn DynSpillToken>, u64)>,
    retain_amort: usize,
}

impl LocalSpillQueue {
    pub fn push_back(&mut self, token: &Arc<dyn DynSpillToken>, id: u64) {
        self.gc();
        self.tokens.push_front((Arc::downgrade(token), id));
    }

    #[expect(unused)]
    pub fn push_front(&mut self, token: &Arc<dyn DynSpillToken>, id: u64) {
        self.gc();
        self.tokens.push_front((Arc::downgrade(token), id));
    }

    #[expect(unused)]
    pub fn pop_front(&mut self) -> Option<Arc<dyn DynSpillToken>> {
        loop {
            let (weak, id) = self.tokens.pop_front()?;
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some(token);
            }
        }
    }

    #[expect(unused)]
    pub fn pop_back(&mut self) -> Option<Arc<dyn DynSpillToken>> {
        loop {
            let (weak, id) = self.tokens.pop_back()?;
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some(token);
            }
        }
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

pub trait SpillContext: Send + Sync + 'static {}

pub trait ParameterFreeSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable;

    fn stats(&self) -> &Arc<SpillContextStatistics>;
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

impl ParameterFreeSpillContext for MostRecentSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let token: &SpillToken<S> = token.as_ref();
        let inner: Arc<SpillTokenInner<S>> = token.inner.clone();
        let inner: Arc<dyn DynSpillToken> = inner;
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(&inner, inner.new_registration_id());
    }

    fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }
}

impl SpillContext for MostRecentSpillContext {}

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

impl ParameterFreeSpillContext for LeastRecentSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let token: &SpillToken<S> = token.as_ref();
        let inner: Arc<SpillTokenInner<S>> = token.inner.clone();
        let inner: Arc<dyn DynSpillToken> = inner;
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(&inner, inner.new_registration_id());
    }

    fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }
}

impl SpillContext for LeastRecentSpillContext {}

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

impl ParameterFreeSpillContext for RandomSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let token: &SpillToken<S> = token.as_ref();
        let inner: Arc<SpillTokenInner<S>> = token.inner.clone();
        let inner: Arc<dyn DynSpillToken> = inner;
        let mut local = self.local.get_or_default().write().unwrap();
        local.push_back(&inner, inner.new_registration_id());
    }

    fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }
}

impl SpillContext for RandomSpillContext {}

impl Debug for RandomSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSpillContext").finish()
    }
}

// Used to normalize divisor to avoid absurdly high scores. Set to 0.1ms.
const BASE_IO_TIME: f64 = 1.0 / 10_000.0;

#[derive(Default)]
pub struct SpillContextStatistics {
    score: RelaxedCell<u64>,
    stats: Mutex<Statistics>,
}

impl SpillContextStatistics {
    pub fn score(&self) -> f64 {
        // Try to re-compute, if lock is taken just take cached value.
        if let Ok(mut stats) = self.stats.try_lock() {
            stats.update(Instant::now());
            let io_time = stats.spill_time + stats.unspill_time;
            let score = stats.spilled_byte_seconds / (BASE_IO_TIME + io_time);
            self.score.store(score.to_bits());
            score
        } else {
            f64::from_bits(self.score.load())
        }
    }

    pub fn add_failed_spill(&self, spill_start: Instant) {
        let spill_time_sec = spill_start.elapsed().as_secs_f64();
        let mut stats = self.stats.lock().unwrap();
        stats.spill_time += spill_time_sec;
        stats.failed_spills += 1;
    }

    pub fn add_successful_spill(&self, n_bytes: usize, spill_start: Instant) {
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        let spill_time_sec = (now - spill_start).as_secs_f64();
        stats.update(now); // Update before mutating bytes_currently_spilled.
        stats.spill_time += spill_time_sec;
        stats.bytes_currently_spilled += n_bytes as u64;
        stats.successful_spills += 1;
    }

    pub fn add_unspill(&self, n_bytes: usize, unspill_start: Instant) {
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        let spill_time_sec = (now - unspill_start).as_secs_f64();
        stats.update(now); // Update before mutating bytes_currently_spilled.
        stats.unspill_time += spill_time_sec;
        stats.bytes_currently_spilled -= n_bytes as u64;
    }
}

struct Statistics {
    spilled_byte_seconds: f64,
    bytes_currently_spilled: u64,
    last_update: Instant,
    spill_time: f64,
    unspill_time: f64,
    successful_spills: u64,
    failed_spills: u64,
}

impl Statistics {
    fn update(&mut self, now: Instant) {
        let delta = now - self.last_update;
        self.spilled_byte_seconds += self.bytes_currently_spilled as f64 * delta.as_secs_f64();
        self.last_update = now;
    }
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            spilled_byte_seconds: 0.0,
            bytes_currently_spilled: 0,
            last_update: Instant::now(),
            spill_time: 0.0,
            unspill_time: 0.0,
            successful_spills: 0,
            failed_spills: 0,
        }
    }
}
