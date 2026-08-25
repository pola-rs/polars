use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use polars_utils::pl_str::PlSmallStr;
use polars_utils::tick_counter::tick_counter;
use rand::RngExt;
use rand::rngs::ThreadRng;
use thread_local::ThreadLocal;

use crate::spill_token::DynSpillToken;
use crate::{SpillToken, Spillable, memory_manager};

mod stats;

pub use stats::SpillContextStatistics;
pub(crate) use stats::UNEXPLORED_SCORE;

fn new_context_id() -> u64 {
    static CONTEXT_ID_CTR: AtomicU64 = AtomicU64::new(0);
    CONTEXT_ID_CTR.fetch_add(1, Ordering::Relaxed)
}

pub struct RegisteredSpillToken {
    token: Weak<dyn DynSpillToken>,
    registration_id: u32,
    timestamp: u64,
}

impl RegisteredSpillToken {
    fn new(token: &Arc<dyn DynSpillToken>, registration_id: u32) -> Self {
        RegisteredSpillToken {
            token: Arc::downgrade(token),
            registration_id,
            timestamp: tick_counter(),
        }
    }

    fn upgrade(&self) -> Option<(Arc<dyn DynSpillToken>, u32)> {
        self.token
            .upgrade()
            .filter(|t| t.current_registration_id() == self.registration_id)
            .map(|t| (t, self.registration_id))
    }
}

#[derive(Default)]
struct LocalStagingArea {
    tokens: Vec<RegisteredSpillToken>,
    retain_amort: usize,
}

impl LocalStagingArea {
    pub fn push(&mut self, token: &Arc<dyn DynSpillToken>, id: u32) {
        self.gc();
        self.tokens.push(RegisteredSpillToken::new(token, id));
    }

    pub fn drain_into(&mut self, target: &mut Vec<RegisteredSpillToken>) {
        target.append(&mut self.tokens);
        self.retain_amort = 0;
    }

    fn gc(&mut self) {
        self.retain_amort += 2; // Grows twice as fast as push.
        if self.retain_amort >= self.tokens.len() {
            self.retain_amort = 0;
            self.tokens.retain(|t| t.upgrade().is_some());
        }
    }
}

#[derive(Default)]
struct SpillQueue {
    tokens: VecDeque<RegisteredSpillToken>,
    retain_amort: usize,
}

impl SpillQueue {
    pub fn push_back(&mut self, token: &Arc<dyn DynSpillToken>, id: u32) {
        self.gc();
        self.tokens.push_back(RegisteredSpillToken::new(token, id));
    }

    pub fn pop_front(&mut self) -> Option<(Arc<dyn DynSpillToken>, u32)> {
        loop {
            let front = self.tokens.pop_front()?;
            if let Some(r) = front.upgrade() {
                return Some(r);
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<(Arc<dyn DynSpillToken>, u32)> {
        loop {
            let back = self.tokens.pop_back()?;
            if let Some(r) = back.upgrade() {
                return Some(r);
            }
        }
    }

    pub fn pop_random(&mut self, rng: &mut ThreadRng) -> Option<(Arc<dyn DynSpillToken>, u32)> {
        while !self.tokens.is_empty() {
            let idx = rng.random_range(0..self.tokens.len());
            let back = self.tokens.swap_remove_back(idx)?;
            if let Some(r) = back.upgrade() {
                return Some(r);
            }
        }
        None
    }

    fn gc(&mut self) {
        self.retain_amort += 2; // Grows twice as fast as push.
        if self.retain_amort >= self.tokens.len() {
            self.retain_amort = 0;
            self.tokens.retain(|t| t.upgrade().is_some());
        }
    }
}

#[repr(u8)]
pub enum SpillContextPolicy {
    MostRecent = 0,
    LeastRecent = 1,
    Random = 2,
}

impl SpillContextPolicy {
    fn from_u8(discriminant: u8) -> Self {
        match discriminant {
            0 => Self::MostRecent,
            1 => Self::LeastRecent,
            2 => Self::Random,
            _ => unreachable!(),
        }
    }
}

pub(crate) struct SpillContextInner {
    staging: ThreadLocal<Mutex<LocalStagingArea>>,
    staging_empty: AtomicBool,
    spill_queue: Mutex<SpillQueue>,
    stats: Arc<SpillContextStatistics>,
    policy: AtomicU8,
    refcount: AtomicU64,
    context_id: AtomicU64,
}

impl SpillContextInner {
    fn new(name: PlSmallStr, policy: SpillContextPolicy) -> Self {
        let ctx_id = new_context_id();
        Self {
            staging: ThreadLocal::default(),
            staging_empty: AtomicBool::new(true),
            spill_queue: Mutex::default(),
            stats: Arc::new(SpillContextStatistics::new(name)),
            policy: AtomicU8::new(policy as u8),
            refcount: AtomicU64::new(0),
            context_id: AtomicU64::new(ctx_id),
        }
    }

    // We need forced locking to make reset not leave orphan spillframes.
    fn drain_staging(&self, queue: &mut SpillQueue, force_lock: bool) {
        if !force_lock
            && (self.staging_empty.load(Ordering::Relaxed)
                || self.staging_empty.swap(true, Ordering::AcqRel))
        {
            return;
        }

        let mut staged_tokens = Vec::new();
        for local in self.staging.iter() {
            local.lock().unwrap().drain_into(&mut staged_tokens);
        }

        match self.policy() {
            SpillContextPolicy::MostRecent | SpillContextPolicy::LeastRecent => {
                staged_tokens.sort_by_key(|t| t.timestamp)
            },
            SpillContextPolicy::Random => {},
        }

        for token in staged_tokens {
            if let Some(t) = token.token.upgrade() {
                queue.push_back(&t, token.registration_id);
            }
        }
    }

    fn reset(&self, name: PlSmallStr, policy: SpillContextPolicy) {
        let ctx_id = new_context_id();
        self.context_id.store(ctx_id, Ordering::Relaxed);
        self.policy.store(policy as u8, Ordering::Relaxed);
        self.stats.reset(name);

        let mut queue = self.spill_queue.lock().unwrap();
        self.drain_staging(&mut queue, true);
        while let Some(t) = queue.pop_back() {
            t.0.unregister();
        }
    }

    fn context_id(&self) -> u64 {
        self.context_id.load(Ordering::Relaxed)
    }

    fn policy(&self) -> SpillContextPolicy {
        SpillContextPolicy::from_u8(self.policy.load(Ordering::Relaxed))
    }

    pub(crate) fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }

    pub(crate) fn drain_while<F: FnMut(Arc<dyn DynSpillToken>, u32) -> bool>(&self, mut f: F) {
        let mut rng = rand::rng();
        let policy = self.policy();

        let mut queue = self.spill_queue.lock().unwrap();
        self.drain_staging(&mut queue, false);

        while let Some((cand, reg_id)) = match policy {
            SpillContextPolicy::MostRecent => queue.pop_back(),
            SpillContextPolicy::LeastRecent => queue.pop_front(),
            SpillContextPolicy::Random => queue.pop_random(&mut rng),
        } {
            if !f(cand, reg_id) {
                break;
            }
        }
    }

    pub(crate) fn reinsert(&self, token: &Arc<dyn DynSpillToken>, reg_id: u32, ctx_id: u64) {
        let mut local = self.staging.get_or_default().lock().unwrap();
        if ctx_id != self.context_id.load(Ordering::Relaxed) {
            return;
        }
        local.push(token, reg_id);

        if self.staging_empty.load(Ordering::Relaxed) {
            self.staging_empty.swap(false, Ordering::AcqRel);
        }
    }
}

// We leak (but do re-use) contexts such that a weak reference does not require any reference
// counting.
static SPILL_CONTEXT_REUSE_ARENA: Mutex<Vec<&'static SpillContextInner>> = Mutex::new(Vec::new());

// A generic strong reference to a context without knowing which kind it is, preventing it from
// resetting and getting re-used.
pub(crate) struct StrongSpillContext(&'static SpillContextInner);

impl StrongSpillContext {
    fn new(name: PlSmallStr, policy: SpillContextPolicy) -> Self {
        let mut arena = SPILL_CONTEXT_REUSE_ARENA.lock().unwrap();
        let inner = if let Some(inner) = arena.pop() {
            inner.reset(name, policy);
            inner
        } else {
            Box::leak(Box::new(SpillContextInner::new(name, policy)))
        };

        // Important: mark as live (refcnt >= 1) before registering.
        inner.refcount.store(1, Ordering::Relaxed);
        let slf = Self(inner);
        memory_manager().register_ctx(slf.downgrade());
        slf
    }

    pub fn downgrade(&self) -> WeakSpillContext {
        WeakSpillContext(self.0, self.0.context_id())
    }

    fn register_no_spill_check<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let reg_id = dyn_arc.register(self.downgrade(), SpillContextParam(()));

        {
            let mut local = self.0.staging.get_or_default().lock().unwrap();
            local.push(&dyn_arc, reg_id);
        }

        if self.0.staging_empty.load(Ordering::Relaxed) {
            self.0.staging_empty.swap(false, Ordering::AcqRel);
        }
    }
}

impl StrongSpillContext {
    pub fn stats(&self) -> &Arc<SpillContextStatistics> {
        self.0.stats()
    }
}

impl Clone for StrongSpillContext {
    fn clone(&self) -> Self {
        self.0.refcount.fetch_add(1, Ordering::Relaxed);
        Self(self.0)
    }
}

impl Drop for StrongSpillContext {
    fn drop(&mut self) {
        if self.0.refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            SPILL_CONTEXT_REUSE_ARENA.lock().unwrap().push(self.0);
        }
    }
}

/// A generic weak reference to a context without knowing which kind it is.
#[derive(Clone)]
pub struct WeakSpillContext(pub(crate) &'static SpillContextInner, pub(crate) u64);

impl WeakSpillContext {
    pub(crate) fn upgrade(&self) -> Option<StrongSpillContext> {
        if self.0.context_id() != self.1 {
            return None;
        }

        self.0.refcount.fetch_add(1, Ordering::Relaxed);
        let strong = StrongSpillContext(self.0);

        // To avoid race conditions, we must check again.
        if self.0.context_id() != self.1 {
            return None;
        }

        Some(strong)
    }

    pub(crate) fn is_dead(&self) -> bool {
        self.0.context_id() != self.1
    }
}

/// An opaque parameter passed into a spill context during registering.
#[derive(Clone)]
pub struct SpillContextParam(pub(crate) ());

impl WeakSpillContext {
    pub fn register<T, S>(&self, token: &T, param: SpillContextParam)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.0.staging.get_or_default().lock().unwrap();
        if self.0.context_id() == self.1 {
            local.push(&dyn_arc, dyn_arc.register(self.clone(), param));
            if self.0.staging_empty.load(Ordering::Relaxed) {
                self.0.staging_empty.swap(false, Ordering::AcqRel);
            }
        }
    }
}

pub trait ParameterFreeSpillContext {
    fn register_no_spill_check<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
        Self: Sized;

    fn register<T, S>(&self, token: &T) -> impl Future<Output = ()>
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
        Self: Sized,
    {
        self.register_no_spill_check(token);
        memory_manager().spill()
    }
}

/// A context that spills the most-recently registered spillable when asked.
#[derive(Clone)]
#[repr(transparent)]
pub struct MostRecentSpillContext(StrongSpillContext);

impl MostRecentSpillContext {
    pub fn new(name: PlSmallStr) -> Self {
        Self(StrongSpillContext::new(
            name,
            SpillContextPolicy::MostRecent,
        ))
    }
}

impl ParameterFreeSpillContext for MostRecentSpillContext {
    fn register_no_spill_check<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        self.0.register_no_spill_check(token);
    }
}

impl Debug for MostRecentSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MostRecentSpillContext")
            .field("name", &self.0.0.stats.name())
            .finish()
    }
}

/// A context that spills the least-recently registered spillable when asked.
#[derive(Clone)]
#[repr(transparent)]
pub struct LeastRecentSpillContext(StrongSpillContext);

impl LeastRecentSpillContext {
    pub fn new(name: PlSmallStr) -> Self {
        Self(StrongSpillContext::new(
            name,
            SpillContextPolicy::LeastRecent,
        ))
    }
}

impl ParameterFreeSpillContext for LeastRecentSpillContext {
    fn register_no_spill_check<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        self.0.register_no_spill_check(token);
    }
}

impl Debug for LeastRecentSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeastRecentSpillContext")
            .field("name", &self.0.0.stats.name())
            .finish()
    }
}

/// A context that spills a random registered spillable when asked.
#[derive(Clone)]
pub struct RandomSpillContext(StrongSpillContext);

impl RandomSpillContext {
    pub fn new(name: PlSmallStr) -> Self {
        Self(StrongSpillContext::new(name, SpillContextPolicy::Random))
    }
}

impl ParameterFreeSpillContext for RandomSpillContext {
    fn register_no_spill_check<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        self.0.register_no_spill_check(token);
    }
}

impl Debug for RandomSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSpillContext")
            .field("name", &self.0.0.stats.name())
            .finish()
    }
}
