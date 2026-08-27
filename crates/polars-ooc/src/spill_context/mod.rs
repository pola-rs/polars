use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use polars_async::executor::TaskPriority;
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

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Timestamp(pub u64);

impl Timestamp {
    pub fn now() -> Self {
        Self(tick_counter())
    }
}

pub(crate) struct RegisteredSpillToken {
    token: Weak<dyn DynSpillToken>,
    pub registration_id: u32,
    pub timestamp: Timestamp,
}

impl RegisteredSpillToken {
    fn new(token: &Arc<dyn DynSpillToken>, registration_id: u32, timestamp: Timestamp) -> Self {
        RegisteredSpillToken {
            token: Arc::downgrade(token),
            registration_id,
            timestamp,
        }
    }

    pub fn upgrade(&self) -> Option<Arc<dyn DynSpillToken>> {
        self.token
            .upgrade()
            .filter(|t| t.current_registration_id() == self.registration_id)
    }

    pub fn is_valid(&self) -> bool {
        self.upgrade().is_some()
    }
}

#[derive(Default)]
struct LocalStagingArea {
    accesses: SpillQueue,
    cancellations: SpillQueue,
    too_small: SpillQueue,
    spilled: SpillQueue,
}

#[derive(Default)]
struct SpillQueue {
    tokens: VecDeque<RegisteredSpillToken>,
    retain_amort: usize,
}

impl SpillQueue {
    pub fn push_back(&mut self, token: RegisteredSpillToken) {
        self.gc();
        self.tokens.push_back(token);
    }

    pub fn push_front(&mut self, token: RegisteredSpillToken) {
        self.gc();
        self.tokens.push_front(token);
    }

    pub fn pop_front(&mut self) -> Option<RegisteredSpillToken> {
        loop {
            let front = self.tokens.pop_front()?;
            if front.is_valid() {
                return Some(front);
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<RegisteredSpillToken> {
        loop {
            let back = self.tokens.pop_back()?;
            if back.is_valid() {
                return Some(back);
            }
        }
    }

    pub fn pop_random(&mut self, rng: &mut ThreadRng) -> Option<RegisteredSpillToken> {
        while !self.tokens.is_empty() {
            let idx = rng.random_range(0..self.tokens.len());
            let back = self.tokens.swap_remove_back(idx)?;
            if back.is_valid() {
                return Some(back);
            }
        }
        None
    }

    fn gc(&mut self) {
        self.retain_amort += 2; // Grows twice as fast as push.
        if self.retain_amort >= self.tokens.len() {
            self.retain_amort = 0;
            self.tokens.retain(|t| t.is_valid());
        }
    }

    pub fn drain_into(&mut self, target: &mut Vec<RegisteredSpillToken>) {
        target.extend(self.tokens.drain(..));
        self.retain_amort = 0;
    }

    pub fn unregister_all(&mut self, context_id: u64) {
        while let Some(rt) = self.pop_back() {
            if let Some(t) = rt.upgrade() {
                t.unregister_from(context_id);
            }
        }
    }

    pub fn extend_front(&mut self, tokens: Vec<RegisteredSpillToken>) {
        for token in tokens.into_iter().rev() {
            if token.is_valid() {
                self.push_front(token);
            }
        }
    }

    pub fn extend_back(&mut self, tokens: Vec<RegisteredSpillToken>) {
        for token in tokens {
            if token.is_valid() {
                self.push_back(token);
            }
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

pub(crate) enum InsertReason {
    Register,
    RegisterSpilled,
    Spill,
    Unpin,
    // The timestamps below are the original registration time.
    #[expect(dead_code)]
    SpillCancelled(Timestamp),
    TooSmall(Timestamp),
}

#[derive(Default)]
struct SharedState {
    spill_queue: SpillQueue,
    unspill_queue: SpillQueue,
}

pub(crate) struct SpillContextInner {
    staging: ThreadLocal<Mutex<LocalStagingArea>>,
    staging_empty: AtomicBool,
    shared: Mutex<SharedState>,
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
            shared: Mutex::default(),
            stats: Arc::new(SpillContextStatistics::new(name)),
            policy: AtomicU8::new(policy as u8),
            refcount: AtomicU64::new(0),
            context_id: AtomicU64::new(ctx_id),
        }
    }

    // We need forced locking to make reset not leave orphan spillframes.
    fn drain_staging(&self, shared: &mut SharedState, force_lock: bool) {
        if !force_lock
            && (self.staging_empty.load(Ordering::Relaxed)
                || self.staging_empty.swap(true, Ordering::AcqRel))
        {
            return;
        }

        let mut accesses = Vec::new();
        let mut cancellations = Vec::new();
        let mut too_small = Vec::new();
        let mut spilled = Vec::new();
        for local in self.staging.iter() {
            let mut lock = local.lock().unwrap();
            lock.accesses.drain_into(&mut accesses);
            lock.cancellations.drain_into(&mut cancellations);
            lock.too_small.drain_into(&mut too_small);
            lock.spilled.drain_into(&mut spilled);
        }

        let policy = self.policy();
        match policy {
            SpillContextPolicy::MostRecent | SpillContextPolicy::LeastRecent => {
                accesses.sort_by_key(|t| t.timestamp);
                cancellations.sort_by_key(|t| t.timestamp);
                too_small.sort_by_key(|t| t.timestamp);
                spilled.sort_by_key(|t| t.timestamp);
            },
            SpillContextPolicy::Random => {},
        }

        shared.spill_queue.extend_back(accesses);
        shared.unspill_queue.extend_back(spilled);

        // A cancelled spill attempt is retried with priority. A token rejected
        // for being too small would fail that same check again, so it goes on
        // the opposite end.
        if matches!(policy, SpillContextPolicy::LeastRecent) {
            shared.spill_queue.extend_front(cancellations);
            shared.spill_queue.extend_back(too_small);
        } else {
            shared.spill_queue.extend_back(cancellations);
            shared.spill_queue.extend_front(too_small);
        }
    }

    fn reset(&self, name: PlSmallStr, policy: SpillContextPolicy) {
        let ctx_id = new_context_id();
        let old_ctx_id = self.context_id.swap(ctx_id, Ordering::Relaxed);
        self.policy.store(policy as u8, Ordering::Relaxed);
        self.stats.reset(name);

        let mut shared = self.shared.lock().unwrap();
        self.drain_staging(&mut shared, true);
        shared.spill_queue.unregister_all(old_ctx_id);
        shared.unspill_queue.unregister_all(old_ctx_id);
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

    pub(crate) fn drain_live_while<F: FnMut(RegisteredSpillToken) -> bool>(&self, mut f: F) {
        let mut rng = rand::rng();
        let policy = self.policy();

        let mut shared = self.shared.lock().unwrap();
        self.drain_staging(&mut shared, false);

        while let Some(rt) = match policy {
            SpillContextPolicy::MostRecent => shared.spill_queue.pop_back(),
            SpillContextPolicy::LeastRecent => shared.spill_queue.pop_front(),
            SpillContextPolicy::Random => shared.spill_queue.pop_random(&mut rng),
        } {
            if !f(rt) {
                break;
            }
        }
    }

    pub(crate) fn insert(
        &self,
        token: &Arc<dyn DynSpillToken>,
        reg_id: u32,
        ctx_id: u64,
        reason: InsertReason,
    ) {
        let mut local = self.staging.get_or_default().lock().unwrap();
        if ctx_id != self.context_id.load(Ordering::Relaxed) {
            return;
        }

        let (queue, ts) = match reason {
            InsertReason::Register => (&mut local.accesses, Timestamp::now()),
            InsertReason::Spill | InsertReason::RegisterSpilled => {
                (&mut local.spilled, Timestamp::now())
            },
            InsertReason::Unpin => (&mut local.accesses, Timestamp::now()),
            InsertReason::SpillCancelled(ts) => (&mut local.cancellations, ts),
            InsertReason::TooSmall(ts) => (&mut local.too_small, ts),
        };

        queue.push_back(RegisteredSpillToken::new(token, reg_id, ts));

        if self.staging_empty.load(Ordering::Relaxed) {
            self.staging_empty.swap(false, Ordering::AcqRel);
        }
    }

    pub(crate) fn schedule_prefetch(&self, ctx_id: u64) {
        let mut shared = self.shared.lock().unwrap();
        if ctx_id != self.context_id.load(Ordering::Relaxed) {
            return;
        }

        self.drain_staging(&mut shared, false);

        let mm = memory_manager();
        let mut rng = rand::rng();
        for _ in 0..self.stats.suggested_prefetch_amount() {
            let Some(permit) = mm.try_get_prefetch_permit() else {
                return;
            };

            let pop = match self.policy() {
                SpillContextPolicy::MostRecent => shared.unspill_queue.pop_front(),
                SpillContextPolicy::LeastRecent => shared.unspill_queue.pop_back(),
                SpillContextPolicy::Random => shared.unspill_queue.pop_random(&mut rng),
            };

            let Some(rt) = pop else { break };
            let Some(token) = rt.upgrade() else {
                continue;
            };

            self.stats.add_prefetch_start();
            polars_async::executor::spawn(TaskPriority::Low, async move {
                token.prefetch().await;
                drop(permit);
            });
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
        token
            .as_ref()
            .upcast()
            .register(self.downgrade(), SpillContextParam(()));
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
    pub fn register_no_spill_check<T, S>(&self, token: &T, param: SpillContextParam)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        token.as_ref().upcast().register(self.clone(), param);
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
