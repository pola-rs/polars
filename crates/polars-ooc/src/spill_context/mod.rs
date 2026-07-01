use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use polars_utils::pl_str::PlSmallStr;
use rand::rngs::ThreadRng;
use rand::RngExt;
use thread_local::ThreadLocal;

use crate::spill_token::DynSpillToken;
use crate::{SpillToken, Spillable, memory_manager};

mod stats;

pub use stats::SpillContextStatistics;
pub(crate) use stats::UNEXPLORED_SCORE;


#[derive(Default)]
struct LocalSpillQueue {
    tokens: VecDeque<(Weak<dyn DynSpillToken>, u32)>,
    retain_amort: usize,
}

impl LocalSpillQueue {
    pub fn push_back(&mut self, token: &Arc<dyn DynSpillToken>, id: u32) {
        self.gc();
        if token.current_registration_id() == id {
            self.tokens.push_back((Arc::downgrade(token), id));
        }
    }

    pub fn push_front(&mut self, token: &Arc<dyn DynSpillToken>, id: u32) {
        self.gc();
        if token.current_registration_id() == id {
            self.tokens.push_front((Arc::downgrade(token), id));
        }
    }

    pub fn pop_front(&mut self) -> Option<(Arc<dyn DynSpillToken>, u32)> {
        loop {
            let (weak, id) = self.tokens.pop_front()?;
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some((token, id));
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<(Arc<dyn DynSpillToken>, u32)> {
        loop {
            let (weak, id) = self.tokens.pop_back()?;
            if let Some(token) = weak.upgrade()
                && token.current_registration_id() == id
            {
                return Some((token, id));
            }
        }
    }

    pub fn pop_random(&mut self, rng: &mut ThreadRng) -> Option<(Arc<dyn DynSpillToken>, u32)> {
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
            _ => unreachable!()
        }
    }
}

pub(crate) struct SpillContextInner {
    local: ThreadLocal<RwLock<LocalSpillQueue>>,
    stats: Arc<SpillContextStatistics>,
    policy: AtomicU8,
    refcount: AtomicU64,
}

impl SpillContextInner {
    fn new(name: PlSmallStr, policy: SpillContextPolicy) -> Self {
        Self {
            local: ThreadLocal::default(),
            stats: Arc::new(SpillContextStatistics::new(name)),
            policy: AtomicU8::new(policy as u8),
            refcount: AtomicU64::new(0),
        }
    }
    
    fn reset(&self, name: PlSmallStr, policy: SpillContextPolicy) {
        self.policy.store(policy as u8, Ordering::Relaxed);
        for local_lock in self.local.iter() {
            let mut local = local_lock.write().unwrap();
            while let Some(token) = local.pop_back() {
                token.0.unregister();
            }
        }
        self.stats.reset(name);
    }
    
    pub(crate) fn is_dead(&self) -> bool {
        self.refcount.load(Ordering::Acquire) == 0
    }
    
    pub fn policy(&self) -> SpillContextPolicy {
        SpillContextPolicy::from_u8(self.policy.load(Ordering::Relaxed))
    }

    pub fn stats(&self) -> &Arc<SpillContextStatistics> {
        &self.stats
    }

    pub fn pop(&self) -> Vec<(Arc<dyn DynSpillToken>, u32)> {
        let mut out = Vec::new();
        let mut rng = rand::rng();
        let policy = self.policy();
        for local_lock in self.local.iter() {
            if let Ok(mut local) = local_lock.try_write() {
                out.extend(match policy {
                    SpillContextPolicy::MostRecent => local.pop_back(),
                    SpillContextPolicy::LeastRecent => local.pop_front(),
                    SpillContextPolicy::Random => local.pop_random(&mut rng),
                });
            }
        }
        out
    }

    pub fn reinsert(&self, token: &Arc<dyn DynSpillToken>, id: u32) {
        let mut local = self.local.get_or_default().write().unwrap();
        match self.policy() {
            SpillContextPolicy::MostRecent => local.push_front(token, id),
            SpillContextPolicy::LeastRecent => local.push_back(token, id),
            SpillContextPolicy::Random => local.push_back(token, id),
        }
    }
}

// We leak (but do re-use) contets such that a weak reference does not require any reference
// counting. It's possible a weak reference to a spill context starts referring to a (logically)
// different context, after the context is re-used, but it is always safe and sound.
static SPILL_CONTEXT_REUSE_ARENA: Mutex<Vec<&'static SpillContextInner>> = Mutex::new(Vec::new());

// A generic strong reference to a context without knowing which kind it is, preventing it from
// resetting and getting re-used.
struct StrongSpillContextInner(&'static SpillContextInner);

impl StrongSpillContextInner {
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
        memory_manager().register_ctx(inner);
        Self(inner)
    }
}

impl Clone for StrongSpillContextInner {
    fn clone(&self) -> Self {
        self.0.refcount.fetch_add(1, Ordering::Relaxed);
        Self(self.0)
    }
}

impl Drop for StrongSpillContextInner {
    fn drop(&mut self) {
        if self.0.refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            SPILL_CONTEXT_REUSE_ARENA.lock().unwrap().push(self.0);
        }
    }
}

/// A generic weak reference to a context without knowing which kind it is.
pub struct WeakSpillContext(pub(crate) &'static SpillContextInner);

/// An opaque parameter passed into a spill context during registering.
pub struct SpillContextParam(pub(crate) ());

impl WeakSpillContext {
    pub fn register<T, S>(&self, token: &T, _param: SpillContextParam)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.0.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.register(&self.0));
    }
}


pub trait ParameterFreeSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
        Self: Sized;
}



/// A context that spills the most-recently registered spillable when asked.
#[derive(Clone)]
#[repr(transparent)]
pub struct MostRecentSpillContext(StrongSpillContextInner);

impl MostRecentSpillContext {
    pub fn new(name: PlSmallStr) -> Self {
        Self(StrongSpillContextInner::new(name, SpillContextPolicy::MostRecent))
    }
}

impl ParameterFreeSpillContext for MostRecentSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.0.0.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.register(self.0.0));
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
pub struct LeastRecentSpillContext(StrongSpillContextInner);

impl LeastRecentSpillContext {
    pub fn new(name: PlSmallStr) -> Self {
        Self(StrongSpillContextInner::new(name, SpillContextPolicy::LeastRecent))
    }
}

impl ParameterFreeSpillContext for LeastRecentSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.0.0.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.register(self.0.0));
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
pub struct RandomSpillContext(StrongSpillContextInner);

impl RandomSpillContext {
    pub fn new(name: PlSmallStr) -> Self {
        Self(StrongSpillContextInner::new(name, SpillContextPolicy::Random))
    }
}

impl ParameterFreeSpillContext for RandomSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable,
    {
        let dyn_arc = token.as_ref().upcast();
        let mut local = self.0.0.local.get_or_default().write().unwrap();
        local.push_back(&dyn_arc, dyn_arc.register(self.0.0));
    }
}

impl Debug for RandomSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSpillContext")
            .field("name", &self.0.0.stats.name())
            .finish()
    }
}
