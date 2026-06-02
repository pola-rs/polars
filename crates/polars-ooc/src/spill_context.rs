use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, RwLock, Weak};

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
}

/// A context that spills the most-recently registered spillable when asked.
pub struct MostRecentSpillContext {
    local: ThreadLocal<RwLock<LocalSpillQueue>>,
}

impl MostRecentSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
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
}

impl LeastRecentSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
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
}

impl RandomSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {
            local: ThreadLocal::default(),
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
}

impl SpillContext for RandomSpillContext {}

impl Debug for RandomSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSpillContext").finish()
    }
}
