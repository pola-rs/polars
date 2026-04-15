use std::fmt::Debug;
use std::sync::Arc;

use crate::{DynSpillToken, SpillToken, SpillTokenInner, Spillable, memory_manager};

pub trait SpillContext: Send + Sync + 'static {}

pub trait ParameterFreeSpillContext {
    fn register<T, S>(&self, token: &T)
    where
        T: AsRef<SpillToken<S>>,
        S: Spillable;
}

/// A context that spills the most-recently registered spillable when asked.
pub struct MostRecentSpillContext {}

impl MostRecentSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {});
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
        let _ = inner;
        // TODO @ ooc
    }
}

impl SpillContext for MostRecentSpillContext {}

impl Debug for MostRecentSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MostRecentSpillContext").finish()
    }
}

/// A context that spills the least-recently registered spillable when asked.
pub struct LeastRecentSpillContext {}

impl LeastRecentSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {});
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
        let _ = inner;
        // TODO @ ooc
    }
}

impl SpillContext for LeastRecentSpillContext {}

impl Debug for LeastRecentSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeastRecentSpillContext").finish()
    }
}

/// A context that spills a random registered spillable when asked.
pub struct RandomSpillContext {}

impl RandomSpillContext {
    pub fn new() -> Arc<Self> {
        let slf = Arc::new(Self {});
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
        let _ = inner;
        // TODO @ ooc
    }
}

impl SpillContext for RandomSpillContext {}

impl Debug for RandomSpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSpillContext").finish()
    }
}
