#[allow(unused)]
mod v1;

mod global_alloc;
mod memory_manager;
mod spill_context;
mod spill_frame;

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub use global_alloc::{Allocator, estimate_memory_usage};
pub use memory_manager::memory_manager;
pub use spill_context::{
    LeastRecentSpillContext, MostRecentSpillContext, ParameterFreeSpillContext, RandomSpillContext,
    SpillContext,
};
pub use spill_frame::SpillFrame;

struct SpillTokenInner<T> {
    // One-to-one with the SpillToken (which isn't Clone).
    value: UnsafeCell<Option<T>>,
}

unsafe impl<T: Send> Send for SpillTokenInner<T> {}
unsafe impl<T: Sync> Sync for SpillTokenInner<T> {}

trait DynSpillToken {}

impl<T: Spillable> DynSpillToken for SpillTokenInner<T> {}

/// A token representing a possibly spilled object T.
pub struct SpillToken<T> {
    inner: Arc<SpillTokenInner<T>>,
}

impl<T> SpillToken<T> {
    /// Creates a new SpillToken containing the given value.
    pub fn new(value: T) -> Self {
        let inner = Arc::new(SpillTokenInner {
            value: UnsafeCell::new(Some(value)),
        });
        Self { inner }
    }

    /// Try to get a reference to the underlying value, returning None if it was spilled.
    pub fn try_get(&self) -> Option<PinnedRef<'_, T>> {
        Some(PinnedRef { inner: &self.inner })
    }

    /// Get a reference to the underlying value, unspilling it if it was spilled.
    pub async fn get(&self) -> PinnedRef<'_, T> {
        // TODO @ ooc: need to despill, track pin count.
        PinnedRef { inner: &self.inner }
    }

    /// Blocking version of get.
    pub fn get_blocking(&self) -> PinnedRef<'_, T> {
        // TODO @ ooc: need to despill, track pin count.
        PinnedRef { inner: &self.inner }
    }

    /// Get a mutable reference to the underlying value, unspilling it if it was spilled.
    pub async fn get_mut(&mut self) -> PinnedMut<'_, T> {
        // TODO @ ooc: need to despill, track pin count.
        PinnedMut { inner: &self.inner }
    }

    /// Blocking version of get_mut.
    pub fn get_mut_blocking(&mut self) -> PinnedMut<'_, T> {
        // TODO @ ooc: need to despill, track pin count.
        PinnedMut { inner: &self.inner }
    }

    /// Consumes this SpillToken, unspilling it if it were spilled.
    pub async fn into_inner(self) -> T {
        unsafe { &mut *self.inner.value.get() }.take().unwrap()
    }

    /// Blocking version of into_inner.
    pub fn into_inner_blocking(self) -> T {
        unsafe { &mut *self.inner.value.get() }.take().unwrap()
    }
}

pub struct PinnedRef<'a, T> {
    inner: &'a SpillTokenInner<T>,
}

impl<'a, T> Deref for PinnedRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.inner.value.get() }.as_ref().unwrap()
    }
}

pub struct PinnedMut<'a, T> {
    inner: &'a SpillTokenInner<T>,
}

impl<'a, T> Deref for PinnedMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.inner.value.get() }.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for PinnedMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.inner.value.get() }.as_mut().unwrap()
    }
}

pub trait Spillable {}
