#[allow(unused)]
mod v1;

mod global_alloc;
mod memory_manager;
mod spill_context;
mod spill_frame;

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

pub use global_alloc::{Allocator, estimate_memory_usage};
pub use memory_manager::memory_manager;
use polars_async::ASYNC;
use polars_utils::UnitVec;
use polars_utils::with_drop::WithDrop;
pub use spill_context::{
    LeastRecentSpillContext, MostRecentSpillContext, ParameterFreeSpillContext, RandomSpillContext,
    SpillContext,
};
pub use spill_frame::SpillFrame;

// SpillTokenInner's state
const SPILLED_BIT: u64 = 1; // Set when value = None and spilled is Some.
const DROPPED_BIT: u64 = 2; // Set when the token owner has dropped. Forbids creation of new pins / lock.
const LOCK_BIT: u64 = 4; // When set no new pins may be made, and at most 1 thread may set this bit.
const HAS_WAITERS_BIT: u64 = 8; // Only updated while holding waiters lock.
const RO_PIN_COUNT_UNIT: u64 = 16; // Added to the state for each active pin.
const RO_PIN_MASK: u64 = u64::MAX << 4;

struct SpillTokenInner<T: Spillable> {
    // May be read if holding LOCK_BIT or a pin, may be written while holding
    // LOCK_BIT and no pins exist.
    value: UnsafeCell<Option<T>>,

    // May be read+written while holding LOCK_BIT (irrespective of pins).
    spilled: UnsafeCell<Option<T::Spilled>>,

    // Lock should not be held for long, only used to register/wake waiters.
    waiters: Mutex<UnitVec<Waker>>,

    // Used to register into contexts, and detect when a token has moved to a
    // different context.
    registration_id: AtomicU64,

    // See above.
    state: AtomicU64,
}

unsafe impl<T: Spillable + Send> Send for SpillTokenInner<T> {}
unsafe impl<T: Spillable + Sync> Sync for SpillTokenInner<T> {}

impl<T: Spillable> SpillTokenInner<T> {
    /// Waits until the state & mask is zero, returning the state.
    async fn wait(&self, mask: u64) -> u64 {
        std::future::poll_fn(|ctx| {
            // Check mask while holding waiter lock to avoid missed notifications.
            let mut waiters = self.waiters.lock().unwrap();
            let mut state = self.state.load(Ordering::Acquire);
            if state & mask != 0 {
                if state & HAS_WAITERS_BIT == 0 {
                    state = self.state.fetch_add(HAS_WAITERS_BIT, Ordering::AcqRel);
                    if state & mask == 0 {
                        self.state.fetch_sub(HAS_WAITERS_BIT, Ordering::Relaxed);
                        return Poll::Ready(state);
                    }
                }
                waiters.push(ctx.waker().clone());
                Poll::Pending
            } else {
                Poll::Ready(state)
            }
        })
        .await
    }

    /// Wakes any waiters.
    ///
    /// Should be called with the state after performing any updates that waiters
    /// could be waiting for (e.g. the pin count reaches 0 or the lock gets released).
    #[inline(always)]
    fn wake_waiters(&self, state: u64) {
        if state & HAS_WAITERS_BIT != 0 {
            self.wake_waiters_slow();
        }
    }

    #[inline(never)]
    #[cold]
    fn wake_waiters_slow(&self) {
        // Modify HAS_WAITERS_BIT only while holding the lock. Don't notify
        // while holding the lock to reduce critical section.
        let mut waiters_lock = self.waiters.lock().unwrap();
        let waiters = core::mem::take(&mut *waiters_lock);
        self.state.fetch_sub(HAS_WAITERS_BIT, Ordering::Relaxed);
        drop(waiters_lock);

        for w in waiters {
            w.wake();
        }
    }

    // Try to pin the value, returning None if it is spilled, locked or dropped.
    fn try_pin(&self) -> Option<PinnedRef<'_, T>> {
        self.state
            .try_update(Ordering::Acquire, Ordering::Relaxed, |state| {
                if state & (SPILLED_BIT | LOCK_BIT | DROPPED_BIT) != 0 {
                    return None;
                }
                Some(state + RO_PIN_COUNT_UNIT)
            })
            .ok()
            .map(|_| PinnedRef { inner: self })
    }

    // Pin the value, grabbing the lock if it is spilled.
    //
    // If the value is locked this will wait for the lock.
    async fn pin_or_lock(&self) -> Option<PinnedRef<'_, T>> {
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state & (SPILLED_BIT | LOCK_BIT | DROPPED_BIT) == 0 {
                match self.state.compare_exchange_weak(
                    state,
                    state + RO_PIN_COUNT_UNIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(PinnedRef { inner: self }),
                    Err(s) => state = s,
                }
            } else if state & (LOCK_BIT | RO_PIN_MASK | DROPPED_BIT) == 0 {
                match self.state.compare_exchange_weak(
                    state,
                    state | LOCK_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return None,
                    Err(s) => state = s,
                }
            } else {
                assert!(state & DROPPED_BIT == 0);
                state = self.wait(LOCK_BIT).await;
            }
        }
    }

    // Locks the (possibly spilled) value, waiting until it's neither pinned or locked.
    async fn lock(&self) {
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state & (LOCK_BIT | RO_PIN_MASK | DROPPED_BIT) == 0 {
                match self.state.compare_exchange_weak(
                    state,
                    state | LOCK_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(s) => state = s,
                }
            } else {
                assert!(state & DROPPED_BIT == 0);
                state = self.wait(LOCK_BIT | RO_PIN_MASK).await;
            }
        }
    }

    async fn pin(&self) -> PinnedRef<'_, T> {
        if let Some(r) = self.try_pin() {
            return r;
        }

        std::hint::cold_path();

        if let Some(r) = self.pin_or_lock().await {
            return r;
        }

        // We now hold the lock, meaning the value was spilled.
        unsafe {
            debug_assert!(
                self.state.load(Ordering::Relaxed) & (SPILLED_BIT | LOCK_BIT | RO_PIN_MASK)
                    == (SPILLED_BIT | LOCK_BIT)
            );
            let lock_guard = WithDrop::new(self, |slf| {
                slf.wake_waiters(slf.state.fetch_and(!LOCK_BIT, Ordering::AcqRel));
            });
            // TODO: re-register unspilled frame?
            let spilled = (*self.spilled.get()).as_ref().unwrap();
            let value = T::unspill(spilled).await;
            self.value.get().write(Some(value));

            WithDrop::dismiss(lock_guard);
            self.wake_waiters(
                self.state
                    .fetch_add(RO_PIN_COUNT_UNIT - LOCK_BIT - SPILLED_BIT, Ordering::AcqRel),
            );
            PinnedRef { inner: self }
        }
    }

    fn pin_blocking(&self) -> PinnedRef<'_, T> {
        if let Some(r) = self.try_pin() {
            return r;
        }

        std::hint::cold_path();

        ASYNC.block_in_place_on(self.pin())
    }

    async fn pin_mut(&self) -> PinnedMut<'_, T> {
        unsafe {
            self.lock().await;
            let lock_guard = WithDrop::new(self, |slf| {
                slf.wake_waiters(slf.state.fetch_and(!LOCK_BIT, Ordering::AcqRel));
            });

            let value = &mut *self.value.get();
            if value.is_none() {
                debug_assert!(self.state.load(Ordering::Relaxed) & SPILLED_BIT == SPILLED_BIT);
                // TODO: re-register unspilled frame?
                let spilled = (*self.spilled.get()).take().unwrap();
                *value = Some(T::unspill(&spilled).await);
                self.state.fetch_sub(SPILLED_BIT, Ordering::Relaxed);
            } else {
                // We are about to mutate, making our current spill invalid.
                *self.spilled.get() = None;
            }

            WithDrop::dismiss(lock_guard);
            PinnedMut { inner: self }
        }
    }

    fn pin_mut_blocking(&self) -> PinnedMut<'_, T> {
        ASYNC.block_in_place_on(self.pin_mut())
    }

    /// # Safety
    /// May only be called if you currently hold a pin.
    unsafe fn unpin(&self) {
        let old_s = self.state.fetch_sub(RO_PIN_COUNT_UNIT, Ordering::AcqRel);
        if old_s & RO_PIN_MASK == RO_PIN_COUNT_UNIT {
            self.wake_waiters(old_s);
        }
    }

    /// # Safety
    /// May only be called if you currently hold a mutable pin.
    unsafe fn unpin_mut(&self) {
        self.wake_waiters(self.state.fetch_sub(LOCK_BIT, Ordering::AcqRel));
    }

    unsafe fn mark_as_dropped(&self) {
        unsafe {
            // The drop bit prevents new locks/pins from being acquired,
            // allowing us to clean up here.
            let old_state = self.state.fetch_or(DROPPED_BIT, Ordering::Acquire);
            if old_state & (LOCK_BIT | RO_PIN_MASK) == 0 {
                self.value.get().replace(None);
                self.spilled.get().replace(None);
            }
            self.registration_id.fetch_add(1, Ordering::Relaxed);
        }
    }
}

trait DynSpillToken: Send + Sync + 'static {
    /// Assign a new context ID to this spill token, invalidating previous ids.
    fn new_registration_id(&self) -> u64;

    /// Returns the current context ID of this spill token without modifying it.
    fn current_registration_id(&self) -> u64;

    /// Whether this token can be spilled. Returns false if pinned, dropped or
    /// already spilled.
    #[expect(unused)]
    fn can_spill(&self) -> bool;

    /// Estimates how many bytes this object takes up in memory. Returns None
    /// if the object is spilled or dropped.
    #[expect(unused)]
    fn estimate_byte_size(&self) -> Option<usize>;

    /// Tries to spill this token. Returns true if successful.
    ///
    /// May return false if the token is already spilled, or is currently pinned.
    /// It may also return None in those cases (to avoid the allocation of the future).
    #[expect(unused)]
    fn try_spill(&self) -> Option<Pin<Box<dyn Future<Output = bool> + Send + '_>>>;
}

impl<T: Spillable> DynSpillToken for SpillTokenInner<T> {
    fn new_registration_id(&self) -> u64 {
        self.registration_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn current_registration_id(&self) -> u64 {
        self.registration_id.load(Ordering::Relaxed)
    }

    fn can_spill(&self) -> bool {
        self.state.load(Ordering::Acquire) & (SPILLED_BIT | DROPPED_BIT | LOCK_BIT | RO_PIN_MASK)
            == 0
    }

    fn estimate_byte_size(&self) -> Option<usize> {
        self.try_pin().map(|p| p.estimate_byte_size())
    }

    fn try_spill(&self) -> Option<Pin<Box<dyn Future<Output = bool> + Send + '_>>> {
        // First, we try setting the lock bit. If anyone else has a pin we don't bother.
        let pin_update = self
            .state
            .try_update(Ordering::Relaxed, Ordering::Acquire, |state| {
                if state & (SPILLED_BIT | LOCK_BIT | DROPPED_BIT | RO_PIN_MASK) != 0 {
                    return None;
                }
                Some(state | LOCK_BIT)
            });
        if pin_update.is_err() {
            return None;
        }

        Some(Box::pin(async {
            let needs_spill = unsafe { (*self.spilled.get()).is_none() };
            let is_exclusive = if needs_spill {
                // Relax the lock to a pin while spilling such that new pins can
                // come in. After all, we still have it in memory right now.
                self.wake_waiters(
                    self.state
                        .fetch_add(RO_PIN_COUNT_UNIT - LOCK_BIT, Ordering::AcqRel),
                );
                let pin_guard = PinnedRef { inner: self };
                let spilled = pin_guard.spill().await;
                core::mem::forget(pin_guard);
                // We can simply re-acquire the lock here blindly, as we still hold
                // our pin meaning no one else could've gotten the lock.
                let old_state = self
                    .state
                    .fetch_add(LOCK_BIT.wrapping_sub(RO_PIN_COUNT_UNIT), Ordering::Acquire);
                unsafe {
                    self.spilled.get().write(Some(spilled));
                }
                old_state & RO_PIN_MASK == RO_PIN_COUNT_UNIT
            } else {
                true
            };

            let state = if is_exclusive {
                // We are the only pin and hold the lock meaning no one else can
                // access value or create new pins.
                unsafe { self.value.get().replace(None) };
                self.state
                    .fetch_add(SPILLED_BIT.wrapping_sub(LOCK_BIT), Ordering::AcqRel)
            } else {
                self.state.fetch_sub(LOCK_BIT, Ordering::AcqRel)
            };
            self.wake_waiters(state);

            is_exclusive
        }))
    }
}

/// A token representing a possibly spilled object T.
pub struct SpillToken<T: Spillable> {
    inner: Arc<SpillTokenInner<T>>,
}

impl<T: Spillable> SpillToken<T> {
    /// Creates a new SpillToken containing the given value.
    pub fn new(value: T) -> Self {
        let inner = Arc::new(SpillTokenInner {
            value: UnsafeCell::new(Some(value)),
            spilled: UnsafeCell::new(None),
            registration_id: AtomicU64::new(0),
            state: AtomicU64::new(0),
            waiters: Mutex::default(),
        });
        Self { inner }
    }

    /// Try to get a reference to the underlying value, returning None if it was spilled.
    pub fn try_get(&self) -> Option<PinnedRef<'_, T>> {
        self.inner.try_pin()
    }

    /// Get a reference to the underlying value, unspilling it if it was spilled.
    pub async fn get(&self) -> PinnedRef<'_, T> {
        self.inner.pin().await
    }

    /// Blocking version of get.
    pub fn get_blocking(&self) -> PinnedRef<'_, T> {
        self.inner.pin_blocking()
    }

    /// Get a mutable reference to the underlying value, unspilling it if it was spilled.
    pub async fn get_mut(&mut self) -> PinnedMut<'_, T> {
        self.inner.pin_mut().await
    }

    /// Blocking version of get_mut.
    pub fn get_mut_blocking(&mut self) -> PinnedMut<'_, T> {
        self.inner.pin_mut_blocking()
    }

    /// Consumes this SpillToken, unspilling it if it were spilled.
    pub async fn into_inner(mut self) -> T {
        let pin = self.get_mut().await;
        unsafe { &mut *pin.inner.value.get() }.take().unwrap()
    }

    /// Blocking version of into_inner.
    pub fn into_inner_blocking(mut self) -> T {
        let pin = self.get_mut_blocking();
        unsafe { &mut *pin.inner.value.get() }.take().unwrap()
    }
}

impl<T: Spillable> Drop for SpillToken<T> {
    fn drop(&mut self) {
        unsafe { self.inner.mark_as_dropped() };
    }
}

pub struct PinnedRef<'a, T: Spillable> {
    inner: &'a SpillTokenInner<T>,
}

impl<'a, T: Spillable> Deref for PinnedRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.inner.value.get() }.as_ref().unwrap()
    }
}

impl<'a, T: Spillable> Drop for PinnedRef<'a, T> {
    fn drop(&mut self) {
        unsafe { self.inner.unpin() }
    }
}

pub struct PinnedMut<'a, T: Spillable> {
    inner: &'a SpillTokenInner<T>,
}

impl<'a, T: Spillable> Deref for PinnedMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.inner.value.get() }.as_ref().unwrap()
    }
}

impl<'a, T: Spillable> DerefMut for PinnedMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.inner.value.get() }.as_mut().unwrap()
    }
}

impl<'a, T: Spillable> Drop for PinnedMut<'a, T> {
    fn drop(&mut self) {
        unsafe { self.inner.unpin_mut() }
    }
}

pub trait Spillable: Send + Sync + 'static {
    type Spilled;

    /// Estimates how many bytes this object takes up in memory.
    fn estimate_byte_size(&self) -> usize;

    /// Spills this value, returning a spilled representation.
    fn spill(&self) -> impl Future<Output = Self::Spilled> + Send;

    /// Given a previously spilled representation
    fn unspill(location: &Self::Spilled) -> impl Future<Output = Self> + Send;
}
