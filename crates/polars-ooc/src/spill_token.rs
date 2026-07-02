use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
use std::time::Instant;

use polars_async::ASYNC;
use polars_utils::UnitVec;
use polars_utils::with_drop::WithDrop;

use crate::{SpillContextParam, Spillable, WeakSpillContext};

// SpillTokenInner's state
const SPILLED_BIT: u64 = 1; // Set when value = None and spilled is Some.
const DROPPED_BIT: u64 = 2; // Set when the token owner has dropped. Forbids creation of new pins / lock.
const LOCK_BIT: u64 = 4; // When set no new pins may be made, and at most 1 thread may set this bit.
const HAS_WAITERS_BIT: u64 = 8; // Only updated while holding waiters lock.
const RO_PIN_COUNT_UNIT: u64 = 16; // Added to the state for each active pin.
const RO_PIN_MASK: u64 = u64::MAX << 4;

enum ValueSlot<T> {
    InMemory(T),
    Spilled {
        n_bytes: usize,
        spill_ctx: WeakSpillContext,
        reinsert_reg_id: u32,
        spill_time_ns: u64,
        spilled_start: Instant,
    },
    Dropped,
}

#[derive(Default)]
struct LockState {
    // Waiter for register/wake.
    waiters: UnitVec<Waker>,

    // The current context this spill token is registered at.
    cur_ctx: Option<WeakSpillContext>,
}

struct SpillTokenInner<T: Spillable> {
    // May be read if holding LOCK_BIT or a pin, may be written while holding
    // LOCK_BIT and no pins exist.
    value_slot: UnsafeCell<ValueSlot<T>>,

    // May be read+written while holding LOCK_BIT (irrespective of pins).
    spilled_value: UnsafeCell<Option<T::Spilled>>,

    // Lock should not be held for long, only used to register/wake waiters or
    // store current registered context.
    lock: Mutex<LockState>,

    // Used to register into contexts, and detect when a token has moved to a
    // different context.
    registration_id: AtomicU32,

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
            let mut lock = self.lock.lock().unwrap();
            let mut state = self.state.load(Ordering::Acquire);
            if state & mask != 0 {
                if state & HAS_WAITERS_BIT == 0 {
                    state = self.state.fetch_add(HAS_WAITERS_BIT, Ordering::AcqRel);
                    if state & mask == 0 {
                        self.state.fetch_sub(HAS_WAITERS_BIT, Ordering::Relaxed);
                        return Poll::Ready(state);
                    }
                }
                lock.waiters.push(ctx.waker().clone());
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
        let mut lock = self.lock.lock().unwrap();
        let waiters = core::mem::take(&mut lock.waiters);
        self.state.fetch_sub(HAS_WAITERS_BIT, Ordering::Relaxed);
        drop(lock);

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

    async fn pin(slf: &Arc<Self>) -> PinnedRef<'_, T> {
        if let Some(r) = slf.try_pin() {
            return r;
        }

        std::hint::cold_path();

        if let Some(r) = slf.pin_or_lock().await {
            return r;
        }

        // We now hold the lock, meaning the value was spilled.
        unsafe {
            debug_assert!(
                slf.state.load(Ordering::Relaxed) & (SPILLED_BIT | LOCK_BIT | RO_PIN_MASK)
                    == (SPILLED_BIT | LOCK_BIT)
            );
            let lock_guard = WithDrop::new(slf, |slf| {
                slf.wake_waiters(slf.state.fetch_and(!LOCK_BIT, Ordering::AcqRel));
            });

            let unspill_start = Instant::now();
            let spilled = (*slf.spilled_value.get()).as_ref().unwrap();
            let value = T::unspill(spilled).await;
            let ValueSlot::Spilled {
                n_bytes,
                spill_ctx,
                reinsert_reg_id,
                spill_time_ns,
                spilled_start,
            } = slf.value_slot.get().replace(ValueSlot::InMemory(value))
            else {
                unreachable!()
            };

            if let Some(strong) = spill_ctx.upgrade() {
                strong
                    .stats()
                    .add_unspill(n_bytes, spill_time_ns, spilled_start, unspill_start);
            }
            if reinsert_reg_id == slf.registration_id.load(Ordering::Relaxed) {
                let dyn_slf: Arc<dyn DynSpillToken> = slf.clone();
                spill_ctx.0.reinsert(&dyn_slf, reinsert_reg_id, spill_ctx.1);
            }

            WithDrop::dismiss(lock_guard);
            slf.wake_waiters(
                slf.state
                    .fetch_add(RO_PIN_COUNT_UNIT - LOCK_BIT - SPILLED_BIT, Ordering::AcqRel),
            );
            PinnedRef { inner: slf }
        }
    }

    fn pin_blocking(slf: &Arc<Self>) -> PinnedRef<'_, T> {
        if let Some(r) = slf.try_pin() {
            return r;
        }

        std::hint::cold_path();

        ASYNC.block_in_place_on(Self::pin(slf))
    }

    async fn pin_mut(slf: &Arc<Self>) -> PinnedMut<'_, T> {
        unsafe {
            slf.lock().await;
            let lock_guard = WithDrop::new(slf, |slf| {
                slf.wake_waiters(slf.state.fetch_and(!LOCK_BIT, Ordering::AcqRel));
            });

            let value_slot = &mut *slf.value_slot.get();
            if let ValueSlot::Spilled {
                n_bytes,
                spill_ctx,
                reinsert_reg_id,
                spill_time_ns,
                spilled_start,
            } = value_slot
            {
                debug_assert!(slf.state.load(Ordering::Relaxed) & SPILLED_BIT == SPILLED_BIT);
                let unspill_start = Instant::now();
                let spilled = (*slf.spilled_value.get()).take().unwrap();
                let value = T::unspill(&spilled).await;

                if let Some(strong) = spill_ctx.upgrade() {
                    strong.stats().add_unspill(
                        *n_bytes,
                        *spill_time_ns,
                        *spilled_start,
                        unspill_start,
                    );
                }
                if *reinsert_reg_id == slf.registration_id.load(Ordering::Relaxed) {
                    let dyn_slf: Arc<dyn DynSpillToken> = slf.clone();
                    spill_ctx
                        .0
                        .reinsert(&dyn_slf, *reinsert_reg_id, spill_ctx.1);
                }

                *value_slot = ValueSlot::InMemory(value);
                slf.state.fetch_sub(SPILLED_BIT, Ordering::Relaxed);
            } else {
                // We are about to mutate, making our current spill invalid.
                *slf.spilled_value.get() = None;
            }

            WithDrop::dismiss(lock_guard);
            PinnedMut { inner: slf }
        }
    }

    fn pin_mut_blocking(slf: &Arc<Self>) -> PinnedMut<'_, T> {
        ASYNC.block_in_place_on(Self::pin_mut(slf))
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

    /// # Safety
    /// May only be called once, by the owning SpillToken.
    unsafe fn mark_as_dropped(&self) {
        unsafe {
            // The drop bit prevents new locks/pins from being acquired,
            // allowing us to clean up here.
            let old_state = self.state.fetch_or(DROPPED_BIT, Ordering::Acquire);
            if old_state & (LOCK_BIT | RO_PIN_MASK) == 0 {
                self.value_slot.get().replace(ValueSlot::Dropped);
                self.spilled_value.get().replace(None);
            }
            let mut lock = self.lock.lock().unwrap();
            self.registration_id.fetch_add(1, Ordering::Relaxed);
            lock.cur_ctx = None;
        }
    }
}

pub enum TrySpillError {
    AlreadySpilled,
    Pinned,
}

pub(crate) trait DynSpillToken: Send + Sync + 'static {
    /// Register this spill token at a new context, returning the registration ID.
    fn register(&self, ctx: WeakSpillContext) -> u32;

    /// Unregisters this spill token from its current context, returning where
    /// it was registered, if anywhere.
    fn unregister(&self) -> Option<WeakSpillContext>;

    /// Returns the current context registration ID of this spill token without modifying it.
    fn current_registration_id(&self) -> u32;

    /// Whether this token can be spilled. Returns false if pinned, dropped or
    /// already spilled.
    fn can_spill(&self) -> bool;

    /// Whether this token is spilled or dropped.
    fn is_spilled_or_dropped(&self) -> bool;

    /// Estimates how many bytes this object takes up in memory. Returns None
    /// if the object cannot be pinned.
    fn estimate_byte_size(&self) -> Option<usize>;

    /// Tries to spill this token. Returns true if successful.
    ///
    /// May return Err if the token is already spilled, or is currently pinned.
    /// If Ok the future may still return false, in which case a racy pin
    /// occurred during spilling.
    fn try_spill(
        &self,
        context: WeakSpillContext,
        registration_id: u32,
    ) -> Result<Pin<Box<dyn Future<Output = bool> + Send + '_>>, TrySpillError>;
}

impl<T: Spillable> DynSpillToken for SpillTokenInner<T> {
    fn register(&self, ctx: WeakSpillContext) -> u32 {
        let mut lock = self.lock.lock().unwrap();
        lock.cur_ctx = Some(ctx);
        self.registration_id.fetch_add(1, Ordering::Release) + 1
    }

    fn unregister(&self) -> Option<WeakSpillContext> {
        let mut lock = self.lock.lock().unwrap();
        self.registration_id.fetch_add(1, Ordering::Release);
        lock.cur_ctx.take()
    }

    fn current_registration_id(&self) -> u32 {
        self.registration_id.load(Ordering::Relaxed)
    }

    fn can_spill(&self) -> bool {
        self.state.load(Ordering::Acquire) & (SPILLED_BIT | DROPPED_BIT | LOCK_BIT | RO_PIN_MASK)
            == 0
    }

    fn is_spilled_or_dropped(&self) -> bool {
        self.state.load(Ordering::Acquire) & (SPILLED_BIT | DROPPED_BIT) != 0
    }

    fn estimate_byte_size(&self) -> Option<usize> {
        self.try_pin().map(|p| p.estimate_byte_size())
    }

    fn try_spill(
        &self,
        ctx: WeakSpillContext,
        registration_id: u32,
    ) -> Result<Pin<Box<dyn Future<Output = bool> + Send + '_>>, TrySpillError> {
        // First, we try setting the lock bit. If anyone else has a pin we don't bother.
        let pin_update = self
            .state
            .try_update(Ordering::Relaxed, Ordering::Acquire, |state| {
                if state & (LOCK_BIT | DROPPED_BIT | RO_PIN_MASK) != 0 {
                    return None;
                }
                Some(state | LOCK_BIT)
            });

        let Ok(state) = pin_update else {
            return Err(TrySpillError::Pinned);
        };
        if state & SPILLED_BIT != 0 {
            // Update re-insertion context, we hold the lock bit and there were no pins so this is safe.
            let ValueSlot::Spilled {
                spill_ctx: reinsert_ctx,
                reinsert_reg_id: reinsert_id,
                ..
            } = (unsafe { &mut *self.value_slot.get() })
            else {
                unreachable!()
            };
            *reinsert_ctx = ctx;
            *reinsert_id = registration_id;

            self.wake_waiters(self.state.fetch_sub(LOCK_BIT, Ordering::AcqRel));
            return Err(TrySpillError::AlreadySpilled);
        }

        Ok(Box::pin(async move {
            let spill_start = Instant::now();
            let needs_spill = unsafe { (*self.spilled_value.get()).is_none() };
            let is_exclusive = if needs_spill {
                // Relax the lock to a pin while spilling such that new pins can
                // come in. After all, we still have it in memory right now.
                self.wake_waiters(
                    self.state
                        .fetch_add(RO_PIN_COUNT_UNIT - LOCK_BIT, Ordering::AcqRel),
                );
                let pin_guard = PinnedRef { inner: self };
                let spilled = pin_guard.spill(&ctx.0.stats().name()).await;
                core::mem::forget(pin_guard);
                // We can simply re-acquire the lock here blindly, as we still hold
                // our pin meaning no one else could've gotten the lock.
                let old_state = self
                    .state
                    .fetch_add(LOCK_BIT.wrapping_sub(RO_PIN_COUNT_UNIT), Ordering::Acquire);
                unsafe {
                    self.spilled_value.get().write(Some(spilled));
                }
                old_state & RO_PIN_MASK == RO_PIN_COUNT_UNIT
            } else {
                true
            };

            let state = if is_exclusive {
                // We are the only pin and hold the lock meaning no one else can
                // access value or create new pins.
                let n_bytes = match unsafe { &*self.value_slot.get() } {
                    ValueSlot::InMemory(val) => val.estimate_byte_size(),
                    _ => unreachable!(),
                };
                let (spill_time_ns, spilled_start) = if let Some(strong) = ctx.upgrade() {
                    strong.stats().add_successful_spill(n_bytes, spill_start)
                } else {
                    // Dummy, context is already dead.
                    (0, Instant::now())
                };
                unsafe {
                    self.value_slot.get().replace(ValueSlot::Spilled {
                        n_bytes,
                        spill_ctx: ctx,
                        reinsert_reg_id: registration_id,
                        spill_time_ns,
                        spilled_start,
                    })
                };
                self.state
                    .fetch_add(SPILLED_BIT.wrapping_sub(LOCK_BIT), Ordering::AcqRel)
            } else {
                if let Some(strong) = ctx.upgrade() {
                    strong.stats().add_failed_spill(spill_start);
                }
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
            value_slot: UnsafeCell::new(ValueSlot::InMemory(value)),
            spilled_value: UnsafeCell::new(None),
            registration_id: AtomicU32::new(0),
            state: AtomicU64::new(0),
            lock: Mutex::default(),
        });
        Self { inner }
    }

    /// Upcast to DynSpillToken.
    pub(crate) fn upcast(&self) -> Arc<dyn DynSpillToken> {
        let inner: Arc<SpillTokenInner<T>> = self.inner.clone();
        inner
    }

    /// Unregisters this spill token from its current context, if any, returning it.
    pub fn unregister(&mut self) -> Option<(WeakSpillContext, SpillContextParam)> {
        let ctx = self.inner.unregister()?;
        let param = SpillContextParam(()); // TODO: get this once we support contexts with parameters.
        Some((ctx, param))
    }

    /// Try to get a reference to the underlying value, returning None if it was spilled.
    pub fn try_get(&self) -> Option<PinnedRef<'_, T>> {
        self.inner.try_pin()
    }

    /// Get a reference to the underlying value, unspilling it if it was spilled.
    pub async fn get(&self) -> PinnedRef<'_, T> {
        SpillTokenInner::pin(&self.inner).await
    }

    /// Blocking version of get.
    pub fn get_blocking(&self) -> PinnedRef<'_, T> {
        SpillTokenInner::pin_blocking(&self.inner)
    }

    /// Get a mutable reference to the underlying value, unspilling it if it was spilled.
    pub async fn get_mut(&mut self) -> PinnedMut<'_, T> {
        SpillTokenInner::pin_mut(&self.inner).await
    }

    /// Blocking version of get_mut.
    pub fn get_mut_blocking(&mut self) -> PinnedMut<'_, T> {
        SpillTokenInner::pin_mut_blocking(&self.inner)
    }

    /// Consumes this SpillToken, unspilling it if it were spilled.
    pub async fn into_inner(mut self) -> T {
        let pin = self.get_mut().await;
        let slot = unsafe { pin.inner.value_slot.get().replace(ValueSlot::Dropped) };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        value
    }

    /// Blocking version of into_inner.
    pub fn into_inner_blocking(mut self) -> T {
        let pin = self.get_mut_blocking();
        let slot = unsafe { pin.inner.value_slot.get().replace(ValueSlot::Dropped) };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        value
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
        let slot = unsafe { &*self.inner.value_slot.get() };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        value
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
        let slot = unsafe { &*self.inner.value_slot.get() };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        value
    }
}

impl<'a, T: Spillable> DerefMut for PinnedMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let slot = unsafe { &mut *self.inner.value_slot.get() };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        value
    }
}

impl<'a, T: Spillable> Drop for PinnedMut<'a, T> {
    fn drop(&mut self) {
        unsafe { self.inner.unpin_mut() }
    }
}
