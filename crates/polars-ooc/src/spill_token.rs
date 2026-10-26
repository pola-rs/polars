use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
use std::time::Instant;

use polars_async::ASYNC;
use polars_utils::UnitVec;
use polars_utils::with_drop::WithDrop;

use crate::memory_manager::{OocActivity, PrefetchInProgressTracker, memory_manager};
use crate::spill_context::InsertReason;
use crate::{SpillContextParam, Spillable, WeakSpillContext};

// SpillTokenInner's state
const SPILLED_BIT: u64 = 1; // Set when value = None and spilled is Some.
const DROPPED_BIT: u64 = 2; // Set when the token owner has dropped. Forbids creation of new pins / lock.
const LOCK_BIT: u64 = 4; // When set no new pins may be made, and at most 1 thread may set this bit.
const HAS_WAITERS_BIT: u64 = 8; // Only updated while holding waiters lock, allows checking for waiters without locking.
const REINSERT_WHEN_SPILLABLE_BIT: u64 = 16; // When set the SpillToken should insert into its registered context once spillable.
const RO_PIN_COUNT_UNIT: u64 = 32; // Added to the state for each active pin.
const RO_PIN_MASK: u64 = u64::MAX << 5;

enum ValueSlot<T> {
    InMemory(T),
    Spilled {
        spill_ctx: WeakSpillContext,
        spill_time_ns: u64,
        spilled_start: Instant,
        _activity: OocActivity,
    },
    Dropped,
}

enum PinOrLockResult<'a, T: Spillable> {
    Pinned(PinnedRef<'a, T>),
    Locked,
    Dropped,
}

#[derive(Default)]
struct LockState {
    // Waiter for register/wake.
    waiters: UnitVec<Waker>,

    // The current context this spill token is registered at.
    cur_ctx: Option<(WeakSpillContext, SpillContextParam)>,

    // The context that spilled the value, while the value is prefetched and not used yet.
    prefetched_by: Option<WeakSpillContext>,
}

struct SpillTokenInner<T: Spillable> {
    // May be read if holding LOCK_BIT or a pin, may be written while holding
    // LOCK_BIT and no pins exist.
    value_slot: UnsafeCell<ValueSlot<T>>,

    // May be read+written while holding LOCK_BIT, or in try_spill after ensuring it has the only
    // pin going from 0 -> 1 pins.
    spilled_value: UnsafeCell<Option<T::Spilled>>,

    // Contains a cached estimated byte size, or MAX if unknown. Should always
    // be and remain known during a spill.
    est_size: AtomicUsize,

    // Lock should not be held for long, only used to register/wake waiters or
    // store current registered context.
    lock: Mutex<LockState>,

    // Used to register into contexts, and detect when a token has moved to a
    // different context.
    registration_id: AtomicU32,

    // See above.
    state: AtomicU64,

    // Set while `LockState::prefetched_by` is set, to check it without locking.
    prefetched_unused: AtomicBool,
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

    fn cached_est_size(&self) -> Option<usize> {
        let sz = self.est_size.load(Ordering::Relaxed);
        if sz < usize::MAX { Some(sz) } else { None }
    }

    fn calc_est_size(&self, pin: &T) -> usize {
        self.cached_est_size().unwrap_or_else(|| {
            let sz = pin.estimate_byte_size();
            self.est_size.store(sz, Ordering::Relaxed);
            sz
        })
    }

    // Try to pin the value, returning Err if it is spilled, locked or dropped.
    fn try_pin(slf: &Arc<Self>) -> Result<PinnedRef<'_, T>, u64> {
        slf.state
            .try_update(Ordering::Acquire, Ordering::Relaxed, |state| {
                if state & (SPILLED_BIT | LOCK_BIT | DROPPED_BIT) != 0 {
                    return None;
                }
                Some(state + RO_PIN_COUNT_UNIT)
            })
            .map(|_| PinnedRef { inner: slf })
    }

    // Pin the value, grabbing the lock if it is spilled.
    //
    // If the value is locked this will wait for the lock.
    async fn pin_or_lock(slf: &Arc<Self>) -> PinOrLockResult<'_, T> {
        let mut state = slf.state.load(Ordering::Relaxed);
        loop {
            if state & (SPILLED_BIT | LOCK_BIT | DROPPED_BIT) == 0 {
                match slf.state.compare_exchange_weak(
                    state,
                    state + RO_PIN_COUNT_UNIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return PinOrLockResult::Pinned(PinnedRef { inner: slf }),
                    Err(s) => state = s,
                }
            } else if state & (LOCK_BIT | RO_PIN_MASK | DROPPED_BIT) == 0 {
                match slf.state.compare_exchange_weak(
                    state,
                    state | LOCK_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return PinOrLockResult::Locked,
                    Err(s) => state = s,
                }
            } else if state & DROPPED_BIT != 0 {
                return PinOrLockResult::Dropped;
            } else {
                state = slf.wait(LOCK_BIT).await;
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
        if let Ok(r) = Self::try_pin(slf) {
            return r;
        }

        Self::pin_impl(slf, false).await.unwrap()
    }

    // Returns None iff dropped.
    #[cold]
    async fn pin_impl(slf: &Arc<Self>, prefetch: bool) -> Option<PinnedRef<'_, T>> {
        std::hint::cold_path();

        match Self::pin_or_lock(slf).await {
            PinOrLockResult::Pinned(p) => return Some(p),
            PinOrLockResult::Dropped => return None,
            PinOrLockResult::Locked => {},
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

            Self::unspill_while_locked(slf, prefetch).await;

            WithDrop::dismiss(lock_guard);
            slf.wake_waiters(
                slf.state
                    .fetch_add(RO_PIN_COUNT_UNIT - LOCK_BIT, Ordering::AcqRel),
            );
            Some(PinnedRef { inner: slf })
        }
    }

    fn pin_blocking(slf: &Arc<Self>) -> PinnedRef<'_, T> {
        if let Ok(r) = Self::try_pin(slf) {
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

            if slf.state.load(Ordering::Relaxed) & SPILLED_BIT != 0 {
                Self::unspill_while_locked(slf, false).await;
            }

            // Clear the spilled value and estimated byte size as we're about to invalidate it through mutable access.
            *slf.spilled_value.get() = None;
            slf.est_size.store(usize::MAX, Ordering::Relaxed);

            WithDrop::dismiss(lock_guard);
            PinnedMut { inner: slf }
        }
    }

    fn pin_mut_blocking(slf: &Arc<Self>) -> PinnedMut<'_, T> {
        ASYNC.block_in_place_on(Self::pin_mut(slf))
    }

    /// # Safety
    /// May only be called if you currently hold the exclusive lock.
    async unsafe fn unspill_while_locked(slf: &Arc<Self>, prefetch: bool) {
        debug_assert!(slf.state.load(Ordering::Relaxed) & SPILLED_BIT == SPILLED_BIT);

        // First, before anything else, we invalidate previous entries in our
        // bookkeeping, and ensure we reinsert when this pin ends. We hold the
        // lock to not race with calls to register().
        let cur_ctx = {
            let lock = slf.lock.lock().unwrap();
            slf.registration_id.fetch_add(1, Ordering::Release);
            slf.state
                .fetch_or(REINSERT_WHEN_SPILLABLE_BIT, Ordering::Release);
            lock.cur_ctx.clone()
        };

        // Now that we have invalidated ourselves from the bookkeeping we can
        // fire off a prefetch request to our context if there is memory headroom.
        if let Some((ctx, _param)) = cur_ctx
            && memory_manager().should_prefetch()
        {
            ctx.0.schedule_prefetch(ctx.1);
        }

        // Do the unspill.
        let n_bytes = slf.cached_est_size().unwrap();
        let unspill_start = Instant::now();
        let spilled = unsafe { (*slf.spilled_value.get()).as_ref().unwrap() };
        let file_bytes = T::spilled_bytes(spilled);
        let value = T::unspill(spilled).await;
        let old_slot = unsafe { slf.value_slot.get().replace(ValueSlot::InMemory(value)) };
        slf.state.fetch_sub(SPILLED_BIT, Ordering::Release);

        let ValueSlot::Spilled {
            spill_ctx,
            spill_time_ns,
            spilled_start,
            ..
        } = old_slot
        else {
            unreachable!()
        };

        // Update stats.
        if let Some(strong) = spill_ctx.upgrade() {
            let stats = strong.stats();
            stats.add_spill_file_read(file_bytes);
            stats.add_unspill(
                n_bytes,
                spill_time_ns,
                spilled_start,
                unspill_start,
                prefetch,
            );
        }
        if prefetch {
            slf.lock.lock().unwrap().prefetched_by = Some(spill_ctx);
            slf.prefetched_unused.store(true, Ordering::Relaxed);
        }
    }

    /// Takes the context that prefetched the value, if it was prefetched and not used yet.
    fn take_prefetched(&self) -> Option<WeakSpillContext> {
        if !self.prefetched_unused.load(Ordering::Relaxed) {
            return None;
        }
        let mut lock = self.lock.lock().unwrap();
        self.prefetched_unused.store(false, Ordering::Relaxed);
        lock.prefetched_by.take()
    }

    /// Counts a consumer access of a prefetched value as a prefetch hit.
    #[inline(always)]
    fn note_use(&self) {
        if self.prefetched_unused.load(Ordering::Relaxed) {
            self.note_use_slow();
        }
    }

    #[cold]
    fn note_use_slow(&self) {
        if let Some(strong) = self.take_prefetched().and_then(|ctx| ctx.upgrade()) {
            strong.stats().add_prefetch_hit();
        }
    }

    fn note_waste(&self) {
        if let Some(strong) = self.take_prefetched().and_then(|ctx| ctx.upgrade()) {
            strong.stats().add_prefetch_waste();
        }
    }

    /// # Safety
    /// May only be called if you currently hold a pin.
    unsafe fn unpin(slf: &Arc<Self>) {
        let old_s = slf.state.fetch_sub(RO_PIN_COUNT_UNIT, Ordering::AcqRel);
        if old_s & RO_PIN_MASK == RO_PIN_COUNT_UNIT {
            if old_s & REINSERT_WHEN_SPILLABLE_BIT != 0 {
                Self::notify_spillable_after_unpin(slf);
            }

            slf.wake_waiters(old_s);
        }
    }

    /// # Safety
    /// May only be called if you currently hold a mutable pin.
    unsafe fn unpin_mut(slf: &Arc<Self>) {
        let old_s = slf.state.fetch_sub(LOCK_BIT, Ordering::AcqRel);
        if old_s & REINSERT_WHEN_SPILLABLE_BIT != 0 {
            Self::notify_spillable_after_unpin(slf);
        }
        slf.wake_waiters(old_s);
    }

    #[cold]
    fn notify_spillable_after_unpin(slf: &Arc<Self>) {
        // Hold registration lock to ensure old_s, cur_ctx, and reg_id are consistent.
        let lock = slf.lock.lock().unwrap();
        let old_s = slf
            .state
            .fetch_and(!REINSERT_WHEN_SPILLABLE_BIT, Ordering::AcqRel);
        let cur_ctx = lock.cur_ctx.clone();
        let reg_id = slf.registration_id.load(Ordering::Relaxed);
        drop(lock);

        if let Some((spill_ctx, _param)) = cur_ctx {
            if old_s & REINSERT_WHEN_SPILLABLE_BIT != 0 {
                let dyn_slf: Arc<dyn DynSpillToken> = slf.clone();
                spill_ctx
                    .0
                    .insert(&dyn_slf, reg_id, spill_ctx.1, InsertReason::Unpin);
            }
        }
    }

    /// Unregisters while holding the registration lock.
    fn unregister_locked(
        &self,
        lock: &mut LockState,
    ) -> Option<(WeakSpillContext, SpillContextParam)> {
        // Clear the re-insert bit, this spillframe is now unregistered.
        self.state
            .fetch_and(!REINSERT_WHEN_SPILLABLE_BIT, Ordering::Relaxed);

        // Increment the registration ID to invalidate previous registration.
        self.registration_id.fetch_add(1, Ordering::Release);
        lock.cur_ctx.take()
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

impl<T, S> SpillTokenInner<T>
where
    T: Clone + Spillable<Spilled = S>,
    S: Clone,
{
    fn clone_impl(slf: &Arc<Self>) -> Self {
        match ASYNC.block_in_place_on(Self::pin_or_lock(slf)) {
            PinOrLockResult::Pinned(r) => {
                slf.note_use();
                return SpillTokenInner {
                    value_slot: UnsafeCell::new(ValueSlot::InMemory(r.clone())),
                    spilled_value: UnsafeCell::new(None),
                    est_size: AtomicUsize::new(slf.cached_est_size().unwrap_or(usize::MAX)),
                    registration_id: AtomicU32::new(0),
                    state: AtomicU64::new(0),
                    lock: Mutex::default(),
                    prefetched_unused: AtomicBool::new(false),
                };
            },
            PinOrLockResult::Dropped => unreachable!(),
            PinOrLockResult::Locked => {},
        }

        // We now hold the lock, meaning the value was spilled.
        unsafe {
            let lock_guard = WithDrop::new(slf, |slf| {
                slf.wake_waiters(slf.state.fetch_and(!LOCK_BIT, Ordering::AcqRel));
            });

            let ValueSlot::Spilled { spill_ctx, .. } = &*lock_guard.value_slot.get() else {
                unreachable!()
            };

            // Simulate a spill.
            let n_bytes = slf.cached_est_size().unwrap();
            let clone_spill_start = Instant::now();
            let spilled_value = (&*lock_guard.spilled_value.get()).as_ref().unwrap().clone();
            let (spill_time_ns, spilled_start) = if let Some(strong) = spill_ctx.upgrade() {
                strong
                    .stats()
                    .add_successful_spill(n_bytes, clone_spill_start)
            } else {
                // Dummy, context is already dead.
                (0, Instant::now())
            };
            SpillTokenInner {
                value_slot: UnsafeCell::new(ValueSlot::Spilled {
                    spill_ctx: spill_ctx.clone(),
                    spill_time_ns,
                    spilled_start,
                    _activity: OocActivity::new(),
                }),
                spilled_value: UnsafeCell::new(Some(spilled_value)),
                est_size: AtomicUsize::new(n_bytes),
                registration_id: AtomicU32::new(0),
                state: AtomicU64::new(SPILLED_BIT),
                lock: Mutex::default(),
                prefetched_unused: AtomicBool::new(false),
            }
        }
    }
}

pub enum TrySpillError {
    AlreadySpilled,
    Pinned,
    Dropped,
}

pub enum SpillStatus {
    InMemory(u64),
    Spilled,
    Pinned,
    Dropped,
}

pub(crate) trait DynSpillToken: Send + Sync + 'static {
    /// Register this spill token at a new context, returning the registration ID.
    fn register(self: Arc<Self>, ctx: WeakSpillContext, param: SpillContextParam) -> u32;

    /// Gets the current context this spill token is registered, if any.
    fn current_ctx(&self) -> Option<(WeakSpillContext, SpillContextParam)>;

    /// Unregisters this spill token from its current context, returning where
    /// it was registered, if anywhere.
    fn unregister(&self) -> Option<(WeakSpillContext, SpillContextParam)>;

    /// Unregisters this spill token, but only if it is currently registered at
    /// the context with the given ID. Returns where it was registered, or None
    /// if it was registered elsewhere or nowhere.
    fn unregister_from(&self, context_id: u64) -> Option<(WeakSpillContext, SpillContextParam)>;

    /// Returns the current context registration ID of this spill token without modifying it.
    fn current_registration_id(&self) -> u32;

    /// Whether this token can be spilled, and if so, its estimated in-memory
    /// size in bytes.
    fn spill_status(self: Arc<Self>) -> SpillStatus;

    /// The estimated size in bytes of this value. May fail if the value is currently locked
    /// exclusively.
    #[expect(unused)]
    fn estimated_byte_size(self: Arc<Self>) -> Option<usize>;

    /// Call this exactly once after removing a SpillToken from its context to
    /// re-insert it once it is spillable again.
    fn cancel_spill_attempt_and_reinsert(
        self: Arc<Self>,
        reg_id: u32,
        context_id: u64,
        reason: InsertReason,
    );

    /// Loads this SpillToken from disk, if it is.
    fn prefetch(self: Arc<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>>;

    /// Tries to spill this token. Returns true if successful.
    ///
    /// May return Err if the token is already spilled, or is currently pinned.
    /// If Ok the future may still return false, in which case a racy pin
    /// occurred during spilling.
    fn try_spill(
        self: Arc<Self>,
        stats_ctx: WeakSpillContext,
    ) -> Result<Pin<Box<dyn Future<Output = bool> + Send>>, TrySpillError>;
}

impl<T: Spillable> DynSpillToken for SpillTokenInner<T> {
    fn register(self: Arc<Self>, ctx: WeakSpillContext, param: SpillContextParam) -> u32 {
        let mut lock = self.lock.lock().unwrap();

        // Clear the re-insert bit, we're registering to a new context.
        let state = self
            .state
            .fetch_and(!REINSERT_WHEN_SPILLABLE_BIT, Ordering::Acquire);

        // Increment the registration ID to invalidate previous registration.
        let reg_id = self.registration_id.fetch_add(1, Ordering::Release) + 1;

        // Set the new current context, and if we were spilled register this as a spilled drain event.
        if let Some((old_ctx, _param)) = lock.cur_ctx.replace((ctx.clone(), param)) {
            if state & SPILLED_BIT != 0 {
                if let Some(strong) = old_ctx.upgrade() {
                    strong.stats().add_spilled_drain_event();
                }
            }
        }

        drop(lock);

        let reason = if state & SPILLED_BIT != 0 {
            InsertReason::RegisterSpilled
        } else {
            InsertReason::Register
        };
        let dyn_arc: Arc<dyn DynSpillToken> = self;
        ctx.0.insert(&dyn_arc, reg_id, ctx.1, reason);
        reg_id
    }

    fn current_ctx(&self) -> Option<(WeakSpillContext, SpillContextParam)> {
        let lock = self.lock.lock().unwrap();
        lock.cur_ctx.clone()
    }

    fn unregister(&self) -> Option<(WeakSpillContext, SpillContextParam)> {
        let mut lock = self.lock.lock().unwrap();
        self.unregister_locked(&mut lock)
    }

    fn unregister_from(&self, context_id: u64) -> Option<(WeakSpillContext, SpillContextParam)> {
        let mut lock = self.lock.lock().unwrap();
        if lock.cur_ctx.as_ref()?.0.1 != context_id {
            return None;
        }
        self.unregister_locked(&mut lock)
    }

    fn current_registration_id(&self) -> u32 {
        self.registration_id.load(Ordering::Relaxed)
    }

    fn spill_status(self: Arc<Self>) -> SpillStatus {
        match Self::try_pin(&self) {
            Ok(p) => SpillStatus::InMemory(p.estimate_byte_size() as u64),
            Err(s) => {
                if s & DROPPED_BIT != 0 {
                    SpillStatus::Dropped
                } else if s & SPILLED_BIT != 0 {
                    SpillStatus::Spilled
                } else if s & (LOCK_BIT | RO_PIN_MASK) != 0 {
                    SpillStatus::Pinned
                } else {
                    unreachable!()
                }
            },
        }
    }

    fn estimated_byte_size(self: Arc<Self>) -> Option<usize> {
        // Fast-path without pinning.
        if let Some(sz) = self.cached_est_size() {
            return Some(sz);
        }

        match Self::try_pin(&self) {
            Ok(p) => Some(self.calc_est_size(&p)),
            Err(_) => self.cached_est_size(),
        }
    }

    fn cancel_spill_attempt_and_reinsert(
        self: Arc<Self>,
        reg_id: u32,
        context_id: u64,
        reason: InsertReason,
    ) {
        // Hold registration lock to ensure old_s and cur_ctx are consistent.
        let lock = self.lock.lock().unwrap();

        // Outdated registration?
        let Some((ctx, _param)) = lock.cur_ctx.clone() else {
            return;
        };
        if ctx.1 != context_id || self.registration_id.load(Ordering::Relaxed) != reg_id {
            return;
        }

        // Set reinsert bit if locked/pinned.
        let old_s = self.state.update(Ordering::AcqRel, Ordering::Relaxed, |s| {
            if s & (LOCK_BIT | RO_PIN_MASK) != 0 {
                s | REINSERT_WHEN_SPILLABLE_BIT
            } else {
                s
            }
        });

        drop(lock);

        // Not locked/pinned/spilled/dropped, reinsert now. If spilled registration of that fact was
        // done by the responsible party, if dropped it doesn't matter.
        if old_s & (LOCK_BIT | RO_PIN_MASK | SPILLED_BIT | DROPPED_BIT) == 0 {
            let dyn_slf: Arc<dyn DynSpillToken> = self.clone();
            ctx.0.insert(&dyn_slf, reg_id, context_id, reason);
        }
    }

    fn prefetch(self: Arc<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        let tracker = PrefetchInProgressTracker::new(self.cached_est_size().unwrap_or(0) as u64);
        Box::pin(async move {
            Self::pin_impl(&self, true).await;
            drop(tracker);
        })
    }

    fn try_spill(
        self: Arc<Self>,
        stats_ctx: WeakSpillContext,
    ) -> Result<Pin<Box<dyn Future<Output = bool> + Send>>, TrySpillError> {
        // First we pin to get the size estimate, and calculate the spilled value. We don't bother if anyone else
        // has a pin or it's already spilled/dropped.
        self.state
            .try_update(Ordering::Acquire, Ordering::Relaxed, |state| {
                if state & (LOCK_BIT | RO_PIN_MASK | DROPPED_BIT | SPILLED_BIT) != 0 {
                    return None;
                }
                Some(state + RO_PIN_COUNT_UNIT)
            })
            .map_err(|s| {
                if s & SPILLED_BIT != 0 {
                    TrySpillError::AlreadySpilled
                } else if s & DROPPED_BIT != 0 {
                    TrySpillError::Dropped
                } else {
                    TrySpillError::Pinned
                }
            })?;

        let owned_pin_guard = WithDrop::new(self, |s| unsafe { SpillTokenInner::unpin(&s) });

        if let Some(strong) = stats_ctx.upgrade() {
            strong.stats().add_spill_start()
        }

        Ok(Box::pin(async move {
            let slf = WithDrop::dismiss(owned_pin_guard);
            let pin_guard = PinnedRef { inner: &slf };

            let spill_start = Instant::now();

            // We have exclusive access to spilled_value as we have an exclusive try_spill pin - no other concurrent
            // try_spill can come here, and any other place we access spilled_value is behind LOCK_BIT or DROPPED_BIT.
            unsafe {
                if (*slf.spilled_value.get()).is_none() {
                    let spilled = pin_guard.spill(&stats_ctx.0.stats().name()).await;
                    // No strong reference: dropping the last one here could re-use the context.
                    stats_ctx
                        .0
                        .stats()
                        .add_spill_file_write(T::spilled_bytes(&spilled));
                    slf.spilled_value.get().write(Some(spilled));
                }
            }

            // Calculate size before exclusive lock to ensure `estimate_byte_size` does not spuriously return None.
            let n_bytes = slf.calc_est_size(&pin_guard);

            // Try to upgrade solo pin into exclusive lock.
            let is_exclusive = slf
                .state
                .try_update(Ordering::Acquire, Ordering::Acquire, |s| {
                    if s & (RO_PIN_MASK | LOCK_BIT) == RO_PIN_COUNT_UNIT {
                        Some(s - RO_PIN_COUNT_UNIT + LOCK_BIT)
                    } else {
                        None
                    }
                })
                .is_ok();

            if is_exclusive {
                core::mem::forget(pin_guard);
                slf.note_waste();

                // We hold the lock meaning no one else can access value or create new pins.
                let (spill_time_ns, spilled_start) = if let Some(strong) = stats_ctx.upgrade() {
                    strong.stats().add_successful_spill(n_bytes, spill_start)
                } else {
                    // Dummy, context is already dead.
                    (0, Instant::now())
                };

                unsafe {
                    slf.value_slot.get().replace(ValueSlot::Spilled {
                        spill_ctx: stats_ctx.clone(),
                        spill_time_ns,
                        spilled_start,
                        _activity: OocActivity::new(),
                    })
                };

                // Insert as spilled in the *current* context, not the context for statistics.
                // Marking as spilled and reading the registration must happen under the same lock,
                // so a concurrent register either observes us as spilled or invalidates our insert.
                let lock = slf.lock.lock().unwrap();
                let state = slf
                    .state
                    .fetch_add(SPILLED_BIT.wrapping_sub(LOCK_BIT), Ordering::AcqRel);
                let cur_ctx = lock.cur_ctx.clone();
                let reg_id = slf.registration_id.load(Ordering::Relaxed);

                drop(lock);
                slf.wake_waiters(state);

                if let Some((ctx, _param)) = cur_ctx {
                    let dyn_slf: Arc<dyn DynSpillToken> = slf.clone();
                    ctx.0.insert(&dyn_slf, reg_id, ctx.1, InsertReason::Spill);
                }
            } else {
                if let Some(strong) = stats_ctx.upgrade() {
                    strong.stats().add_failed_spill(spill_start);
                }
            }

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
            est_size: AtomicUsize::new(usize::MAX),
            registration_id: AtomicU32::new(0),
            state: AtomicU64::new(0),
            lock: Mutex::default(),
            prefetched_unused: AtomicBool::new(false),
        });
        Self { inner }
    }

    /// Upcast to DynSpillToken.
    pub(crate) fn upcast(&self) -> Arc<dyn DynSpillToken> {
        let inner: Arc<SpillTokenInner<T>> = self.inner.clone();
        inner
    }

    /// Gets the current context this spill token is registered, if any.
    pub fn current_ctx(&self) -> Option<(WeakSpillContext, SpillContextParam)> {
        self.inner.current_ctx()
    }

    /// Unregisters this spill token from its current context, if any, returning it.
    pub fn unregister(&mut self) -> Option<(WeakSpillContext, SpillContextParam)> {
        self.inner.unregister()
    }

    /// Try to get a reference to the underlying value, returning None if it was spilled.
    pub fn try_get(&self) -> Option<PinnedRef<'_, T>> {
        SpillTokenInner::try_pin(&self.inner).ok()
    }

    /// Get a reference to the underlying value, unspilling it if it was spilled.
    pub async fn get(&self) -> PinnedRef<'_, T> {
        let pin = SpillTokenInner::pin(&self.inner).await;
        self.inner.note_use();
        pin
    }

    /// Blocking version of get.
    pub fn get_blocking(&self) -> PinnedRef<'_, T> {
        let pin = SpillTokenInner::pin_blocking(&self.inner);
        self.inner.note_use();
        pin
    }

    /// Get a mutable reference to the underlying value, unspilling it if it was spilled.
    pub async fn get_mut(&mut self) -> PinnedMut<'_, T> {
        let pin = SpillTokenInner::pin_mut(&self.inner).await;
        self.inner.note_use();
        pin
    }

    /// Blocking version of get_mut.
    pub fn get_mut_blocking(&mut self) -> PinnedMut<'_, T> {
        let pin = SpillTokenInner::pin_mut_blocking(&self.inner);
        self.inner.note_use();
        pin
    }

    /// Consumes this SpillToken, unspilling it if it were spilled.
    pub async fn into_inner(mut self) -> T {
        let pin = self.get_mut().await;
        let slot = unsafe { pin.inner.value_slot.get().replace(ValueSlot::Dropped) };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        pin.inner.state.fetch_or(DROPPED_BIT, Ordering::Release);
        value
    }

    /// Blocking version of into_inner.
    pub fn into_inner_blocking(mut self) -> T {
        let pin = self.get_mut_blocking();
        let slot = unsafe { pin.inner.value_slot.get().replace(ValueSlot::Dropped) };
        let ValueSlot::InMemory(value) = slot else {
            unreachable!()
        };
        pin.inner.state.fetch_or(DROPPED_BIT, Ordering::Release);
        value
    }
}

impl<T, S> Clone for SpillToken<T>
where
    T: Clone + Spillable<Spilled = S>,
    S: Clone,
{
    fn clone(&self) -> Self {
        // Note: we don't register the clone, perhaps we should?
        Self {
            inner: Arc::new(SpillTokenInner::clone_impl(&self.inner)),
        }
    }
}

impl<T: Spillable> Drop for SpillToken<T> {
    fn drop(&mut self) {
        unsafe { self.inner.mark_as_dropped() };
    }
}

impl<T: Spillable> Drop for SpillTokenInner<T> {
    fn drop(&mut self) {
        self.note_waste();
    }
}

pub struct PinnedRef<'a, T: Spillable> {
    inner: &'a Arc<SpillTokenInner<T>>,
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
        unsafe { SpillTokenInner::unpin(self.inner) }
    }
}

pub struct PinnedMut<'a, T: Spillable> {
    inner: &'a Arc<SpillTokenInner<T>>,
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
        unsafe { SpillTokenInner::unpin_mut(self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MostRecentSpillContext;

    #[derive(Clone)]
    struct Value(Vec<u8>);

    impl Spillable for Value {
        type Spilled = Vec<u8>;

        fn estimate_byte_size(&self) -> usize {
            self.0.len()
        }

        async fn spill(&self, _context_id: &str) -> Vec<u8> {
            self.0.clone()
        }

        async fn unspill(location: &Vec<u8>) -> Self {
            Value(location.clone())
        }
    }

    fn activity() -> u64 {
        memory_manager().activity_count()
    }

    fn spill(token: &SpillToken<Value>, ctx: &MostRecentSpillContext) {
        let spill = token.upcast().try_spill(ctx.downgrade()).ok().unwrap();
        assert!(ASYNC.block_in_place_on(spill));
    }

    fn hits_and_waste(ctx: &MostRecentSpillContext) -> (u64, u64) {
        ctx.downgrade()
            .upgrade()
            .unwrap()
            .stats()
            .prefetch_hits_and_waste()
    }

    #[test]
    fn prefetch_hits_and_waste() {
        let _guard = crate::test_lock();
        let ctx = MostRecentSpillContext::new("test".into(), None);
        let prefetch = |token: &SpillToken<Value>| {
            ASYNC.block_in_place_on(token.upcast().prefetch());
        };

        // Used after the prefetch. Internal pins do not count as use.
        let token = SpillToken::new(Value(vec![1; 100]));
        spill(&token, &ctx);
        prefetch(&token);
        assert!(matches!(
            token.upcast().spill_status(),
            SpillStatus::InMemory(100)
        ));
        assert_eq!(hits_and_waste(&ctx), (0, 0));
        assert_eq!(token.get_blocking().0.len(), 100);
        assert_eq!(hits_and_waste(&ctx), (1, 0));

        // Spilled again before use.
        spill(&token, &ctx);
        prefetch(&token);
        spill(&token, &ctx);
        assert_eq!(hits_and_waste(&ctx), (1, 1));

        // Dropped before use.
        prefetch(&token);
        drop(token);
        assert_eq!(hits_and_waste(&ctx), (1, 2));

        // A clone uses the value.
        let token = SpillToken::new(Value(vec![2; 100]));
        spill(&token, &ctx);
        prefetch(&token);
        let clone = token.clone();
        assert_eq!(hits_and_waste(&ctx), (2, 2));
        drop((token, clone));
        assert_eq!(hits_and_waste(&ctx), (2, 2));
    }

    #[test]
    fn activity_follows_spilled_tokens() {
        let _guard = crate::test_lock();
        let ctx = MostRecentSpillContext::new("test".into(), None);
        let start = activity();

        // Spill, then unspill.
        let token = SpillToken::new(Value(vec![1; 100]));
        spill(&token, &ctx);
        assert_eq!(activity(), start + 1);
        assert_eq!(token.get_blocking().0.len(), 100);
        assert_eq!(activity(), start);

        // Spill, then drop.
        spill(&token, &ctx);
        assert_eq!(activity(), start + 1);
        drop(token);
        assert_eq!(activity(), start);

        // A clone of a spilled token is spilled too.
        let token = SpillToken::new(Value(vec![2; 100]));
        spill(&token, &ctx);
        let clone = token.clone();
        assert_eq!(activity(), start + 2);
        drop(clone);
        assert_eq!(activity(), start + 1);
        drop(token);
        assert_eq!(activity(), start);
    }

    // Waits twice in `unspill`, so a test can act while an unspill holds the lock.
    static UNSPILL_GATE: std::sync::Barrier = std::sync::Barrier::new(2);

    #[derive(Clone)]
    struct GatedValue(Vec<u8>);

    impl Spillable for GatedValue {
        type Spilled = Vec<u8>;

        fn estimate_byte_size(&self) -> usize {
            self.0.len()
        }

        async fn spill(&self, _context_id: &str) -> Vec<u8> {
            self.0.clone()
        }

        async fn unspill(location: &Vec<u8>) -> Self {
            UNSPILL_GATE.wait();
            UNSPILL_GATE.wait();
            GatedValue(location.clone())
        }
    }

    #[test]
    fn activity_with_drop_during_spill_and_prefetch() {
        let _guard = crate::test_lock();
        let ctx = MostRecentSpillContext::new("test".into(), None);
        let start = activity();

        // Dropped while being spilled.
        let token = SpillToken::new(Value(vec![1; 100]));
        let spill_fut = token.upcast().try_spill(ctx.downgrade()).ok().unwrap();
        drop(token);
        ASYNC.block_in_place_on(spill_fut);
        assert_eq!(activity(), start);

        // Dropped while a prefetch holds the lock and unspills.
        let token = SpillToken::new(GatedValue(vec![2; 100]));
        let spill_fut = token.upcast().try_spill(ctx.downgrade()).ok().unwrap();
        assert!(ASYNC.block_in_place_on(spill_fut));
        assert_eq!(activity(), start + 1);
        let prefetch = token.upcast().prefetch();
        let prefetch_thread = std::thread::spawn(move || ASYNC.block_in_place_on(prefetch));
        UNSPILL_GATE.wait();
        drop(token);
        assert_eq!(activity(), start + 1);
        UNSPILL_GATE.wait();
        prefetch_thread.join().unwrap();
        assert_eq!(activity(), start);
        // The prefetched value is never used.
        assert_eq!(
            ctx.downgrade()
                .upgrade()
                .unwrap()
                .stats()
                .prefetch_hits_and_waste(),
            (0, 1)
        );
    }
}
