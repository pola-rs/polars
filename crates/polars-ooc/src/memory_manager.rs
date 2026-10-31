use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use polars_async::ASYNC;
use polars_async::executor::{self, JoinHandle, TaskPriority};
use polars_config::config;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::total_ord::TotalOrd;
use polars_utils::with_drop::WithDrop;
use tokio::sync::{Mutex as AsyncMutex, Notify, OwnedSemaphorePermit, Semaphore as AsyncSemaphore};

// How much worse than the best achieved (sample) score are we willing to look
// for spillables.
const EXPLORE_BEYOND_BEST_SCORE_THRESHOLD: f64 = 20.0;

// Maximum number of SpillFrame candidates we'll consider per attempt.
const SPILL_FRAME_BATCH_SIZE: u64 = 256;

use crate::WeakSpillContext;
use crate::spill_context::{
    InsertReason, PrefetchScheduleResult, RegisteredSpillToken, UNEXPLORED_SCORE,
};
use crate::spill_token::{DynSpillToken, SpillStatus, TrySpillError};

static MEMORY_MANAGER: LazyLock<MemoryManager> = LazyLock::new(MemoryManager::new);

/// Return a reference to the global [`MemoryManager`].
pub fn memory_manager() -> &'static MemoryManager {
    &MEMORY_MANAGER
}

pub struct MemoryManager {
    contexts: RwLock<Vec<WeakSpillContext>>,
    finding_spill_lock: AsyncMutex<()>,
    finding_prefetch_lock: AsyncMutex<()>,
    spill_semaphore: Arc<AsyncSemaphore>,
    prefetch_semaphore: Arc<AsyncSemaphore>,
    est_spill_in_progress: AtomicU64,
    est_prefetch_in_progress: AtomicU64,
    reserved: AtomicU64,
    /// Notified when a spill finishes or is given up.
    spill_done: Notify,
    // Number of live `OocActivity` guards.
    activity: AtomicU64,
}

impl MemoryManager {
    fn new() -> Self {
        Self {
            contexts: RwLock::new(Vec::new()),
            finding_spill_lock: AsyncMutex::new(()),
            finding_prefetch_lock: AsyncMutex::new(()),
            // Read once here: the semaphores are sized at construction, and
            // MEMORY_MANAGER is a LazyLock, so changing these afterwards has no
            // effect.
            spill_semaphore: Arc::new(AsyncSemaphore::new(config().ooc_max_parallel_spill_tasks())),
            prefetch_semaphore: Arc::new(AsyncSemaphore::new(
                config().ooc_max_parallel_prefetch_tasks(),
            )),
            est_spill_in_progress: AtomicU64::new(0),
            est_prefetch_in_progress: AtomicU64::new(0),
            reserved: AtomicU64::new(0),
            spill_done: Notify::new(),
            activity: AtomicU64::new(0),
        }
    }

    /// The memory in use plus incoming memory.
    fn usage_with_incoming(&self) -> u64 {
        crate::estimate_memory_usage()
            + self.est_prefetch_in_progress.load(Ordering::Relaxed)
            + self.reserved.load(Ordering::Relaxed)
    }

    /// The bytes still to spill to fit in the budget, after the spills in progress.
    fn spill_deficit(&self) -> u64 {
        self.usage_with_incoming()
            .saturating_sub(self.est_spill_in_progress.load(Ordering::Relaxed))
            .saturating_sub(config().ooc_memory_budget_bytes())
    }

    pub(crate) fn should_spill(&self) -> bool {
        self.spill_deficit() > 0
    }

    /// Whether the memory in use is over the budget, also counting values that are being
    /// spilled, as they stay in memory until written.
    fn is_over_budget(&self) -> bool {
        self.usage_with_incoming() > config().ooc_memory_budget_bytes()
    }

    pub(crate) fn should_prefetch(&self) -> bool {
        if self.activity.load(Ordering::Relaxed) == 0 {
            return false;
        }

        let usage = crate::estimate_memory_usage();
        let likely_incoming = self.est_prefetch_in_progress.load(Ordering::Relaxed)
            + self.reserved.load(Ordering::Relaxed);
        (usage + likely_incoming) < config().ooc_memory_prefetch_bytes()
    }

    /// Whether a spill is running, selected or done and still on disk.
    pub(crate) fn has_activity(&self) -> bool {
        self.activity.load(Ordering::SeqCst) != 0
    }

    #[cfg(test)]
    pub(crate) fn activity_count(&self) -> u64 {
        self.activity.load(Ordering::SeqCst)
    }

    pub(crate) fn try_get_prefetch_permit(&self) -> Option<OwnedSemaphorePermit> {
        self.prefetch_semaphore.clone().try_acquire_owned().ok()
    }

    fn clean_contexts(&self) {
        if let Ok(mut ctxs) = self.contexts.try_write() {
            ctxs.retain(|ctx| !ctx.is_dead());
        }
    }

    pub(crate) fn register_ctx(&self, ctx: WeakSpillContext) {
        self.contexts.write().unwrap().push(ctx);
    }

    #[inline(always)]
    pub async fn spill(&self) {
        if self.should_spill() {
            self.do_spill().await
        } else if self.should_prefetch() {
            self.do_prefetch().await
        }
        if self.is_over_budget() {
            self.wait_for_spills().await
        }
    }

    #[inline(always)]
    pub fn spill_blocking(&self) {
        if self.should_spill() {
            self.do_spill_blocking()
        } else if self.should_prefetch() {
            self.do_prefetch_blocking()
        }
        if self.is_over_budget() {
            ASYNC.block_in_place_on(self.wait_for_spills())
        }
    }

    /// Waits while the memory in use is over the budget, spilling more when needed. Returns
    /// when no spill is in progress, as then nothing more will free memory.
    #[inline(never)]
    #[cold]
    async fn wait_for_spills(&self) {
        loop {
            if self.should_spill() {
                self.do_spill().await;
            }
            if !self.wait_for_spill_done(|| self.is_over_budget()).await {
                return;
            }
        }
    }

    /// Waits for a spill in progress to finish if `condition` holds. Returns false, without
    /// waiting, if it doesn't hold or no spill is in progress.
    async fn wait_for_spill_done(&self, condition: impl Fn() -> bool) -> bool {
        // Create the future before checking, so a spill that finishes in between still
        // wakes it.
        let spill_done = self.spill_done.notified();
        if !condition() || self.est_spill_in_progress.load(Ordering::Relaxed) == 0 {
            return false;
        }
        spill_done.await;
        true
    }

    #[inline(never)]
    #[cold]
    fn do_spill_blocking(&self) {
        ASYNC.block_in_place_on(self.do_spill())
    }

    /// Reserves `bytes` of memory for work that is about to allocate it. While the reservation
    /// lives, its bytes count as memory in use for spilling and prefetching.
    ///
    /// Before returning, this spills until the memory in use plus all reservations fits in the
    /// budget, waiting for the spills to finish. It stops early, without waiting for other work,
    /// when nothing more can be spilled and no spill is in progress.
    pub async fn reserve(&self, bytes: u64) -> MemoryReservation {
        let old = self.reserved.fetch_add(bytes, Ordering::Relaxed);
        PEAK_RESERVED.fetch_max(old + bytes);
        let reservation = MemoryReservation { bytes };
        while crate::estimate_memory_usage() + self.reserved.load(Ordering::Relaxed)
            > config().ooc_memory_budget_bytes()
        {
            if let Some((ctx, spillables)) = self.find_spillables().await {
                let mut handles = Vec::with_capacity(spillables.len());
                self.spawn_spills(ctx, spillables, |h| handles.push(h))
                    .await;
                for handle in handles {
                    handle.await;
                }
                continue;
            }
            // Spills started elsewhere can still free memory.
            if !self.wait_for_spill_done(|| true).await {
                break;
            }
        }
        reservation
    }

    pub fn reserve_blocking(&self, bytes: u64) -> MemoryReservation {
        ASYNC.block_in_place_on(self.reserve(bytes))
    }

    #[inline(never)]
    #[cold]
    async fn do_spill(&self) {
        while self.should_spill() {
            let Some((ctx, spillables)) = self.find_spillables().await else {
                return;
            };
            self.spawn_spills(ctx, spillables, drop).await;
        }
    }

    /// Spawns a task to spill each of `spillables`, as permits become available, and passes its
    /// handle to `on_spawn`. A dropped handle does not stop its task.
    async fn spawn_spills(
        &self,
        ctx: WeakSpillContext,
        spillables: Vec<(
            Arc<dyn DynSpillToken>,
            SpillInProgressTracker,
            RegisteredSpillToken,
        )>,
        mut on_spawn: impl FnMut(JoinHandle<()>),
    ) {
        let successful_spill = Arc::new(WithDrop::new(
            (AtomicBool::new(false), ctx.clone()),
            move |(success, weak_ctx)| {
                if !success.load(Ordering::Relaxed) {
                    if let Some(strong) = weak_ctx.upgrade() {
                        strong.stats().finish_spill_exploration_event(false);
                    }
                }
            },
        ));

        let task_metrics = ctx.task_metrics();
        for (spillable, spill_in_progress, rt) in spillables {
            let permit = self.spill_semaphore.clone().acquire_owned().await.unwrap();

            let successful_spill = successful_spill.clone();
            let ctx = ctx.clone();

            on_spawn(executor::spawn(
                TaskPriority::High,
                task_metrics.as_deref(),
                async move {
                    // Spill, or reinsert if a failure.
                    match spillable.clone().try_spill(ctx.clone()) {
                        Ok(spill_success) => {
                            if spill_success.await {
                                if !successful_spill.0.swap(true, Ordering::Relaxed) {
                                    if let Some(strong) = ctx.upgrade() {
                                        strong.stats().finish_spill_exploration_event(true);
                                    }
                                }
                            } else {
                                // A racy pin interrupted us, the value is still in memory.
                                spillable.cancel_spill_attempt_and_reinsert(
                                    rt.registration_id,
                                    ctx.1,
                                    InsertReason::Unpin,
                                );
                            }
                        },
                        Err(TrySpillError::Pinned) => {
                            spillable.cancel_spill_attempt_and_reinsert(
                                rt.registration_id,
                                ctx.1,
                                InsertReason::Unpin,
                            );
                        },
                        Err(TrySpillError::AlreadySpilled) | Err(TrySpillError::Dropped) => {},
                    }

                    drop(permit);
                    drop(spill_in_progress);
                },
            ));
        }
    }

    #[inline(never)]
    #[cold]
    fn do_prefetch_blocking(&self) {
        ASYNC.block_in_place_on(self.do_prefetch())
    }

    #[inline(never)]
    #[cold]
    async fn do_prefetch(&self) {
        let Ok(finding_prefetch_guard) = self.finding_prefetch_lock.try_lock() else {
            // Someone else is scheduling prefetches.
            return;
        };

        // TODO: don't loop over all contexts here, keep track of good ones and inspect those plus a couple random ones.
        let contexts = self.contexts.read().unwrap();
        let mut has_dead_context = false;
        let mut live_contexts = Vec::new();
        let mut rng = rand::rng();
        for ctx in contexts.iter() {
            if ctx.is_dead() {
                has_dead_context = true;
                continue;
            };

            // Thompson sampling.
            let score_sample = ctx.0.stats().sample_prefetch_score(&mut rng);
            assert!(!score_sample.is_nan());
            live_contexts.push((ctx.clone(), score_sample));
        }
        drop(contexts);

        // Find the best contexts and loop over their candidates. For each
        // candidate we prefetch it.
        live_contexts.sort_by(|a, b| a.1.tot_cmp(&b.1).reverse());
        let best_explored_score = live_contexts
            .iter()
            .map(|(_ctx, score)| *score)
            .find(|s| *s < UNEXPLORED_SCORE)
            .unwrap_or_default();

        for (ctx, score) in live_contexts {
            if self.prefetch_semaphore.available_permits() == 0 {
                break;
            }

            // Rejected by the bandit; scores are sorted descending so the rest are too.
            if score <= 0.0 {
                break;
            }

            // Refuse to consider contexts which are significantly worse than
            // the best already-explored one.
            if score * EXPLORE_BEYOND_BEST_SCORE_THRESHOLD < best_explored_score {
                break;
            }

            let Some(strong) = ctx.upgrade() else {
                continue;
            };

            let stats = strong.stats();
            stats.start_prefetch_exploration_event();
            match ctx.0.schedule_prefetch(ctx.1) {
                PrefetchScheduleResult::Okay => stats.finish_prefetch_exploration_event(true),
                PrefetchScheduleResult::NoPermitsLeft => {
                    stats.finish_prefetch_exploration_event(true);
                    break;
                },
                PrefetchScheduleResult::NothingToPrefetch
                | PrefetchScheduleResult::StaleContext => {
                    stats.finish_prefetch_exploration_event(false)
                },
            }
        }

        drop(finding_prefetch_guard);
        if has_dead_context {
            self.clean_contexts();
        }
    }

    #[inline(never)]
    #[cold]
    async fn find_spillables(
        &self,
    ) -> Option<(
        WeakSpillContext,
        Vec<(
            Arc<dyn DynSpillToken>,
            SpillInProgressTracker,
            RegisteredSpillToken,
        )>,
    )> {
        let _activity = OocActivity::new();
        crate::decay::note_spill();

        // TODO: don't block here under a certain memory threshold.
        let finding_spill_guard = self.finding_spill_lock.lock().await;
        let mut deficit = self.spill_deficit();
        if deficit == 0 {
            return None;
        }

        // TODO: don't loop over all contexts here, keep track of good ones and inspect those plus a couple random ones.
        let contexts = self.contexts.read().unwrap();
        let mut has_dead_context = false;
        let mut live_contexts = Vec::new();
        let mut rng = rand::rng();
        for ctx in contexts.iter() {
            if ctx.is_dead() {
                has_dead_context = true;
                continue;
            };

            // Thompson sampling.
            let score_sample = ctx.0.stats().sample_spill_score(&mut rng);
            assert!(!score_sample.is_nan());
            live_contexts.push((ctx.clone(), score_sample));
        }
        drop(contexts);

        // Find the best contexts and loop over their candidates. For each
        // candidate we check if it can be spilled else we reinsert it.
        live_contexts.sort_by(|a, b| a.1.tot_cmp(&b.1).reverse());
        let best_explored_score = live_contexts
            .iter()
            .map(|(_ctx, score)| *score)
            .find(|s| *s < UNEXPLORED_SCORE)
            .unwrap_or_default();

        let mut out = None;
        let min_spill = config().ooc_spill_min_bytes();
        for (ctx, score) in live_contexts {
            // Refuse to consider contexts which are significantly worse than
            // the best already-explored one.
            if score * EXPLORE_BEYOND_BEST_SCORE_THRESHOLD < best_explored_score {
                break;
            }

            let Some(strong) = ctx.upgrade() else {
                continue;
            };
            strong.stats().start_spill_exploration_event();

            let mut num_considered = 0;
            let mut candidates = Vec::new();
            ctx.0.drain_live_while(|rt| {
                let Some(cand) = rt.upgrade() else {
                    return true;
                };
                match cand.clone().spill_status() {
                    SpillStatus::InMemory(sz) if sz >= min_spill => {
                        deficit = deficit.saturating_sub(sz);
                        candidates.push((cand, SpillInProgressTracker::new(sz), rt));
                    },
                    SpillStatus::InMemory(_) => {
                        cand.cancel_spill_attempt_and_reinsert(
                            rt.registration_id,
                            ctx.1,
                            InsertReason::TooSmall(rt.timestamp),
                        );
                    },
                    SpillStatus::Pinned => {
                        cand.cancel_spill_attempt_and_reinsert(
                            rt.registration_id,
                            ctx.1,
                            InsertReason::Unpin,
                        );
                    },
                    // A spilled token is re-inserted by whoever unspills it, a
                    // dropped one does not need re-inserting at all.
                    SpillStatus::Spilled | SpillStatus::Dropped => {},
                }

                num_considered += 1;
                deficit > 0 && num_considered < SPILL_FRAME_BATCH_SIZE
            });

            if candidates.is_empty() {
                strong.stats().finish_spill_exploration_event(false);
            } else {
                out = Some((ctx, candidates));
                break;
            }
        }

        drop(finding_spill_guard);
        if has_dead_context {
            self.clean_contexts();
        }
        out
    }
}

/// Memory reserved with [`MemoryManager::reserve`], released on drop.
#[must_use]
pub struct MemoryReservation {
    bytes: u64,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        memory_manager()
            .reserved
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// Counted while alive, so the memory manager knows when no spill is running
/// and nothing is spilled.
///
/// A guard for the next step of a spill is always created before the guard of
/// the previous step is dropped, so the count never drops to zero in between.
pub(crate) struct OocActivity(());

impl OocActivity {
    pub fn new() -> Self {
        memory_manager().activity.fetch_add(1, Ordering::SeqCst);
        Self(())
    }
}

impl Drop for OocActivity {
    fn drop(&mut self) {
        let old = memory_manager().activity.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(old > 0);
    }
}

/// Used to update the total bytes of estimated spills in progress.
struct SpillInProgressTracker {
    bytes: u64,
    _activity: OocActivity,
}

impl SpillInProgressTracker {
    pub fn new(bytes: u64) -> Self {
        memory_manager()
            .est_spill_in_progress
            .fetch_add(bytes, Ordering::Relaxed);
        Self {
            bytes,
            _activity: OocActivity::new(),
        }
    }
}

impl Drop for SpillInProgressTracker {
    fn drop(&mut self) {
        let mm = memory_manager();
        mm.est_spill_in_progress
            .fetch_sub(self.bytes, Ordering::Relaxed);
        mm.spill_done.notify_waiters();
    }
}

/// Used to update the total bytes of estimated prefetches in progress.
pub(crate) struct PrefetchInProgressTracker {
    bytes: u64,
}

impl PrefetchInProgressTracker {
    pub fn new(bytes: u64) -> Self {
        let old = memory_manager()
            .est_prefetch_in_progress
            .fetch_add(bytes, Ordering::Relaxed);
        PEAK_PREFETCH_IN_PROGRESS.fetch_max(old + bytes);
        Self { bytes }
    }
}

static PEAK_PREFETCH_IN_PROGRESS: RelaxedCell<u64> = RelaxedCell::new_u64(0);
static PEAK_RESERVED: RelaxedCell<u64> = RelaxedCell::new_u64(0);

/// The highest prefetch bytes in flight and reserved bytes seen so far, in this process.
pub(crate) fn peak_prefetch_and_reserved() -> (u64, u64) {
    (PEAK_PREFETCH_IN_PROGRESS.load(), PEAK_RESERVED.load())
}

impl Drop for PrefetchInProgressTracker {
    fn drop(&mut self) {
        memory_manager()
            .est_prefetch_in_progress
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;
    use std::task::{Context, Waker};

    use super::*;

    const OVER_BUDGET_VARS: [(&str, &str); 2] = [
        // Keeps polling spill() from starting the global decay timer.
        ("POLARS_OOC_PRESSURE_DECAY", "0"),
        ("POLARS_OOC_MEMORY_BUDGET_MB", "1"),
    ];
    const OVER_BUDGET_USAGE: i64 = 16 << 20;

    /// Puts the memory manager over budget until dropped.
    struct OverBudget;

    impl OverBudget {
        fn new() -> Self {
            for (var, val) in OVER_BUDGET_VARS {
                unsafe { std::env::set_var(var, val) };
                config().reload_env_var(var);
            }
            crate::global_alloc::add_test_usage(OVER_BUDGET_USAGE);
            OverBudget
        }
    }

    impl Drop for OverBudget {
        fn drop(&mut self) {
            crate::global_alloc::add_test_usage(-OVER_BUDGET_USAGE);
            for (var, _) in OVER_BUDGET_VARS {
                unsafe { std::env::remove_var(var) };
                config().reload_env_var(var);
            }
        }
    }

    #[test]
    fn spill_waits_while_over_budget() {
        let _guard = crate::test_lock();
        let _over_budget = OverBudget::new();
        let mm = memory_manager();
        let mut cx = Context::from_waker(Waker::noop());

        // Nothing frees memory without a spill in progress, so there is nothing to wait for.
        assert!(pin!(mm.spill()).poll(&mut cx).is_ready());

        let spill_in_progress = SpillInProgressTracker::new(1);
        let mut spill = pin!(mm.spill());
        assert!(spill.as_mut().poll(&mut cx).is_pending());
        drop(spill_in_progress);
        assert!(spill.as_mut().poll(&mut cx).is_ready());
    }
}
