use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock, Weak};

use polars_async::ASYNC;
use polars_config::config;
use polars_utils::total_ord::TotalOrd;
use tokio::sync::Mutex as AsyncMutex;

// How much worse than the best achieved (sample) score are we willing to look
// for spillables.
const EXPLORE_BEYOND_BEST_SCORE_THRESHOLD: f64 = 20.0;

use crate::spill_context::UNEXPLORED_SCORE;
use crate::spill_token::TrySpillError;
use crate::{DynSpillToken, SpillContext};

static MEMORY_MANAGER: LazyLock<MemoryManager> = LazyLock::new(MemoryManager::new);

/// Return a reference to the global [`MemoryManager`].
pub fn memory_manager() -> &'static MemoryManager {
    &MEMORY_MANAGER
}

pub struct MemoryManager {
    contexts: RwLock<Vec<Weak<dyn SpillContext>>>,
    finding_spill_lock: AsyncMutex<()>,
    est_spill_in_progress: AtomicU64,
}

impl MemoryManager {
    fn new() -> Self {
        Self {
            contexts: RwLock::new(Vec::new()),
            finding_spill_lock: AsyncMutex::new(()),
            est_spill_in_progress: AtomicU64::new(0),
        }
    }

    fn should_spill(&self) -> bool {
        let usage = crate::estimate_memory_usage();
        let likely_dealt_with = self.est_spill_in_progress.load(Ordering::Relaxed);
        usage.saturating_sub(likely_dealt_with) > config().ooc_memory_budget_bytes()
    }

    fn clean_contexts(&self) {
        if let Ok(mut ctxs) = self.contexts.try_write() {
            ctxs.retain(|ctx| ctx.strong_count() > 0);
        }
    }

    pub fn register_ctx<C: SpillContext>(&self, ctx: &Arc<C>) {
        let weak = Arc::downgrade(ctx);
        self.contexts.write().unwrap().push(weak);
    }

    #[inline(always)]
    pub async fn spill(&self) {
        if self.should_spill() {
            self.do_spill().await
        }
    }

    #[inline(always)]
    pub fn spill_blocking(&self) {
        if self.should_spill() {
            self.do_spill_blocking()
        }
    }

    #[inline(never)]
    #[cold]
    fn do_spill_blocking(&self) {
        ASYNC.block_in_place_on(self.do_spill())
    }

    #[inline(never)]
    #[cold]
    async fn do_spill(&self) {
        while self.should_spill() {
            let Some((ctx, spillables)) = self.find_spillables().await else {
                return;
            };

            let mut successful_spill = false;
            for (spillable, id, sz) in spillables {
                // Spill, or reinsert if a failure.
                match spillable.try_spill(ctx.stats(), Arc::downgrade(&ctx), id) {
                    Ok(spill_success) => {
                        if spill_success.await {
                            successful_spill = true;
                        } else {
                            ctx.reinsert(&spillable, id);
                        }
                    },
                    Err(TrySpillError::Pinned) => {
                        ctx.reinsert(&spillable, id);
                    },
                    Err(TrySpillError::AlreadySpilled) => {},
                }

                self.est_spill_in_progress
                    .fetch_sub(sz as u64, Ordering::Relaxed);
            }

            ctx.stats().finish_exploration_event(successful_spill);
        }
    }

    #[inline(never)]
    #[cold]
    async fn find_spillables(
        &self,
    ) -> Option<(
        Arc<dyn SpillContext>,
        Vec<(Arc<dyn DynSpillToken>, u64, usize)>,
    )> {
        // TODO: don't block here under a certain memory threshold.
        let finding_spill_guard = self.finding_spill_lock.lock().await;

        // TODO: don't loop over all contexts here, keep track of good ones and inspect those plus a couple random ones.
        let contexts = self.contexts.read().unwrap();
        let mut has_dead_context = false;
        let mut live_contexts = Vec::new();
        let mut rng = rand::rng();
        for weak_ctx in contexts.iter() {
            let Some(ctx) = weak_ctx.upgrade() else {
                has_dead_context = true;
                continue;
            };

            // Thompson sampling.
            let score_sample = ctx.stats().sample_score(&mut rng);
            assert!(!score_sample.is_nan());
            live_contexts.push((ctx, score_sample));
        }
        drop(contexts);

        // Find the best context and loop over its candidates. For each
        // candidate we check if it can be spilled else we reinsert it.
        let min_spill = config().ooc_spill_min_bytes();
        live_contexts.sort_by(|a, b| a.1.tot_cmp(&b.1).reverse());
        let best_explored_score = live_contexts
            .iter()
            .map(|(_ctx, score)| *score)
            .find(|s| *s < UNEXPLORED_SCORE)
            .unwrap_or_default();

        let mut out = None;
        for (ctx, score) in live_contexts {
            // Refuse to consider contexts which are significantly worse than
            // the best already-explored one.
            if score * EXPLORE_BEYOND_BEST_SCORE_THRESHOLD < best_explored_score {
                break;
            }

            ctx.stats().start_exploration_event();

            let mut total_est_spill = 0;
            let mut candidates = Vec::new();
            for (cand, id) in ctx.pop() {
                if cand.can_spill()
                    && let Some(sz) = cand.estimate_byte_size()
                    && sz as u64 >= min_spill
                {
                    total_est_spill += sz as u64;
                    candidates.push((cand, id, sz));
                } else {
                    if !cand.is_spilled_or_dropped() {
                        ctx.reinsert(&cand, id);
                    }
                }
            }

            if candidates.is_empty() {
                ctx.stats().finish_exploration_event(false);
            } else {
                // Increment the spill-in-progress to avoid eager over-spilling.
                self.est_spill_in_progress
                    .fetch_add(total_est_spill, Ordering::Relaxed);
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
