use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use polars_config::{SpillFormat, SpillPolicy};
use polars_core::prelude::DataFrame;

use super::df_store::{DataFrameStore, LoadStatus, SpillState};
use super::spiller::Spiller;
use super::token::{SlotId, Token};
use super::treiber_stack::TreiberStack;

static MEMORY_MANAGER: LazyLock<MemoryManager> = LazyLock::new(MemoryManager::default);

/// Return a reference to the global [`MemoryManager`].
pub fn mm() -> &'static MemoryManager {
    &MEMORY_MANAGER
}

const UNREGISTERED_THREAD: u64 = u64::MAX;

thread_local! {
    static THREAD_IDX: Cell<u64> = const { Cell::new(UNREGISTERED_THREAD) };
}

/// Per-thread byte tracking.
struct ThreadDrift {
    total_local_bytes: AtomicUsize,
    last_sync_bytes: AtomicUsize,
}

impl ThreadDrift {
    fn new() -> Self {
        Self {
            total_local_bytes: AtomicUsize::new(0),
            last_sync_bytes: AtomicUsize::new(0),
        }
    }
}

/// Describes how an operator accesses its buffered data.
///
/// The eviction algorithm uses this to pick the best spill candidate:
/// - [`NoPattern`](AccessPattern::NoPattern): no ordering constraint.
/// - [`Fifo`](AccessPattern::Fifo): evict the newest (last-in) entry first.
#[derive(Debug, Clone, Copy, Default)]
pub enum AccessPattern {
    #[default]
    NoPattern,
    Fifo,
}

/// Per-operator spill context registered via [`MemoryManager::register_context`].
///
/// Each operator creates one context. When data is stored with
/// [`AccessPattern::Fifo`], the entry is pushed onto the context's
/// stack. During spill collection the memory manager scans all
/// contexts' stacks instead of scanning the full store.
#[derive(Default)]
pub struct SpillContext {
    fifo_entries: TreiberStack<SlotId>,
}

impl std::fmt::Debug for SpillContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillContext").finish()
    }
}

impl SpillContext {
    pub(crate) fn push(&self, id: SlotId) {
        self.fifo_entries.push(id);
    }
}

/// Global memory manager that tracks [`DataFrame`]s behind opaque [`Token`]s.
///
/// Each stored frame lives in a lock-free [`DataFrameStore`]. Memory usage is
/// tracked per-thread with drift-based synchronization to a global counter,
/// avoiding atomic contention on every store/take. When the budget is exceeded
/// the manager can spill frames to disk and reload them transparently.
pub struct MemoryManager {
    pub(crate) spiller: Spiller,
    pub(crate) df_store: DataFrameStore,
    contexts: TreiberStack<Arc<SpillContext>>,
    total_bytes: AtomicUsize,
    thread_trackers: boxcar::Vec<ThreadDrift>,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(polars_config::config().ooc_spill_format())
    }
}

impl MemoryManager {
    /// Create a new [`MemoryManager`] with the given spill format.
    pub fn new(format: SpillFormat) -> Self {
        Self {
            spiller: Spiller::new(format),
            df_store: DataFrameStore::new(),
            contexts: TreiberStack::default(),
            total_bytes: AtomicUsize::new(0),
            thread_trackers: boxcar::Vec::new(),
        }
    }

    fn budget(&self) -> usize {
        let fraction = polars_config::config().ooc_memory_budget_fraction();
        (polars_utils::sys::total_memory() as f64 * fraction) as usize
    }

    /// Return the index of the calling thread, registering it on first call.
    fn thread_idx(&self) -> u64 {
        THREAD_IDX.with(|cell| {
            let idx = cell.get();
            if idx != UNREGISTERED_THREAD {
                return idx;
            }
            self.register_thread(cell)
        })
    }

    /// Allocate a new thread tracker and cache its index.
    #[cold]
    #[inline(never)]
    fn register_thread(&self, cell: &Cell<u64>) -> u64 {
        let new_idx = self.thread_trackers.push(ThreadDrift::new()) as u64;
        cell.set(new_idx);
        new_idx
    }

    /// Add `delta` bytes to the calling thread's local counter and flush
    /// the drift to the global `total_bytes` when the threshold is exceeded.
    /// Returns `true` if the memory budget is exceeded after flushing.
    fn track_bytes(&self, delta: isize) -> bool {
        let td = &self.thread_trackers[self.thread_idx() as usize];
        let new = td
            .total_local_bytes
            .load(Ordering::Relaxed)
            .wrapping_add(delta as usize);
        td.total_local_bytes.store(new, Ordering::Relaxed);

        let drift = new.wrapping_sub(td.last_sync_bytes.load(Ordering::Relaxed)) as isize;
        if drift.abs() < polars_config::config().ooc_drift_threshold() as isize {
            return false;
        }

        td.last_sync_bytes.store(new, Ordering::Relaxed);
        self.sync_drift(drift);
        matches!(
            polars_config::config().ooc_spill_policy(),
            SpillPolicy::Spill
        ) && self.total_bytes.load(Ordering::Relaxed) > self.budget()
    }

    fn sync_drift(&self, drift: isize) {
        if drift >= 0 {
            self.total_bytes
                .fetch_add(drift as usize, Ordering::Relaxed);
        } else {
            self.total_bytes
                .fetch_sub(drift.unsigned_abs(), Ordering::Relaxed);
        }
    }

    /// Prepare the memory manager for a new query.
    ///
    /// - Resets spill escalation level to 0 (first spill pass is conservative).
    /// - Drops all SpillContexts from the previous query.
    ///
    /// Called once at the start of each query.
    pub fn init_for_query(&self) {
        self.spiller.reset_spill_level();
        self.contexts.clear();
        self.total_bytes.store(0, Ordering::Relaxed);
        for (_, td) in self.thread_trackers.iter() {
            td.total_local_bytes.store(0, Ordering::Relaxed);
            td.last_sync_bytes.store(0, Ordering::Relaxed);
        }
    }

    /// Register a per-operator [`SpillContext`] for spill tracking.
    ///
    /// The returned `Arc` should be held by the operator for its lifetime.
    /// The memory manager keeps a clone for spill collection scanning.
    pub fn register_context(&self) -> Arc<SpillContext> {
        let ctx: Arc<SpillContext> = Arc::default();
        self.contexts.push(ctx.clone());
        ctx
    }

    /// Return the row count of the stored [`DataFrame`].
    pub fn height(&self, token: &Token) -> usize {
        self.df_store.height(token.id().index)
    }

    /// Pin the entry so it has lower priority during spill collection.
    /// Unpinned entries are always spilled first; pinned entries are only
    /// spilled if freeing unpinned entries alone is not enough.
    pub fn pin(&self, token: &Token) {
        let id = token.id();
        self.df_store.set_pinned(id.index, id.generation, true);
    }

    /// Unpin the entry so it has higher priority during spill collection.
    pub fn unpin(&self, token: &Token) {
        let id = token.id();
        self.df_store.set_pinned(id.index, id.generation, false);
    }

    /// Remove the entry for this [`Token`], update memory accounting, and
    /// delete the spill file if the frame was spilled. Called by [`Token::drop`].
    pub(crate) fn drop_token(&self, token: &Token) {
        let id = token.id();
        let Some((spill_state, size, seq)) = self.df_store.remove(id.index, id.generation) else {
            return;
        };
        self.track_bytes(-(size as isize));
        if spill_state == SpillState::Spilled {
            self.spiller.delete_spill_file(id.index, id.generation, seq);
        }
    }

    /// Insert a [`DataFrame`] into the store. Returns the [`Token`] and
    /// whether the memory budget was exceeded after insertion.
    fn insert(
        &self,
        df: DataFrame,
        ctx: &SpillContext,
        access_pattern: AccessPattern,
    ) -> (Token, bool) {
        let size_bytes = df.estimated_size();
        let (idx, generation) = self.df_store.insert(df, size_bytes);
        let token = Token::new(idx, generation);

        match access_pattern {
            AccessPattern::Fifo => ctx.push(token.id()),
            // TODO: NoPattern entries are never pushed onto any SpillContext
            // stack, so the spill collector cannot see or evict them.
            AccessPattern::NoPattern => {},
        }

        let should_spill = self.track_bytes(size_bytes as isize);

        (token, should_spill)
    }

    /// Store a [`DataFrame`] and return a [`Token`] that can retrieve it later.
    /// May trigger spilling if the memory budget is exceeded.
    pub async fn store(
        &self,
        df: DataFrame,
        ctx: &SpillContext,
        access_pattern: AccessPattern,
    ) -> Token {
        let (token, should_spill) = self.insert(df, ctx, access_pattern);
        if should_spill {
            self.spill().await;
        }
        token
    }

    /// Blocking variant of [`store`](Self::store).
    pub fn store_blocking(
        &self,
        df: DataFrame,
        ctx: &SpillContext,
        access_pattern: AccessPattern,
    ) -> Token {
        let (token, should_spill) = self.insert(df, ctx, access_pattern);
        if should_spill {
            self.spill_blocking();
        }
        token
    }

    /// Take the [`DataFrame`] out of the manager, consuming the [`Token`].
    pub async fn take_df(&self, token: Token) -> DataFrame {
        let id = token.id();
        loop {
            self.ensure_loaded(&token).await;
            if let Some((df, size)) = self.df_store.take(id.index, id.generation) {
                self.track_bytes(-(size as isize));
                // Suppress Token::drop — slot already freed by take().
                std::mem::forget(token);
                return df;
            }
            // Re-spilled between ensure_loaded and take — retry.
        }
    }

    /// Clone the stored [`DataFrame`] without consuming the [`Token`].
    pub async fn df(&self, token: &Token) -> DataFrame {
        let id = token.id();
        loop {
            self.ensure_loaded(token).await;
            if let Some(guard) = self.df_store.get(id.index, id.generation) {
                return guard.with_df(|df| df.clone());
            }
            // Re-spilled between ensure_loaded and get — retry.
        }
    }

    /// Blocking variant of [`df`](Self::df).
    pub fn df_blocking(&self, token: &Token) -> DataFrame {
        let id = token.id();
        loop {
            self.ensure_loaded_blocking(token);
            if let Some(guard) = self.df_store.get(id.index, id.generation) {
                return guard.with_df(|df| df.clone());
            }
        }
    }

    /// Apply a mutating closure to the stored [`DataFrame`] in place.
    pub async fn with_df_mut<F, R>(&self, token: &Token, f: F) -> R
    where
        F: FnOnce(&mut DataFrame) -> R,
    {
        let id = token.id();
        loop {
            self.ensure_loaded(token).await;
            let old_size = self.df_store.size_bytes(id.index);
            if let Some(mut guard) = self.df_store.get_mut(id.index, id.generation) {
                let r = guard.with_df_mut(f);
                drop(guard); // updates height/size_bytes atomics
                let new_size = self.df_store.size_bytes(id.index);
                self.track_bytes(new_size as isize - old_size as isize);
                return r;
            }
            // Re-spilled between ensure_loaded and get_mut — retry.
        }
    }

    /// Ensure the entry is loaded into memory. If spilled, one thread
    /// loads from disk while others register wakers and yield.
    async fn ensure_loaded(&self, token: &Token) {
        let id = token.id();
        loop {
            match self.df_store.try_load(id.index, id.generation) {
                LoadStatus::AlreadyLoaded => return,
                LoadStatus::Claimed => {
                    let seq = self.df_store.spill_seq(id.index);
                    if polars_config::config().verbose() {
                        eprintln!(
                            "[ooc] reload index={} generation={} seq={seq}",
                            id.index, id.generation
                        );
                    }
                    let df = self.spiller.load_blocking(id.index, id.generation, seq);
                    let size = df.estimated_size();
                    self.df_store.finish_load(id.index, id.generation, df);
                    self.track_bytes(size as isize);
                    return;
                },
                LoadStatus::Waiting => {
                    let slot = self.df_store.locate(id.index);
                    let mut notified = std::pin::pin!(slot.notify.notified());
                    notified.as_mut().enable();
                    let meta = self.df_store.load_meta(id.index);
                    if matches!(meta.spill(), SpillState::Loading | SpillState::Spilling) {
                        notified.await;
                    }
                },
            }
        }
    }

    /// Blocking variant of [`ensure_loaded`](Self::ensure_loaded). Spins
    /// with [`std::thread::yield_now`] while another thread is loading.
    fn ensure_loaded_blocking(&self, token: &Token) {
        let id = token.id();
        loop {
            match self.df_store.try_load(id.index, id.generation) {
                LoadStatus::AlreadyLoaded => return,
                LoadStatus::Claimed => {
                    let seq = self.df_store.spill_seq(id.index);
                    if polars_config::config().verbose() {
                        eprintln!(
                            "[ooc] reload index={} generation={} seq={seq}",
                            id.index, id.generation
                        );
                    }
                    let df = self.spiller.load_blocking(id.index, id.generation, seq);
                    let size = df.estimated_size();
                    self.df_store.finish_load(id.index, id.generation, df);
                    self.track_bytes(size as isize);
                    return;
                },
                LoadStatus::Waiting => std::thread::yield_now(),
            }
        }
    }

    /// Currently delegates to [`spill_blocking`](Self::spill_blocking).
    async fn spill(&self) {
        self.spill_blocking();
    }

    /// Spill frames to disk to free memory. Collects entries via
    /// [`collect_spill_entries`](Self::collect_spill_entries), then writes
    /// them to disk with no locks held.
    fn spill_blocking(&self) {
        if polars_config::config().verbose() {
            eprintln!(
                "[ooc] spill_trigger total_bytes={} budget={}",
                self.total_bytes(),
                self.budget()
            );
        }
        let (pending, freed) = self.collect_spill_entries();

        // All slots in Spilling state — write to disk, then finalize.
        for (idx, generation, seq, df) in pending {
            if polars_config::config().verbose() {
                eprintln!(
                    "[ooc] spill index={idx} generation={generation} seq={seq} size={} rows={}",
                    df.estimated_size(),
                    df.height()
                );
            }
            self.spiller.spill(idx, generation, seq, df);
            // Transition Spilling → Spilled now that the file exists.
            self.df_store.finish_spill(idx, generation);
        }

        if polars_config::config().verbose() {
            eprintln!(
                "[ooc] spill_end freed={freed} total_after={}",
                self.total_bytes()
            );
        }
    }

    /// Collect entries to spill by scanning all registered [`SpillContext`]s.
    ///
    /// Escalates the target amount on repeated calls: 1/8 → 1/4 → 1/2
    /// of budget.
    fn collect_spill_entries(&self) -> (Vec<(u32, u32, u32, DataFrame)>, usize) {
        let fraction = self.spiller.spill_fraction_and_escalate();
        let must_free = (self.budget() as f64 * fraction) as usize;

        let current = self.total_bytes();

        if polars_config::config().verbose() {
            eprintln!(
                "[ooc] spill total_bytes={current} budget={} must_free={must_free} fraction={fraction}",
                self.budget(),
            );
        }

        let mut freed = 0usize;
        let mut pending: Vec<(u32, u32, u32, DataFrame)> = Vec::new();

        freed += self.collect_from_contexts(must_free, &mut pending);

        (pending, freed)
    }

    /// Scan all registered [`SpillContext`] lists and spill eligible entries.
    ///
    /// Stale and already-spilled entries are skipped. Entries remain in the
    /// list permanently — after reload they become InMemory again and are
    /// eligible for future spills. Contexts are dropped at query boundaries
    /// via [`init_for_query`](Self::init_for_query).
    fn collect_from_contexts(
        &self,
        must_free: usize,
        pending: &mut Vec<(u32, u32, u32, DataFrame)>,
    ) -> usize {
        let mut freed = 0usize;

        self.contexts.scan(|ctx| {
            if freed >= must_free {
                return;
            }

            ctx.fifo_entries.scan(|id| {
                if freed >= must_free {
                    return;
                }

                // Skip pinned or non-InMemory entries without entering
                // try_spill's CAS loop.
                let meta = self.df_store.load_meta(id.index);
                if meta.generation() != id.generation
                    || meta.spill() != SpillState::InMemory
                    || meta.is_pinned()
                {
                    return;
                }

                if let Some((df, size, seq)) = self.df_store.try_spill(id.index, id.generation) {
                    self.track_bytes(-(size as isize));
                    pending.push((id.index, id.generation, seq, df));
                    freed += size;
                }
            });
        });

        freed
    }

    /// Approximate total bytes tracked across all threads.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("policy", &polars_config::config().ooc_spill_policy())
            .field("total_bytes", &self.total_bytes())
            .field("budget", &self.budget())
            .field("slots", &self.df_store.len())
            .finish_non_exhaustive()
    }
}
