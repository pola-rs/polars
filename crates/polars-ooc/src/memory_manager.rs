use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Weak};
use std::task::{Poll, Waker};

use parking_lot::Mutex;
use polars_config::{SpillFormat, SpillPolicy};
use polars_core::prelude::DataFrame;
use slotmap::{SlotMap, new_key_type};

use crate::linked_list::LockFreeLinkedList;
use crate::spiller::Spiller;
use crate::token::Token;

new_key_type! {
    pub(crate) struct DfKey;
}

const _: () = assert!(size_of::<DfKey>() == 8);

static MEMORY_MANAGER: LazyLock<MemoryManager> = LazyLock::new(MemoryManager::default);

/// Return a reference to the global [`MemoryManager`].
pub fn mm() -> &'static MemoryManager {
    &MEMORY_MANAGER
}

const UNREGISTERED_THREAD: u64 = u64::MAX;

thread_local! {
    static THREAD_IDX: Cell<u64> = const { Cell::new(UNREGISTERED_THREAD) };
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

#[derive(Debug)]
enum EntryState {
    InMemory,
    Spilled,
    Loading(Vec<Waker>),
}

impl EntryState {
    fn is_in_memory(&self) -> bool {
        matches!(self, Self::InMemory)
    }

    fn is_spilled(&self) -> bool {
        matches!(self, Self::Spilled)
    }
}

struct Entry {
    df: DataFrame,
    size_bytes: usize,
    height: usize,
    state: EntryState,
    is_pinned: bool,
}

#[derive(Default)]
struct ThreadLocalData {
    slots: SlotMap<DfKey, Entry>,
    total_local_bytes: usize,
    last_sync_total_bytes: usize,
}

impl ThreadLocalData {
    #[inline]
    fn drift(&self) -> isize {
        self.total_local_bytes as isize - self.last_sync_total_bytes as isize
    }

    #[inline]
    fn drift_threshold_reached(&self) -> bool {
        self.drift().unsigned_abs() >= polars_config::config().ooc_drift_threshold() as usize
    }

    fn try_sync(&mut self, global_bytes: &AtomicUsize) -> bool {
        if !self.drift_threshold_reached() {
            return false;
        }
        let drift = self.drift();
        self.last_sync_total_bytes = self.total_local_bytes;
        global_bytes.fetch_add(drift as usize, Ordering::Relaxed);
        true
    }

    /// Update an entry's cached height and size after its DataFrame was
    /// mutated in place, then sync the drift to the global counter.
    fn update_entry_size(&mut self, key: DfKey, global_bytes: &AtomicUsize) {
        let entry = self.get_mut(key);
        entry.height = entry.df.height();
        let new_size = entry.df.estimated_size();
        let old_size = std::mem::replace(&mut entry.size_bytes, new_size);
        self.total_local_bytes = self.total_local_bytes + new_size - old_size;
        self.try_sync(global_bytes);
    }

    fn get(&self, key: DfKey) -> &Entry {
        self.slots.get(key).expect("missing memory manager entry")
    }

    fn get_mut(&mut self, key: DfKey) -> &mut Entry {
        self.slots
            .get_mut(key)
            .expect("missing memory manager entry")
    }
}

#[derive(Default)]
struct ThreadLocalMemoryManager(Mutex<ThreadLocalData>);

/// Per-operator spill context registered via [`MemoryManager::register_context`].
///
/// Each operator creates one context. When data is stored with
/// [`AccessPattern::Fifo`], the entry is pushed onto the context's
/// linked list. During spill collection the memory manager iterates
/// all contexts' lists instead of scanning the full SlotMap.
pub struct SpillContext {
    fifo_entries: LockFreeLinkedList,
}

impl SpillContext {
    fn new() -> Self {
        Self {
            fifo_entries: LockFreeLinkedList::new(),
        }
    }

    pub(crate) fn push(&self, thread_idx: u64, key: DfKey) {
        self.fifo_entries.push((thread_idx, key));
    }
}

/// Global memory manager that tracks [`DataFrame`]s behind opaque [`Token`]s.
///
/// Each stored frame is assigned to a thread-local slot. Memory usage is tracked
/// per-thread with drift-based synchronization to a global counter, avoiding
/// atomic contention on every store/take. When the budget is exceeded the manager
/// can spill frames to disk and reload them transparently.
pub struct MemoryManager {
    pub(crate) spiller: Spiller,
    stores: boxcar::Vec<ThreadLocalMemoryManager>,
    contexts: Mutex<Vec<Weak<SpillContext>>>,
    total_bytes: AtomicUsize,
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
            stores: boxcar::Vec::new(),
            contexts: Mutex::new(Vec::new()),
            total_bytes: AtomicUsize::new(0),
        }
    }

    fn budget(&self) -> usize {
        let fraction = polars_config::config().ooc_memory_budget_fraction();
        (polars_utils::sys::total_memory() as f64 * fraction) as usize
    }

    /// Reset spill escalation level back to 0. Called at the start of
    /// each query so the first spill pass is conservative (1/8 of budget).
    pub fn reset_spill_level(&self) {
        self.spiller.reset_spill_level();
    }

    /// Register a per-operator [`SpillContext`] for spill tracking.
    ///
    /// The returned `Arc` should be held by the operator for its lifetime.
    /// The memory manager stores a `Weak` reference and uses it during
    /// spill collection. When the operator drops its `Arc`, the context
    /// is automatically cleaned up.
    pub fn register_context(&self) -> Arc<SpillContext> {
        let ctx = Arc::new(SpillContext::new());
        self.contexts.lock().push(Arc::downgrade(&ctx));
        ctx
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

    /// Allocate a new thread-local store and cache its index.
    #[cold]
    #[inline(never)]
    fn register_thread(&self, cell: &Cell<u64>) -> u64 {
        let new_idx = self.stores.push(ThreadLocalMemoryManager::default()) as u64;
        cell.set(new_idx);
        new_idx
    }

    /// Lock the thread-local store that owns this [`Token`].
    fn lock(&self, token: &Token) -> parking_lot::MutexGuard<'_, ThreadLocalData> {
        self.stores[token.thread_idx() as usize].0.lock()
    }

    /// Return the row count of the stored [`DataFrame`].
    pub fn height(&self, token: &Token) -> usize {
        let tl = self.lock(token);
        tl.get(token.key).height
    }

    /// Pin the entry so it has lower priority during spill collection.
    /// Unpinned entries are always spilled first; pinned entries are only
    /// spilled if freeing unpinned entries alone is not enough.
    pub fn pin(&self, token: &Token) {
        let mut tl = self.lock(token);
        tl.get_mut(token.key).is_pinned = true;
    }

    /// Unpin the entry so it has higher priority during spill collection.
    pub fn unpin(&self, token: &Token) {
        let mut tl = self.lock(token);
        tl.get_mut(token.key).is_pinned = false;
    }

    /// Remove the entry for this [`Token`], update memory accounting, and
    /// delete the spill file if the frame was spilled. Called by [`Token::drop`].
    pub(crate) fn drop_token(&self, token: &Token) {
        let mut tl = self.lock(token);
        let Some(entry) = tl.slots.remove(token.key) else {
            return;
        };
        tl.total_local_bytes -= entry.size_bytes;
        tl.try_sync(&self.total_bytes);
        if entry.state.is_spilled() {
            self.spiller.delete_spill_file(token);
        }
    }

    /// Insert a [`DataFrame`] into the calling thread's store. Returns the
    /// [`Token`] and whether the memory budget was exceeded after insertion.
    ///
    /// Based on the [`AccessPattern`], the entry may be pushed onto the
    /// context's linked list for FIFO spill tracking.
    fn insert(
        &self,
        df: DataFrame,
        ctx: &SpillContext,
        access_pattern: AccessPattern,
    ) -> (Token, bool) {
        let size_bytes = df.estimated_size();
        let height = df.height();
        let idx = self.thread_idx();
        let (key, should_spill) = {
            let mut tl = self.stores[idx as usize].0.lock();
            tl.total_local_bytes += size_bytes;
            let key = tl.slots.insert(Entry {
                df,
                size_bytes,
                height,
                state: EntryState::InMemory,
                is_pinned: false,
            });
            (key, self.should_spill(&mut tl))
        };

        match access_pattern {
            AccessPattern::Fifo => ctx.push(idx, key),
            AccessPattern::NoPattern => {},
        }

        (Token::new(idx, key), should_spill)
    }

    /// Check whether the global memory budget is exceeded. Syncs the local
    /// drift first; returns `false` without checking if the drift is too small.
    fn should_spill(&self, tl: &mut ThreadLocalData) -> bool {
        matches!(
            polars_config::config().ooc_spill_policy(),
            SpillPolicy::Spill
        ) && tl.try_sync(&self.total_bytes)
            && self.total_bytes.load(Ordering::Relaxed) > self.budget()
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
            self.spill(token.thread_idx()).await;
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
            self.spill_blocking(token.thread_idx());
        }
        token
    }

    /// Take the [`DataFrame`] out of the manager, consuming the [`Token`].
    /// The token's [`Drop`] impl handles slot removal and memory accounting.
    pub async fn take_df(&self, token: Token) -> DataFrame {
        self.ensure_loaded(&token).await;
        let mut tl = self.lock(&token);
        std::mem::take(&mut tl.get_mut(token.key).df)
    }

    /// Clone the stored [`DataFrame`] without consuming the [`Token`].
    /// If the frame was spilled, reloads it into memory first (only one
    /// thread loads; others wait via wakers).
    pub async fn df(&self, token: &Token) -> DataFrame {
        self.ensure_loaded(token).await;
        let tl = self.lock(token);
        tl.get(token.key).df.clone()
    }

    /// Blocking variant of [`df`](Self::df).
    pub fn df_blocking(&self, token: &Token) -> DataFrame {
        self.ensure_loaded_blocking(token);
        let tl = self.lock(token);
        tl.get(token.key).df.clone()
    }

    /// Apply a mutating closure to the stored [`DataFrame`] in place.
    /// Reloads from disk first if the frame was spilled.
    ///
    /// The closure must not call methods on [`MemoryManager`] that lock the
    /// same thread-local store. The store is locked for the duration and
    /// re-entering would deadlock.
    pub async fn with_df_mut<F, R>(&self, token: &Token, f: F) -> R
    where
        F: FnOnce(&mut DataFrame) -> R,
    {
        self.ensure_loaded(token).await;
        let mut tl = self.lock(token);
        let r = f(&mut tl.get_mut(token.key).df);
        tl.update_entry_size(token.key, &self.total_bytes);
        r
    }

    /// Try to claim the load. Returns `None` if already in memory,
    /// `Some(true)` if we transitioned Spilled → Loading (caller must
    /// load), `Some(false)` if another thread is already loading.
    fn try_claim_load(&self, token: &Token) -> Option<bool> {
        let mut tl = self.lock(token);
        let entry = tl.get_mut(token.key);
        match &entry.state {
            EntryState::InMemory => None,
            EntryState::Spilled => {
                entry.state = EntryState::Loading(Vec::new());
                Some(true)
            },
            EntryState::Loading(_) => Some(false),
        }
    }

    /// Complete a load: put the DataFrame back, transition to InMemory,
    /// update accounting, and wake all parked waiters.
    fn finish_load(&self, token: &Token, df: DataFrame) {
        let mut tl = self.lock(token);
        let entry = tl.get_mut(token.key);
        let wakers = match std::mem::replace(&mut entry.state, EntryState::InMemory) {
            EntryState::Loading(wakers) => wakers,
            _ => unreachable!(),
        };
        let size = df.estimated_size();
        entry.df = df;
        entry.size_bytes = size;
        tl.total_local_bytes += size;
        tl.try_sync(&self.total_bytes);
        for w in wakers {
            w.wake();
        }
    }

    /// Ensure the entry is loaded into memory. If spilled, this thread
    /// loads from disk while others register wakers and yield. If another
    /// thread is already loading, registers a waker and suspends.
    async fn ensure_loaded(&self, token: &Token) {
        loop {
            let Some(is_loader) = self.try_claim_load(token) else {
                return;
            };

            if is_loader {
                if polars_config::config().verbose() {
                    eprintln!("[ooc] reload thread={}", token.thread_idx());
                }
                let df = self.spiller.load(token).await;
                self.finish_load(token, df);
                return;
            }

            // Another thread is loading — register waker and yield.
            std::future::poll_fn(|cx| {
                let mut tl = self.lock(token);
                let entry = tl.get_mut(token.key);
                match &mut entry.state {
                    EntryState::Loading(wakers) => {
                        wakers.push(cx.waker().clone());
                        Poll::Pending
                    },
                    EntryState::InMemory => Poll::Ready(()),
                    EntryState::Spilled => {
                        unreachable!("entry cannot be Spilled while a loader is active")
                    },
                }
            })
            .await;
        }
    }

    /// Blocking variant of [`ensure_loaded`](Self::ensure_loaded). Spins
    /// with [`std::thread::yield_now`] while another thread is loading.
    fn ensure_loaded_blocking(&self, token: &Token) {
        loop {
            let Some(is_loader) = self.try_claim_load(token) else {
                return;
            };

            if is_loader {
                if polars_config::config().verbose() {
                    eprintln!("[ooc] reload thread={}", token.thread_idx());
                }
                let df = self.spiller.load_blocking(token);
                self.finish_load(token, df);
                return;
            }

            std::thread::yield_now();
        }
    }

    /// Currently delegates to [`spill_blocking`](Self::spill_blocking)
    /// because async disk I/O is not yet implemented.
    async fn spill(&self, trigger_thread: u64) {
        self.spill_blocking(trigger_thread);
    }

    /// Spill frames to disk to free memory. Collects entries via
    /// [`collect_spill_entries`](Self::collect_spill_entries), then writes
    /// them to disk with no locks held.
    fn spill_blocking(&self, trigger_thread: u64) {
        if polars_config::config().verbose() {
            let total = self.total_bytes.load(Ordering::Relaxed);
            eprintln!(
                "[ooc] spill_trigger thread={trigger_thread} total_bytes={total} budget={}",
                self.budget()
            );
        }
        let (pending, freed) = self.collect_spill_entries(trigger_thread);

        // All locks released — write to disk.
        for (store_idx, key, df) in pending {
            if polars_config::config().verbose() {
                eprintln!(
                    "[ooc] spill store={store_idx} size={} rows={}",
                    df.estimated_size(),
                    df.height()
                );
            }
            self.spiller.spill(store_idx, key, df);
        }

        if polars_config::config().verbose() {
            let after = self.total_bytes.load(Ordering::Relaxed);
            eprintln!("[ooc] spill_end freed={freed} total_after={after}");
        }
    }

    /// Collect entries to spill by iterating all registered
    /// [`SpillContext`]s' linked lists. Returns the collected
    /// `(store_idx, DfKey, DataFrame)` triples and total bytes freed.
    ///
    /// Escalates the target amount on repeated calls: 1/8 → 1/4 → 1/2
    /// of budget.
    fn collect_spill_entries(&self, _trigger_thread: u64) -> (Vec<(u64, DfKey, DataFrame)>, usize) {
        let fraction = self.spiller.spill_fraction_and_escalate();
        let must_free = (self.budget() as f64 * fraction) as usize;

        let current = self.total_bytes.load(Ordering::Relaxed);

        if polars_config::config().verbose() {
            eprintln!(
                "[ooc] spill total_bytes={current} budget={} must_free={must_free} fraction={fraction}",
                self.budget(),
            );
        }

        let mut freed = 0usize;
        let mut pending: Vec<(u64, DfKey, DataFrame)> = Vec::new();

        freed += self.collect_from_contexts(must_free, &mut pending);

        (pending, freed)
    }

    /// Collect entries by iterating all registered [`SpillContext`]
    /// linked lists. Each list is walked from head (newest first). Stale
    /// entries (removed or already spilled) are skipped.
    fn collect_from_contexts(
        &self,
        must_free: usize,
        pending: &mut Vec<(u64, DfKey, DataFrame)>,
    ) -> usize {
        let contexts = self.contexts.lock();
        let mut freed = 0usize;
        let num_stores = self.stores.count();

        for weak_ctx in contexts.iter() {
            if freed >= must_free {
                break;
            }
            let Some(ctx) = weak_ctx.upgrade() else {
                continue;
            };
            for &(thread_idx, key) in ctx.fifo_entries.iter() {
                if freed >= must_free {
                    break;
                }
                let store_idx = thread_idx as usize;
                if store_idx >= num_stores {
                    continue;
                }
                let mut tl = self.stores[store_idx].0.lock();
                let Some(entry) = tl.slots.get(key) else {
                    continue;
                };
                if !entry.state.is_in_memory() {
                    continue;
                }
                freed += self.take_entry_for_spill(&mut tl, key, thread_idx, pending);
            }
        }

        freed
    }

    /// Take a single entry's DataFrame for spilling. Marks the entry as
    /// spilled and pushes `(store_idx, key, DataFrame)` to `pending`. Returns bytes freed.
    fn take_entry_for_spill(
        &self,
        tl: &mut ThreadLocalData,
        key: DfKey,
        store_idx: u64,
        pending: &mut Vec<(u64, DfKey, DataFrame)>,
    ) -> usize {
        let entry = tl.get_mut(key);
        assert!(
            entry.state.is_in_memory(),
            "attempted to spill an entry in state {:?}",
            entry.state
        );
        let df = std::mem::take(&mut entry.df);
        let size = entry.size_bytes;
        entry.size_bytes = 0;
        entry.state = EntryState::Spilled;
        tl.total_local_bytes -= size;
        pending.push((store_idx, key, df));
        size
    }

    /// Approximate total bytes tracked across all threads.
    ///
    /// This may lag behind actual memory usage because each thread only syncs its
    /// local counter to the global atomic when the drift exceeds the threshold.
    /// An exact variant that flushes all pending drifts can be added later.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Delete the process spill directory and all remaining spill files.
    ///
    /// Should be called at end of query. Individual token drops (layer 1)
    /// handle most files, but this catches any stragglers and removes the
    /// directory itself.
    pub fn cleanup(&self) {
        self.spiller.cleanup();
        crate::cleaner::shutdown();
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("policy", &polars_config::config().ooc_spill_policy())
            .field("total_bytes", &self.total_bytes.load(Ordering::Relaxed))
            .field("budget", &self.budget())
            .field("num_stores", &self.stores.count())
            .finish_non_exhaustive()
    }
}
