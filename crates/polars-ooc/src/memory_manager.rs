use std::cell::Cell;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use polars_config::{SpillFormat, SpillPolicy};
use polars_core::prelude::DataFrame;
use slotmap::{SlotMap, new_key_type};

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
/// - [`NoPattern`](AccessPattern::NoPattern): evict all entries above `ooc_spill_min_bytes`.
/// - [`Fifo`](AccessPattern::Fifo): evict the **newest** (last-in) entry.
#[derive(Debug, Clone, Copy, Default)]
pub enum AccessPattern {
    #[default]
    NoPattern,
    Fifo,
}

struct Entry {
    df: DataFrame,
    size_bytes: usize,
    height: usize,
    is_spilled: bool,
    is_pinned: bool,
    access_pattern: AccessPattern,
}

#[derive(Default)]
struct ThreadLocalData {
    slots: SlotMap<DfKey, Entry>,
    contexts: Vec<SpillContext>,
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

/// Operator-level context shared with the memory manager.
///
/// Operators register these via [`MemoryManager::register_context`] to
/// share metadata about their buffering behaviour. The memory manager
/// can consult them during spill decisions (e.g. prioritize spilling
/// operators that buffer the most, or prefer operators with cheaper
/// reload cost).
///
/// Currently a placeholder. Fields will be added as the spill heuristics
/// evolve.
#[derive(Debug, Clone, Default)]
pub struct SpillContext {
    // TODO: operator name, expected buffer size, spill priority, etc.
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
            total_bytes: AtomicUsize::new(0),
        }
    }

    fn budget(&self) -> usize {
        let fraction = polars_config::config().ooc_memory_budget_fraction();
        (polars_utils::sys::total_memory() as f64 * fraction) as usize
    }

    /// Register operator-level context for spill analysis.
    ///
    /// Stored in the calling thread's local data and consulted during
    /// spill decisions.
    pub fn register_context(&self, ctx: SpillContext) {
        let idx = self.thread_idx();
        self.stores[idx as usize].0.lock().contexts.push(ctx);
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
        if entry.is_spilled {
            self.spiller.delete_spill_file(token);
        }
    }

    /// Insert a [`DataFrame`] into the calling thread's store. Returns the
    /// [`Token`] and whether the memory budget was exceeded after insertion.
    fn insert(&self, df: DataFrame, access_pattern: AccessPattern) -> (Token, bool) {
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
                is_spilled: false,
                is_pinned: false,
                access_pattern,
            });
            (key, self.should_spill(&mut tl))
        };

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
    pub async fn store(&self, df: DataFrame, pattern: AccessPattern) -> Token {
        let (token, should_spill) = self.insert(df, pattern);
        if should_spill {
            self.spill(token.thread_idx()).await;
        }
        token
    }

    /// Blocking variant of [`store`](Self::store).
    pub fn store_blocking(&self, df: DataFrame, pattern: AccessPattern) -> Token {
        let (token, should_spill) = self.insert(df, pattern);
        if should_spill {
            self.spill_blocking(token.thread_idx());
        }
        token
    }

    /// Take the [`DataFrame`] out of the manager, consuming the [`Token`].
    /// The token's [`Drop`] impl handles slot removal and memory accounting.
    pub async fn take_df(&self, token: Token) -> DataFrame {
        {
            let mut tl = self.lock(&token);
            let entry = tl.get_mut(token.key);
            if !entry.is_spilled {
                return std::mem::take(&mut entry.df);
            }
        }
        if polars_config::config().verbose() {
            eprintln!("[ooc] reload thread={} op=take_df", token.thread_idx());
        }
        self.spiller.load(&token).await
    }

    /// Clone the stored [`DataFrame`] without consuming the [`Token`].
    /// Loads from disk and deletes the spill file if spilled, but does not
    /// cache the result back or update memory accounting. The returned
    /// DataFrame is in-flight data owned by the caller, not tracked by
    /// the memory manager.
    pub async fn df(&self, token: &Token) -> DataFrame {
        {
            let tl = self.lock(token);
            let entry = tl.get(token.key);
            if !entry.is_spilled {
                return entry.df.clone();
            }
        }
        self.spiller.load(token).await
    }

    /// Blocking variant of [`df`](Self::df).
    pub fn df_blocking(&self, token: &Token) -> DataFrame {
        let tl = self.lock(token);
        let entry = tl.get(token.key);
        if !entry.is_spilled {
            return entry.df.clone();
        }
        drop(tl);
        self.spiller.load_blocking(token)
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
        {
            let mut tl = self.lock(token);
            let entry = tl.get_mut(token.key);
            if !entry.is_spilled {
                let r = f(&mut entry.df);
                tl.update_entry_size(token.key, &self.total_bytes);
                return r;
            }
        }
        // Reload from disk without holding the lock.
        let df = self.spiller.load(token).await;
        let mut tl = self.lock(token);
        let entry = tl.get_mut(token.key);
        entry.df = df;
        entry.is_spilled = false;
        let r = f(&mut entry.df);
        tl.update_entry_size(token.key, &self.total_bytes);
        r
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

    /// Collect entries to spill across all stores. Returns the collected
    /// `(Token, DataFrame)` pairs and total bytes freed.
    ///
    /// The trigger thread's store is always visited first (local-first),
    /// then the remaining stores. Within that ordering the priority is:
    ///
    /// 1. **FIFO** — newest first (highest slot key), unpinned only.
    /// 2. **NoPattern** — all entries above `ooc_spill_min_bytes` (unpinned first, then pinned).
    ///
    /// Entries are taken out under locks; the caller writes to disk after
    /// all locks are released.
    ///
    /// Escalates the target amount on repeated calls: 1/8 → 1/4 → 1/2
    /// of budget.
    fn collect_spill_entries(&self, trigger_thread: u64) -> (Vec<(u64, DfKey, DataFrame)>, usize) {
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
        let num_stores = self.stores.count();

        // Local store first, then the rest.
        let local = trigger_thread as usize;
        let store_order: Vec<usize> = std::iter::once(local)
            .chain((0..num_stores).filter(|&i| i != local))
            .collect();

        // Phase 1: FIFO entries (newest first).
        for &store_idx in &store_order {
            if freed >= must_free {
                break;
            }
            freed += self.collect_fifo_from_store(store_idx, must_free - freed, &mut pending);
        }

        // Phase 2: NoPattern entries above min_spill_size (unpinned first, then pinned).
        for pinned in [false, true] {
            if freed >= must_free {
                break;
            }
            for &store_idx in &store_order {
                if freed >= must_free {
                    break;
                }
                freed += self.collect_no_pattern_from_store(
                    store_idx,
                    must_free - freed,
                    pinned,
                    &mut pending,
                );
            }
        }

        (pending, freed)
    }

    /// Collect FIFO entries to spill (newest first). When `pinned` is false,
    /// FIFO entries are never pinned, so this always collects unpinned entries.
    fn collect_fifo_from_store(
        &self,
        store_idx: usize,
        remaining: usize,
        pending: &mut Vec<(u64, DfKey, DataFrame)>,
    ) -> usize {
        let mut tl = self.stores[store_idx].0.lock();
        let mut freed = 0usize;

        let mut fifo_keys: Vec<DfKey> = tl
            .slots
            .keys()
            .filter(|&k| {
                let e = tl.get(k);
                matches!(e.access_pattern, AccessPattern::Fifo) && !e.is_spilled
            })
            .collect();
        fifo_keys.reverse();

        for key in fifo_keys {
            if freed >= remaining {
                break;
            }
            freed += self.take_entry_for_spill(&mut tl, key, store_idx as u64, pending);
        }

        tl.try_sync(&self.total_bytes);
        freed
    }

    /// Collect NoPattern entries to spill. Evicts all entries whose size
    /// exceeds `ooc_spill_min_bytes` in a single pass. When `pinned` is
    /// false, only collects unpinned entries; when true, only collects pinned.
    fn collect_no_pattern_from_store(
        &self,
        store_idx: usize,
        remaining: usize,
        pinned: bool,
        pending: &mut Vec<(u64, DfKey, DataFrame)>,
    ) -> usize {
        let mut tl = self.stores[store_idx].0.lock();
        let min_size = polars_config::config().ooc_spill_min_bytes() as usize;
        let mut freed = 0usize;

        let keys: Vec<DfKey> = tl
            .slots
            .iter()
            .filter(|(_, entry)| {
                matches!(entry.access_pattern, AccessPattern::NoPattern)
                    && !entry.is_spilled
                    && entry.is_pinned == pinned
                    && entry.size_bytes >= min_size
            })
            .map(|(key, _)| key)
            .collect();

        for key in keys {
            if freed >= remaining {
                break;
            }
            freed += self.take_entry_for_spill(&mut tl, key, store_idx as u64, pending);
        }

        tl.try_sync(&self.total_bytes);
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
            !entry.is_spilled,
            "attempted to spill an already-spilled entry"
        );
        let df = std::mem::take(&mut entry.df);
        let size = entry.size_bytes;
        entry.size_bytes = 0;
        entry.is_spilled = true;
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
