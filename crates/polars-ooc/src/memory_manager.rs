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
/// - [`NoPattern`](AccessPattern::NoPattern):
/// - [`Fifo`](AccessPattern::Fifo): evict the **newest** (last-in) entry.
#[derive(Debug, Clone, Copy, Default)]
pub enum AccessPattern {
    #[default]
    NoPattern,
    Fifo,
}

#[allow(dead_code)]
struct Entry {
    df: DataFrame,
    size_bytes: usize,
    height: usize,
    is_spilled: bool,
    access_pattern: AccessPattern,
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

const MEMORY_BUDGET_FRACTION: f64 = 0.7;

/// Global memory manager that tracks [`DataFrame`]s behind opaque [`Token`]s.
///
/// Each stored frame is assigned to a thread-local slot. Memory usage is tracked
/// per-thread with drift-based synchronization to a global counter, avoiding
/// atomic contention on every store/take. When the budget is exceeded the manager
/// can spill frames to disk and reload them transparently.
pub struct MemoryManager {
    policy: SpillPolicy,
    spiller: Spiller,
    stores: boxcar::Vec<ThreadLocalMemoryManager>,
    total_bytes: AtomicUsize,
    budget: usize,
}

impl Default for MemoryManager {
    fn default() -> Self {
        let cfg = polars_config::config();
        Self::new(cfg.ooc_spill_policy(), cfg.ooc_spill_format())
    }
}

impl MemoryManager {
    /// Create a new [`MemoryManager`] with the given spill policy and format.
    pub fn new(policy: SpillPolicy, format: SpillFormat) -> Self {
        let budget = (polars_utils::sys::total_memory() as f64 * MEMORY_BUDGET_FRACTION) as usize;
        Self {
            policy,
            spiller: Spiller::new(format),
            stores: boxcar::Vec::new(),
            total_bytes: AtomicUsize::new(0),
            budget,
        }
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
            self.spiller.delete(token);
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
                access_pattern,
            });
            (key, self.should_spill(&mut tl))
        };

        (Token::new(idx, key), should_spill)
    }

    /// Check whether the global memory budget is exceeded. Syncs the local
    /// drift first; returns `false` without checking if the drift is too small.
    fn should_spill(&self, tl: &mut ThreadLocalData) -> bool {
        matches!(self.policy, SpillPolicy::Spill)
            && tl.try_sync(&self.total_bytes)
            && self.total_bytes.load(Ordering::Relaxed) > self.budget
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
        self.spiller.load(&token)
    }

    /// Clone the stored [`DataFrame`] without consuming the [`Token`].
    pub async fn df(&self, token: &Token) -> DataFrame {
        {
            let tl = self.lock(token);
            let entry = tl.get(token.key);
            if !entry.is_spilled {
                return entry.df.clone();
            }
        }
        self.spiller.load(token)
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
        let df = self.spiller.load(token);
        let mut tl = self.lock(token);
        let entry = tl.get_mut(token.key);
        entry.df = df;
        entry.is_spilled = false;
        let r = f(&mut entry.df);
        tl.update_entry_size(token.key, &self.total_bytes);
        r
    }

    /// Spill frames from the given thread's store to disk to free memory.
    async fn spill(&self, _thread_idx: u64) {
        unimplemented!("spilling to disk")
    }

    /// Blocking variant of [`spill`](Self::spill).
    fn spill_blocking(&self, _thread_idx: u64) {
        unimplemented!("spilling to disk")
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
            .field("policy", &self.policy)
            .field("total_bytes", &self.total_bytes.load(Ordering::Relaxed))
            .field("budget", &self.budget)
            .field("num_stores", &self.stores.count())
            .finish_non_exhaustive()
    }
}
