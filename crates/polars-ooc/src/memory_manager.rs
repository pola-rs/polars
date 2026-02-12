//! Core implementation of the out-of-core memory manager.
//!
//! The [`MemoryManager`] is a global singleton (`LazyLock<MemoryManager>`) that
//! owns a dynamically-sized array of per-thread stores.

use std::cell::Cell;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use parking_lot::Mutex;
use polars_core::error::{PolarsError, PolarsResult, polars_bail, polars_err};
use polars_core::prelude::DataFrame;
use slotmap::{Key, KeyData, SlotMap, new_key_type};

use crate::spiller::{Format, Spiller};
use crate::token::Token;

new_key_type! {
    struct DfKey;
}

static MEMORY_MANAGER: LazyLock<MemoryManager> = LazyLock::new(MemoryManager::default);

/// Get the global memory manager.
pub fn mm() -> &'static MemoryManager {
    &MEMORY_MANAGER
}

const UNREGISTERED: u32 = u32::MAX;
static NEXT_IDX: AtomicU32 = AtomicU32::new(0);

thread_local! {
    static THREAD_IDX: Cell<u32> = const { Cell::new(UNREGISTERED) };
}

#[derive(Debug, Clone, Copy, Default)]
pub enum SpillPolicy {
    #[default]
    InMemory,
    Spill,
}

struct Entry {
    df: DataFrame,
    size_bytes: usize,
    is_spilled: bool,
}

/// Returns the number of executor threads.
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
        .get()
}

/// Per-thread local memory budget (64 MB). When a thread's local usage
/// exceeds this, it signals the global `MemoryManager` to coordinate
/// (e.g. trigger spilling).
const LOCAL_LIMIT: usize = 64 * 1024 * 1024;

struct ThreadLocalData {
    slots: SlotMap<DfKey, Entry>,
    local_bytes: usize,
}

/// A per-thread store protected by a `Mutex`.
struct ThreadLocalMemoryManager(Mutex<ThreadLocalData>);

impl ThreadLocalMemoryManager {
    fn new() -> Self {
        Self(Mutex::new(ThreadLocalData {
            slots: SlotMap::with_key(),
            local_bytes: 0,
        }))
    }
}

/// Global memory manager for the streaming engine's buffered DataFrames.
///
/// Holds a per-thread array of [`ThreadLocalMemoryManager`] stores, each
/// containing a `SlotMap` of [`Entry`] values. Threads register lazily on
/// first [`store`](Self::store) call and get a dedicated slot.
pub struct MemoryManager {
    policy: SpillPolicy,
    spiller: Spiller,
    stores: Box<[ThreadLocalMemoryManager]>,
    total_bytes: AtomicUsize,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(SpillPolicy::default())
    }
}

impl MemoryManager {
    pub fn new(policy: SpillPolicy) -> Self {
        let n = num_threads();
        Self {
            policy,
            spiller: Spiller::new(Format::Ipc),
            stores: (0..n).map(|_| ThreadLocalMemoryManager::new()).collect(),
            total_bytes: AtomicUsize::new(0),
        }
    }

    fn thread_idx(&self) -> u32 {
        THREAD_IDX.with(|cell| {
            let idx = cell.get();
            if idx != UNREGISTERED {
                return idx;
            }
            let new_idx = NEXT_IDX.fetch_add(1, Ordering::Relaxed) % self.stores.len() as u32;
            cell.set(new_idx);
            new_idx
        })
    }

    fn key(token: &Token) -> DfKey {
        DfKey::from(KeyData::from_ffi(token.key_ffi()))
    }

    fn lock(&self, token: &Token) -> parking_lot::MutexGuard<'_, ThreadLocalData> {
        self.stores[token.thread_idx() as usize].0.lock()
    }

    fn not_found(token: &Token) -> PolarsError {
        polars_err!(
            OutOfCore: "data not found (store={}, key={})",
            token.thread_idx(), token.key_ffi()
        )
    }

    /// Remove and return the entry. Returns `Ok(Some(df))` if in memory,
    /// `Ok(None)` if spilled (entry removed), `Err` if not found.
    fn try_take(&self, token: &Token) -> PolarsResult<Option<DataFrame>> {
        let key = Self::key(token);
        let mut tl = self.lock(token);
        let entry = tl.slots.remove(key).ok_or_else(|| Self::not_found(token))?;
        tl.local_bytes -= entry.size_bytes;
        self.total_bytes
            .fetch_sub(entry.size_bytes, Ordering::Relaxed);
        Ok((!entry.is_spilled).then_some(entry.df))
    }

    /// Insert a DataFrame into the thread-local store. Returns `(token, over_limit)`.
    fn insert(&self, df: DataFrame) -> (Token, bool) {
        let size_bytes = df.estimated_size();
        let height = df.height() as u32;
        let idx = self.thread_idx();

        let (key, over_limit) = {
            let mut tl = self.stores[idx as usize].0.lock();
            tl.local_bytes += size_bytes;
            let key = tl.slots.insert(Entry {
                df,
                size_bytes,
                is_spilled: false,
            });
            (
                key,
                !matches!(self.policy, SpillPolicy::InMemory) && tl.local_bytes > LOCAL_LIMIT,
            )
        };

        self.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);
        (Token::new(idx, height, key.data().as_ffi()), over_limit)
    }

    /// Store a DataFrame and return a `Token`.
    ///
    /// If over budget, triggers spill coordination asynchronously.
    pub async fn store(&self, df: DataFrame) -> PolarsResult<Token> {
        let (token, over_limit) = self.insert(df);
        if over_limit {
            self.coordinate_spill(token.thread_idx()).await?;
        }
        Ok(token)
    }

    /// Store a DataFrame and return a `Token`.
    ///
    /// If over budget, triggers spill coordination with blocking I/O.
    pub fn store_sync(&self, df: DataFrame) -> PolarsResult<Token> {
        let (token, over_limit) = self.insert(df);
        if over_limit {
            self.coordinate_spill_sync(token.thread_idx())?;
        }
        Ok(token)
    }

    /// Get a **clone** of the stored DataFrame.
    ///
    /// If the data was spilled to disk, loads it back asynchronously.
    pub async fn df(&self, token: &Token) -> PolarsResult<DataFrame> {
        let key = Self::key(token);
        {
            let tl = self.lock(token);
            let entry = tl.slots.get(key).ok_or_else(|| Self::not_found(token))?;
            if !entry.is_spilled {
                return Ok(entry.df.clone());
            }
        }
        self.spiller.load(token.key_ffi()).await
    }

    /// Take ownership of the stored DataFrame, removing it from storage.
    ///
    /// If the data was spilled to disk, loads it back asynchronously.
    pub async fn take(&self, token: Token) -> PolarsResult<DataFrame> {
        match self.try_take(&token)? {
            Some(df) => Ok(df),
            None => self.spiller.load(token.key_ffi()).await,
        }
    }

    /// Take ownership of the stored DataFrame, removing it from storage.
    ///
    /// If the data was spilled to disk, loads it back with blocking I/O.
    pub fn take_sync(&self, token: Token) -> PolarsResult<DataFrame> {
        match self.try_take(&token)? {
            Some(df) => Ok(df),
            None => self.spiller.load_sync(token.key_ffi()),
        }
    }

    /// Mutate the stored DataFrame in place.
    ///
    /// If the data was spilled, loads it back from disk first, applies `f`,
    /// then keeps it in memory.
    pub async fn with_df_mut<F, R>(&self, token: &mut Token, f: F) -> PolarsResult<R>
    where
        F: FnOnce(&mut DataFrame) -> R,
    {
        let key = Self::key(token);
        {
            let mut tl = self.lock(token);
            let entry = tl
                .slots
                .get_mut(key)
                .ok_or_else(|| Self::not_found(token))?;
            if !entry.is_spilled {
                let r = f(&mut entry.df);
                token.set_height(entry.df.height());
                return Ok(r);
            }
        }
        let df = self.spiller.load(token.key_ffi()).await?;
        let mut tl = self.lock(token);
        let entry = tl
            .slots
            .get_mut(key)
            .expect("entry disappeared after spill load");
        entry.df = df;
        entry.is_spilled = false;
        let r = f(&mut entry.df);
        token.set_height(entry.df.height());
        Ok(r)
    }

    /// Trigger spilling to free memory.
    async fn coordinate_spill(&self, _thread_idx: u32) -> PolarsResult<()> {
        match self.policy {
            SpillPolicy::InMemory => {
                polars_bail!(OutOfCore: "coordinate_spill called with InMemory policy")
            },
            SpillPolicy::Spill => {
                // TODO: implement spill.
                unimplemented!("spill coordination")
            },
        }
    }

    /// Trigger spilling to free memory (blocking).
    fn coordinate_spill_sync(&self, _thread_idx: u32) -> PolarsResult<()> {
        match self.policy {
            SpillPolicy::InMemory => {
                polars_bail!(OutOfCore: "coordinate_spill_sync called with InMemory policy")
            },
            SpillPolicy::Spill => {
                // TODO: implement spill.
                unimplemented!("spill coordination")
            },
        }
    }

    /// Total bytes currently stored across all threads.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Frees the stored DataFrame without returning it.
    pub fn drop_sync(&self, token: Token) -> PolarsResult<()> {
        self.take_sync(token).map(|_| ())
    }

    /// Frees the stored DataFrames without returning them.
    pub fn drop_all_sync(&self, tokens: impl IntoIterator<Item = Token>) -> PolarsResult<()> {
        for token in tokens {
            self.drop_sync(token)?;
        }
        Ok(())
    }

    /// Take all DataFrames from an iterator of tokens (sync).
    pub fn take_all_sync(
        &self,
        tokens: impl IntoIterator<Item = Token>,
    ) -> PolarsResult<Vec<DataFrame>> {
        tokens.into_iter().map(|t| self.take_sync(t)).collect()
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("policy", &self.policy)
            .field("num_stores", &self.stores.len())
            .finish_non_exhaustive()
    }
}
