use std::borrow::Cow;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use bitflags::bitflags;
use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_ops::prelude::ChunkJoinOptIds;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::unique_id::UniqueId;

use super::NodeTimer;

pub type JoinTuplesCache = Arc<Mutex<PlHashMap<String, ChunkJoinOptIds>>>;

#[derive(Default)]
pub struct WindowCache {
    groups: RwLock<PlHashMap<String, GroupPositions>>,
    join_tuples: RwLock<PlHashMap<String, Arc<ChunkJoinOptIds>>>,
    map_idx: RwLock<PlHashMap<String, Arc<IdxCa>>>,
}

impl WindowCache {
    pub(crate) fn clear(&self) {
        let mut g = self.groups.write().unwrap();
        g.clear();
        let mut g = self.join_tuples.write().unwrap();
        g.clear();
    }

    pub fn get_groups(&self, key: &str) -> Option<GroupPositions> {
        let g = self.groups.read().unwrap();
        g.get(key).cloned()
    }

    pub fn insert_groups(&self, key: String, groups: GroupPositions) {
        let mut g = self.groups.write().unwrap();
        g.insert(key, groups);
    }

    pub fn get_join(&self, key: &str) -> Option<Arc<ChunkJoinOptIds>> {
        let g = self.join_tuples.read().unwrap();
        g.get(key).cloned()
    }

    pub fn insert_join(&self, key: String, join_tuples: Arc<ChunkJoinOptIds>) {
        let mut g = self.join_tuples.write().unwrap();
        g.insert(key, join_tuples);
    }

    pub fn get_map(&self, key: &str) -> Option<Arc<IdxCa>> {
        let g = self.map_idx.read().unwrap();
        g.get(key).cloned()
    }

    pub fn insert_map(&self, key: String, idx: Arc<IdxCa>) {
        let mut g = self.map_idx.write().unwrap();
        g.insert(key, idx);
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub(super) struct StateFlags: u8 {
        /// More verbose logging
        const VERBOSE = 0x01;
        /// Indicates that window expression's [`GroupTuples`] may be cached.
        const CACHE_WINDOW_EXPR = 0x02;
        /// Indicates the expression has a window function
        const HAS_WINDOW = 0x04;
    }
}

impl Default for StateFlags {
    fn default() -> Self {
        StateFlags::CACHE_WINDOW_EXPR
    }
}

impl StateFlags {
    fn init() -> Self {
        let verbose = verbose();
        let mut flags: StateFlags = Default::default();
        if verbose {
            flags |= StateFlags::VERBOSE;
        }
        flags
    }
    fn as_u8(self) -> u8 {
        unsafe { std::mem::transmute(self) }
    }
}

impl From<u8> for StateFlags {
    fn from(value: u8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

struct CachedValue {
    /// The number of times the cache will still be read.
    /// Zero means that there will be no more reads and the cache can be dropped.
    remaining_hits: AtomicI64,
    df: DataFrame,
}

/// State/ cache that is maintained during the Execution of the physical plan.
#[derive(Clone)]
pub struct ExecutionState {
    // cached by a `.cache` call and kept in memory for the duration of the plan.
    df_cache: Arc<RwLock<PlHashMap<UniqueId, Arc<CachedValue>>>>,
    pub schema_cache: Arc<RwLock<Option<SchemaRef>>>,
    /// Used by Window Expressions to cache intermediate state
    pub window_cache: Arc<WindowCache>,
    // every join/union split gets an increment to distinguish between schema state
    pub branch_idx: usize,
    pub flags: RelaxedCell<u8>,
    pub ext_contexts: Arc<Vec<DataFrame>>,
    node_timer: Option<NodeTimer>,
    stop: Arc<RelaxedCell<bool>>,
}

impl ExecutionState {
    pub fn new() -> Self {
        let mut flags: StateFlags = Default::default();
        if verbose() {
            flags |= StateFlags::VERBOSE;
        }
        Self {
            df_cache: Default::default(),
            schema_cache: Default::default(),
            window_cache: Default::default(),
            branch_idx: 0,
            flags: RelaxedCell::from(StateFlags::init().as_u8()),
            ext_contexts: Default::default(),
            node_timer: None,
            stop: Arc::new(RelaxedCell::from(false)),
        }
    }

    /// Toggle this to measure execution times.
    pub fn time_nodes(&mut self, start: std::time::Instant) {
        self.node_timer = Some(NodeTimer::new(start))
    }
    pub fn has_node_timer(&self) -> bool {
        self.node_timer.is_some()
    }

    pub fn finish_timer(self) -> PolarsResult<DataFrame> {
        self.node_timer.unwrap().finish()
    }

    // Timings should be a list of (start, end, name) where the start
    // and end are raw durations since the query start as nanoseconds.
    pub fn record_raw_timings(&self, timings: &[(u64, u64, String)]) {
        for &(start, end, ref name) in timings {
            self.node_timer.as_ref().unwrap().store_duration(
                Duration::from_nanos(start),
                Duration::from_nanos(end),
                name.to_string(),
            );
        }
    }

    // This is wrong when the U64 overflows which will never happen.
    pub fn should_stop(&self) -> PolarsResult<()> {
        try_raise_keyboard_interrupt();
        polars_ensure!(!self.stop.load(), ComputeError: "query interrupted");
        Ok(())
    }

    pub fn cancel_token(&self) -> Arc<RelaxedCell<bool>> {
        self.stop.clone()
    }

    pub fn record<T, F: FnOnce() -> T>(&self, func: F, name: Cow<'static, str>) -> T {
        match &self.node_timer {
            None => func(),
            Some(timer) => {
                let start = std::time::Instant::now();
                let out = func();
                let end = std::time::Instant::now();

                timer.store(start, end, name.as_ref().to_string());
                out
            },
        }
    }

    /// Partially clones and partially clears state
    /// This should be used when splitting a node, like a join or union
    pub fn split(&self) -> Self {
        Self {
            df_cache: self.df_cache.clone(),
            schema_cache: Default::default(),
            window_cache: Default::default(),
            branch_idx: self.branch_idx,
            flags: self.flags.clone(),
            ext_contexts: self.ext_contexts.clone(),
            node_timer: self.node_timer.clone(),
            stop: self.stop.clone(),
        }
    }

    pub fn set_schema(&self, schema: SchemaRef) {
        let mut lock = self.schema_cache.write().unwrap();
        *lock = Some(schema);
    }

    /// Clear the schema. Typically at the end of a projection.
    pub fn clear_schema_cache(&self) {
        let mut lock = self.schema_cache.write().unwrap();
        *lock = None;
    }

    /// Get the schema.
    pub fn get_schema(&self) -> Option<SchemaRef> {
        let lock = self.schema_cache.read().unwrap();
        lock.clone()
    }

    pub fn set_df_cache(&self, id: &UniqueId, df: DataFrame, cache_hits: u32) {
        if self.verbose() {
            eprintln!("CACHE SET: cache id: {id}");
        }

        let value = Arc::new(CachedValue {
            remaining_hits: AtomicI64::new(cache_hits as i64),
            df,
        });

        let prev = self.df_cache.write().unwrap().insert(*id, value);
        assert!(prev.is_none(), "duplicate set cache: {id}");
    }

    pub fn get_df_cache(&self, id: &UniqueId) -> DataFrame {
        let guard = self.df_cache.read().unwrap();
        let value = guard.get(id).expect("cache not prefilled");
        let remaining = value.remaining_hits.fetch_sub(1, Ordering::Relaxed);
        if remaining < 0 {
            panic!("cache used more times than expected: {id}");
        }
        if self.verbose() {
            eprintln!("CACHE HIT: cache id: {id}");
        }
        if remaining == 1 {
            drop(guard);
            let value = self.df_cache.write().unwrap().remove(id).unwrap();
            if self.verbose() {
                eprintln!("CACHE DROP: cache id: {id}");
            }
            Arc::into_inner(value).unwrap().df
        } else {
            value.df.clone()
        }
    }

    /// Clear the cache used by the Window expressions
    pub fn clear_window_expr_cache(&self) {
        self.window_cache.clear();
    }

    fn set_flags(&self, f: &dyn Fn(StateFlags) -> StateFlags) {
        let flags: StateFlags = self.flags.load().into();
        let flags = f(flags);
        self.flags.store(flags.as_u8());
    }

    /// Indicates that window expression's [`GroupTuples`] may be cached.
    pub fn cache_window(&self) -> bool {
        let flags: StateFlags = self.flags.load().into();
        flags.contains(StateFlags::CACHE_WINDOW_EXPR)
    }

    /// Indicates that window expression's [`GroupTuples`] may be cached.
    pub fn has_window(&self) -> bool {
        let flags: StateFlags = self.flags.load().into();
        flags.contains(StateFlags::HAS_WINDOW)
    }

    /// More verbose logging
    pub fn verbose(&self) -> bool {
        let flags: StateFlags = self.flags.load().into();
        flags.contains(StateFlags::VERBOSE)
    }

    pub fn remove_cache_window_flag(&mut self) {
        self.set_flags(&|mut flags| {
            flags.remove(StateFlags::CACHE_WINDOW_EXPR);
            flags
        });
    }

    pub fn insert_cache_window_flag(&mut self) {
        self.set_flags(&|mut flags| {
            flags.insert(StateFlags::CACHE_WINDOW_EXPR);
            flags
        });
    }
    // this will trigger some conservative
    pub fn insert_has_window_function_flag(&mut self) {
        self.set_flags(&|mut flags| {
            flags.insert(StateFlags::HAS_WINDOW);
            flags
        });
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
