use std::borrow::Cow;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, RwLock};

use bitflags::{bitflags, Flags};
use once_cell::sync::OnceCell;
use polars_core::config::verbose;
use polars_core::frame::group_by::GroupsProxy;
use polars_core::prelude::*;
use polars_ops::prelude::ChunkJoinOptIds;
#[cfg(any(
    feature = "parquet",
    feature = "csv",
    feature = "ipc",
    feature = "json"
))]
use polars_plan::logical_plan::FileFingerPrint;

#[cfg(any(
    feature = "ipc",
    feature = "parquet",
    feature = "csv",
    feature = "json"
))]
use super::file_cache::FileCache;
use crate::physical_plan::node_timer::NodeTimer;

pub type JoinTuplesCache = Arc<Mutex<PlHashMap<String, ChunkJoinOptIds>>>;
pub type GroupsProxyCache = Arc<RwLock<PlHashMap<String, GroupsProxy>>>;

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
        /// If set, the expression is evaluated in the
        /// streaming engine.
        const IN_STREAMING = 0x08;
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

/// State/ cache that is maintained during the Execution of the physical plan.
pub struct ExecutionState {
    // cached by a `.cache` call and kept in memory for the duration of the plan.
    df_cache: Arc<Mutex<PlHashMap<usize, Arc<OnceCell<DataFrame>>>>>,
    // cache file reads until all branches got there file, then we delete it
    #[cfg(any(
        feature = "ipc",
        feature = "parquet",
        feature = "csv",
        feature = "json"
    ))]
    pub(crate) file_cache: FileCache,
    pub(super) schema_cache: RwLock<Option<SchemaRef>>,
    /// Used by Window Expression to prevent redundant grouping
    pub(super) group_tuples: GroupsProxyCache,
    /// Used by Window Expression to prevent redundant joins
    pub(super) join_tuples: JoinTuplesCache,
    // every join/union split gets an increment to distinguish between schema state
    pub(super) branch_idx: usize,
    pub(super) flags: AtomicU8,
    pub(super) ext_contexts: Arc<Vec<DataFrame>>,
    node_timer: Option<NodeTimer>,
}

impl ExecutionState {
    /// Toggle this to measure execution times.
    pub(crate) fn time_nodes(&mut self) {
        self.node_timer = Some(NodeTimer::new())
    }
    pub(super) fn has_node_timer(&self) -> bool {
        self.node_timer.is_some()
    }

    pub(crate) fn finish_timer(self) -> PolarsResult<DataFrame> {
        self.node_timer.unwrap().finish()
    }

    pub(super) fn record<T, F: FnOnce() -> T>(&self, func: F, name: Cow<'static, str>) -> T {
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
    pub(super) fn split(&self) -> Self {
        Self {
            df_cache: self.df_cache.clone(),
            #[cfg(any(
                feature = "ipc",
                feature = "parquet",
                feature = "csv",
                feature = "json"
            ))]
            file_cache: self.file_cache.clone(),
            schema_cache: Default::default(),
            group_tuples: Default::default(),
            join_tuples: Default::default(),
            branch_idx: self.branch_idx,
            flags: AtomicU8::new(self.flags.load(Ordering::Relaxed)),
            ext_contexts: self.ext_contexts.clone(),
            node_timer: self.node_timer.clone(),
        }
    }

    /// clones and partially clears state
    pub(super) fn clone(&self) -> Self {
        Self {
            df_cache: self.df_cache.clone(),
            #[cfg(any(
                feature = "ipc",
                feature = "parquet",
                feature = "csv",
                feature = "json"
            ))]
            file_cache: self.file_cache.clone(),
            schema_cache: self.schema_cache.read().unwrap().clone().into(),
            group_tuples: self.group_tuples.clone(),
            join_tuples: self.join_tuples.clone(),
            branch_idx: self.branch_idx,
            flags: AtomicU8::new(self.flags.load(Ordering::Relaxed)),
            ext_contexts: self.ext_contexts.clone(),
            node_timer: self.node_timer.clone(),
        }
    }

    #[cfg(not(any(feature = "parquet", feature = "csv", feature = "ipc")))]
    pub(crate) fn with_finger_prints(_finger_prints: Option<usize>) -> Self {
        Self::new()
    }
    #[cfg(any(
        feature = "parquet",
        feature = "csv",
        feature = "ipc",
        feature = "json"
    ))]
    pub(crate) fn with_finger_prints(finger_prints: Option<Vec<FileFingerPrint>>) -> Self {
        Self {
            df_cache: Arc::new(Mutex::new(PlHashMap::default())),
            schema_cache: Default::default(),
            #[cfg(any(
                feature = "ipc",
                feature = "parquet",
                feature = "csv",
                feature = "json"
            ))]
            file_cache: FileCache::new(finger_prints),
            group_tuples: Arc::new(RwLock::new(PlHashMap::default())),
            join_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            branch_idx: 0,
            flags: AtomicU8::new(StateFlags::init().as_u8()),
            ext_contexts: Default::default(),
            node_timer: None,
        }
    }

    pub fn new() -> Self {
        let mut flags: StateFlags = Default::default();
        if verbose() {
            flags |= StateFlags::VERBOSE;
        }
        Self {
            df_cache: Default::default(),
            schema_cache: Default::default(),
            #[cfg(any(
                feature = "ipc",
                feature = "parquet",
                feature = "csv",
                feature = "json"
            ))]
            file_cache: FileCache::new(None),
            group_tuples: Default::default(),
            join_tuples: Default::default(),
            branch_idx: 0,
            flags: AtomicU8::new(StateFlags::init().as_u8()),
            ext_contexts: Default::default(),
            node_timer: None,
        }
    }
    pub(crate) fn set_schema(&self, schema: SchemaRef) {
        let mut lock = self.schema_cache.write().unwrap();
        *lock = Some(schema);
    }

    /// Clear the schema. Typically at the end of a projection.
    pub(crate) fn clear_schema_cache(&self) {
        let mut lock = self.schema_cache.write().unwrap();
        *lock = None;
    }

    /// Get the schema.
    pub(crate) fn get_schema(&self) -> Option<SchemaRef> {
        let lock = self.schema_cache.read().unwrap();
        lock.clone()
    }

    pub(crate) fn get_df_cache(&self, key: usize) -> Arc<OnceCell<DataFrame>> {
        let mut guard = self.df_cache.lock().unwrap();
        guard
            .entry(key)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone()
    }

    /// Clear the cache used by the Window expressions
    pub(crate) fn clear_window_expr_cache(&self) {
        {
            let mut lock = self.group_tuples.write().unwrap();
            lock.clear();
        }
        let mut lock = self.join_tuples.lock().unwrap();
        lock.clear();
    }

    fn set_flags(&self, f: &dyn Fn(StateFlags) -> StateFlags) {
        let flags: StateFlags = self.flags.load(Ordering::Relaxed).into();
        let flags = f(flags);
        self.flags.store(flags.as_u8(), Ordering::Relaxed);
    }

    /// Indicates that window expression's [`GroupTuples`] may be cached.
    pub(super) fn cache_window(&self) -> bool {
        let flags: StateFlags = self.flags.load(Ordering::Relaxed).into();
        flags.contains(StateFlags::CACHE_WINDOW_EXPR)
    }

    /// Indicates that window expression's [`GroupTuples`] may be cached.
    pub(super) fn has_window(&self) -> bool {
        let flags: StateFlags = self.flags.load(Ordering::Relaxed).into();
        flags.contains(StateFlags::HAS_WINDOW)
    }

    /// More verbose logging
    pub(super) fn verbose(&self) -> bool {
        let flags: StateFlags = self.flags.load(Ordering::Relaxed).into();
        flags.contains(StateFlags::VERBOSE)
    }

    pub(super) fn remove_cache_window_flag(&mut self) {
        self.set_flags(&|mut flags| {
            flags.remove(StateFlags::CACHE_WINDOW_EXPR);
            flags
        });
    }

    pub(super) fn insert_cache_window_flag(&mut self) {
        self.set_flags(&|mut flags| {
            flags.insert(StateFlags::CACHE_WINDOW_EXPR);
            flags
        });
    }
    // this will trigger some conservative
    pub(super) fn insert_has_window_function_flag(&mut self) {
        self.set_flags(&|mut flags| {
            flags.insert(StateFlags::HAS_WINDOW);
            flags
        });
    }

    #[cfg(feature = "streaming")]
    pub(super) fn set_in_streaming_engine(&mut self) {
        self.set_flags(&|mut flags| {
            flags.insert(StateFlags::IN_STREAMING);
            flags
        });
    }

    #[cfg(feature = "streaming")]
    pub(super) fn in_streaming_engine(&self) -> bool {
        let flags: StateFlags = self.flags.load(Ordering::Relaxed).into();
        flags.contains(StateFlags::IN_STREAMING)
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
