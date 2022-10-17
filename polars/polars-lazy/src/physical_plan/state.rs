use std::borrow::Cow;
use std::sync::Mutex;

use bitflags::bitflags;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::frame::hash_join::JoinOptIds;
use polars_core::prelude::*;
#[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
use polars_plan::logical_plan::FileFingerPrint;

#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
use super::file_cache::FileCache;
use crate::physical_plan::node_timer::NodeTimer;

pub type JoinTuplesCache = Arc<Mutex<PlHashMap<String, JoinOptIds>>>;
pub type GroupsProxyCache = Arc<Mutex<PlHashMap<String, GroupsProxy>>>;

bitflags! {
    pub(super) struct StateFlags: u8 {
        /// More verbose logging
        const VERBOSE = 0x01;
        /// Indicates that window expression's [`GroupTuples`] may be cached.
        const CACHE_WINDOW_EXPR = 0x02;
        /// Indicates that a groupby operations groups may overlap.
        /// If this is the case, an `explode` will yield more values than rows in original `df`,
        /// this breaks some assumptions
        const OVERLAPPING_GROUPS = 0x04;
    }
}

impl Default for StateFlags {
    fn default() -> Self {
        StateFlags::CACHE_WINDOW_EXPR
    }
}

impl StateFlags {
    fn init() -> Self {
        let verbose = std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("0") == "1";
        let mut flags: StateFlags = Default::default();
        if verbose {
            flags |= StateFlags::VERBOSE;
        }
        flags
    }
}

/// State/ cache that is maintained during the Execution of the physical plan.
pub struct ExecutionState {
    // cached by a `.cache` call and kept in memory for the duration of the plan.
    df_cache: Arc<Mutex<PlHashMap<usize, DataFrame>>>,
    // cache file reads until all branches got there file, then we delete it
    #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
    pub(crate) file_cache: FileCache,
    pub(super) schema_cache: Option<SchemaRef>,
    /// Used by Window Expression to prevent redundant grouping
    pub(super) group_tuples: GroupsProxyCache,
    /// Used by Window Expression to prevent redundant joins
    pub(super) join_tuples: JoinTuplesCache,
    // every join/union split gets an increment to distinguish between schema state
    pub(super) branch_idx: usize,
    pub(super) flags: StateFlags,
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
            }
        }
    }

    /// Partially clones and partially clears state
    pub(super) fn split(&self) -> Self {
        Self {
            df_cache: self.df_cache.clone(),
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            file_cache: self.file_cache.clone(),
            schema_cache: Default::default(),
            group_tuples: Default::default(),
            join_tuples: Default::default(),
            branch_idx: self.branch_idx,
            flags: self.flags,
            ext_contexts: self.ext_contexts.clone(),
            node_timer: self.node_timer.clone(),
        }
    }

    /// clones and partially clears state
    pub(super) fn clone(&self) -> Self {
        Self {
            df_cache: self.df_cache.clone(),
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            file_cache: self.file_cache.clone(),
            schema_cache: self.schema_cache.clone(),
            group_tuples: self.group_tuples.clone(),
            join_tuples: self.join_tuples.clone(),
            branch_idx: self.branch_idx,
            flags: self.flags,
            ext_contexts: self.ext_contexts.clone(),
            node_timer: self.node_timer.clone(),
        }
    }

    #[cfg(not(any(feature = "parquet", feature = "csv-file", feature = "ipc")))]
    pub(crate) fn with_finger_prints(finger_prints: Option<usize>) -> Self {
        Self::new()
    }
    #[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
    pub(crate) fn with_finger_prints(finger_prints: Option<Vec<FileFingerPrint>>) -> Self {
        Self {
            df_cache: Arc::new(Mutex::new(PlHashMap::default())),
            schema_cache: Default::default(),
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            file_cache: FileCache::new(finger_prints),
            group_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            join_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            branch_idx: 0,
            flags: StateFlags::init(),
            ext_contexts: Default::default(),
            node_timer: None,
        }
    }

    pub fn new() -> Self {
        let verbose = std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("0") == "1";
        let mut flags: StateFlags = Default::default();
        if verbose {
            flags |= StateFlags::VERBOSE;
        }
        Self {
            df_cache: Default::default(),
            schema_cache: Default::default(),
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            file_cache: FileCache::new(None),
            group_tuples: Default::default(),
            join_tuples: Default::default(),
            branch_idx: 0,
            flags: StateFlags::init(),
            ext_contexts: Default::default(),
            node_timer: None,
        }
    }
    pub(crate) fn set_schema(&mut self, schema: SchemaRef) {
        self.schema_cache = Some(schema);
    }

    /// Set the schema. Typically at the start of a projection.
    pub(crate) fn may_set_schema(&mut self, df: &DataFrame, exprs_len: usize) {
        if exprs_len > 1 && df.get_columns().len() > 10 {
            let schema = Arc::new(df.schema());
            self.set_schema(schema);
        }
    }

    /// Clear the schema. Typically at the end of a projection.
    pub(crate) fn clear_schema_cache(&mut self) {
        self.schema_cache = None;
    }

    /// Get the schema.
    pub(crate) fn get_schema(&self) -> Option<SchemaRef> {
        self.schema_cache.clone()
    }

    /// Check if we have DataFrame in cache
    pub(crate) fn cache_hit(&self, key: &usize) -> Option<DataFrame> {
        let guard = self.df_cache.lock().unwrap();
        guard.get(key).cloned()
    }

    /// Store DataFrame in cache.
    pub(crate) fn store_cache(&self, key: usize, df: DataFrame) {
        let mut guard = self.df_cache.lock().unwrap();
        guard.insert(key, df);
    }

    /// Clear the cache used by the Window expressions
    pub(crate) fn clear_expr_cache(&self) {
        {
            let mut lock = self.group_tuples.lock().unwrap();
            lock.clear();
        }
        let mut lock = self.join_tuples.lock().unwrap();
        lock.clear();
    }

    /// Indicates that window expression's [`GroupTuples`] may be cached.
    pub(super) fn cache_window(&self) -> bool {
        self.flags.contains(StateFlags::CACHE_WINDOW_EXPR)
    }

    /// Indicates that a groupby operations groups may overlap.
    /// If this is the case, an `explode` will yield more values than rows in original `df`,
    /// this breaks some assumptions
    pub(super) fn overlapping_groups(&self) -> bool {
        self.flags.contains(StateFlags::OVERLAPPING_GROUPS)
    }

    /// More verbose logging
    pub(super) fn verbose(&self) -> bool {
        self.flags.contains(StateFlags::VERBOSE)
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
