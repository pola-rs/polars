#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
use super::file_cache::FileCache;
#[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
use crate::prelude::file_caching::FileFingerPrint;
use bitflags::bitflags;
use parking_lot::Mutex;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::frame::hash_join::JoinOptIds;
use polars_core::prelude::*;

pub type JoinTuplesCache = Arc<Mutex<PlHashMap<String, JoinOptIds>>>;
pub type GroupsProxyCache = Arc<Mutex<PlHashMap<String, GroupsProxy>>>;

bitflags! {
    pub(super) struct StateFlags: u8 {
        const VERBOSE = 0x01;
        const CACHE_WINDOW_EXPR = 0x02;
        const FILTER_NODE = 0x03;
    }
}

impl Default for StateFlags {
    fn default() -> Self {
        StateFlags::CACHE_WINDOW_EXPR
    }
}

impl StateFlags {
    fn init() -> Self {
        let verbose = std::env::var("POLARS_VERBOSE").is_ok();
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
    df_cache: Arc<Mutex<PlHashMap<String, DataFrame>>>,
    // cache file reads until all branches got there file, then we delete it
    #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
    pub(super) file_cache: FileCache,
    pub(super) schema_cache: Option<SchemaRef>,
    /// Used by Window Expression to prevent redundant grouping
    pub(super) group_tuples: GroupsProxyCache,
    /// Used by Window Expression to prevent redundant joins
    pub(super) join_tuples: JoinTuplesCache,
    // every join/union split gets an increment to distinguish between schema state
    pub(super) branch_idx: usize,
    pub(super) flags: StateFlags,
}

impl ExecutionState {
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
        }
    }

    pub fn new() -> Self {
        let verbose = std::env::var("POLARS_VERBOSE").is_ok();
        let mut flags: StateFlags = Default::default();
        if verbose {
            flags |= StateFlags::VERBOSE;
        }
        Self {
            df_cache: Arc::new(Mutex::new(PlHashMap::default())),
            schema_cache: Default::default(),
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            file_cache: FileCache::new(None),
            group_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            join_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            branch_idx: 0,
            flags: StateFlags::init(),
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
    pub(crate) fn cache_hit(&self, key: &str) -> Option<DataFrame> {
        let guard = self.df_cache.lock();
        guard.get(key).cloned()
    }

    /// Store DataFrame in cache.
    pub(crate) fn store_cache(&self, key: String, df: DataFrame) {
        let mut guard = self.df_cache.lock();
        guard.insert(key, df);
    }

    /// Clear the cache used by the Window expressions
    pub(crate) fn clear_expr_cache(&self) {
        {
            let mut lock = self.group_tuples.lock();
            lock.clear();
        }
        let mut lock = self.join_tuples.lock();
        lock.clear();
    }

    pub(super) fn cache_window(&self) -> bool {
        self.flags.contains(StateFlags::CACHE_WINDOW_EXPR)
    }

    pub(super) fn verbose(&self) -> bool {
        self.flags.contains(StateFlags::VERBOSE)
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
