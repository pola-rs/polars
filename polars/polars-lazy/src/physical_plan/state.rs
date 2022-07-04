#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
use super::file_cache::FileCache;
#[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
use crate::prelude::file_caching::FileFingerPrint;
use parking_lot::{Mutex, RwLock};
use polars_core::frame::groupby::GroupsProxy;
use polars_core::frame::hash_join::JoinOptIds;
use polars_core::prelude::*;

pub type JoinTuplesCache = Arc<Mutex<PlHashMap<String, JoinOptIds>>>;
pub type GroupsProxyCache = Arc<Mutex<PlHashMap<String, GroupsProxy>>>;

/// State/ cache that is maintained during the Execution of the physical plan.
#[derive(Clone)]
pub struct ExecutionState {
    // cached by a `.cache` call and kept in memory for the duration of the plan.
    df_cache: Arc<Mutex<PlHashMap<String, DataFrame>>>,
    // cache file reads until all branches got there file, then we delete it
    #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
    pub(crate) file_cache: FileCache,
    pub(crate) schema_cache: Arc<RwLock<Vec<Option<SchemaRef>>>>,
    /// Used by Window Expression to prevent redundant grouping
    pub(crate) group_tuples: GroupsProxyCache,
    /// Used by Window Expression to prevent redundant joins
    pub(crate) join_tuples: JoinTuplesCache,
    pub(crate) verbose: bool,
    pub(crate) cache_window: bool,
    // every join/union split gets an increment to distinguish between schema state
    pub(crate) join_branch: usize,
}

impl ExecutionState {
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
            verbose: std::env::var("POLARS_VERBOSE").is_ok(),
            cache_window: true,
            join_branch: 0,
        }
    }

    pub fn new() -> Self {
        Self {
            df_cache: Arc::new(Mutex::new(PlHashMap::default())),
            schema_cache: Default::default(),
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            file_cache: FileCache::new(None),
            group_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            join_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            verbose: std::env::var("POLARS_VERBOSE").is_ok(),
            cache_window: true,
            join_branch: 0,
        }
    }
    pub(crate) fn set_schema(&self, schema: SchemaRef) {
        let mut opt = self.schema_cache.write();
        if opt.len() <= self.join_branch {
            opt.resize(self.join_branch + 1, None)
        }
        opt[self.join_branch] = Some(schema);
    }

    /// Set the schema. Typically at the start of a projection.
    pub(crate) fn may_set_schema(&self, df: &DataFrame, exprs_len: usize) {
        if exprs_len > 1 && df.get_columns().len() > 10 {
            let schema = Arc::new(df.schema());
            self.set_schema(schema);
        }
    }

    /// Clear the schema. Typically at the end of a projection.
    pub(crate) fn clear_schema_cache(&self) {
        let read_lock = self.schema_cache.read();
        if read_lock.len() > self.join_branch {
            drop(read_lock);
            let mut write_lock = self.schema_cache.write();
            write_lock[self.join_branch] = None
        }
    }

    /// Get the schema.
    pub(crate) fn get_schema(&self) -> Option<SchemaRef> {
        let opt = self.schema_cache.read();
        if opt.len() <= self.join_branch {
            None
        } else {
            opt[self.join_branch].clone()
        }
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
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
