use parking_lot::{Mutex, RwLock};
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use std::ops::Deref;

pub type JoinTuplesCache = Arc<Mutex<PlHashMap<String, Vec<(IdxSize, Option<IdxSize>)>>>>;
pub type GroupsProxyCache = Arc<Mutex<PlHashMap<String, GroupsProxy>>>;

/// State/ cache that is maintained during the Execution of the physical plan.
#[derive(Clone)]
pub struct ExecutionState {
    df_cache: Arc<Mutex<PlHashMap<String, DataFrame>>>,
    pub schema_cache: Arc<RwLock<Option<SchemaRef>>>,
    /// Used by Window Expression to prevent redundant grouping
    pub(crate) group_tuples: GroupsProxyCache,
    /// Used by Window Expression to prevent redundant joins
    pub(crate) join_tuples: JoinTuplesCache,
    pub(crate) verbose: bool,
    pub(crate) cache_window: bool,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self {
            df_cache: Arc::new(Mutex::new(PlHashMap::default())),
            schema_cache: Arc::new(RwLock::new(None)),
            group_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            join_tuples: Arc::new(Mutex::new(PlHashMap::default())),
            verbose: std::env::var("POLARS_VERBOSE").is_ok(),
            cache_window: true,
        }
    }
    pub(crate) fn set_schema(&self, schema: SchemaRef) {
        let mut opt = self.schema_cache.write();
        *opt = Some(schema)
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
        let mut lock = self.schema_cache.write();
        *lock = None;
    }

    /// Get the schema.
    pub(crate) fn get_schema(&self) -> Option<SchemaRef> {
        let opt = self.schema_cache.read();
        opt.deref().clone()
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
