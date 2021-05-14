use ahash::RandomState;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type JoinTuplesCache = Arc<Mutex<HashMap<String, Vec<(u32, Option<u32>)>, RandomState>>>;
pub type GroupTuplesCache = Arc<Mutex<HashMap<String, GroupTuples, RandomState>>>;

/// State/ cache that is maintained during the Execution of the physical plan.
#[derive(Clone)]
pub struct ExecutionState {
    df_cache: Arc<Mutex<HashMap<String, DataFrame, RandomState>>>,
    /// Used by Window Expression to prevent redundant grouping
    pub(crate) group_tuples: GroupTuplesCache,
    /// Used by Window Expression to prevent redundant joins
    pub(crate) join_tuples: JoinTuplesCache,
    pub(crate) verbose: bool,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self {
            df_cache: Arc::new(Mutex::new(HashMap::with_hasher(RandomState::default()))),
            group_tuples: Arc::new(Mutex::new(HashMap::with_hasher(RandomState::default()))),
            join_tuples: Arc::new(Mutex::new(HashMap::with_hasher(RandomState::default()))),
            verbose: std::env::var("POLARS_VERBOSE").is_ok(),
        }
    }

    /// Check if we have DataFrame in cache
    pub fn cache_hit(&self, key: &str) -> Option<DataFrame> {
        let guard = self.df_cache.lock().unwrap();
        guard.get(key).cloned()
    }

    /// Store DataFrame in cache.
    pub fn store_cache(&self, key: String, df: DataFrame) {
        let mut guard = self.df_cache.lock().unwrap();
        guard.insert(key, df);
    }

    /// Clear the cache used by the Window expressions
    pub fn clear_expr_cache(&self) {
        {
            let mut lock = self.group_tuples.lock().unwrap();
            lock.clear();
        }
        let mut lock = self.join_tuples.lock().unwrap();
        lock.clear();
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
