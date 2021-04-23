use ahash::RandomState;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// State/ cache that is maintained during the Execution of the physical plan.
#[derive(Clone)]
pub struct ExecutionState {
    df_cache: Arc<Mutex<HashMap<String, DataFrame, RandomState>>>,
    gb_cache: Arc<Mutex<HashMap<String, GroupTuples, RandomState>>>,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self {
            df_cache: Arc::new(Mutex::new(HashMap::with_hasher(RandomState::default()))),
            gb_cache: Arc::new(Mutex::new(HashMap::with_hasher(RandomState::default()))),
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
}

impl Default for ExecutionState {
    fn default() -> Self {
        ExecutionState::new()
    }
}
