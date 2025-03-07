use std::sync::atomic::Ordering;

use super::*;

pub struct CacheExec {
    pub input: Option<Box<dyn Executor>>,
    pub id: usize,
    pub count: u32,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        match &mut self.input {
            // Cache node
            None => {
                if state.verbose() {
                    eprintln!("CACHE HIT: cache id: {:x}", self.id);
                }
                let cache = state.get_df_cache(self.id, self.count);
                let previous = cache.0.fetch_sub(1, Ordering::Relaxed);
                debug_assert!(previous >= 0);

                Ok(cache.1.get().unwrap().clone())
            },
            // Cache Prefill node
            Some(input) => {
                if state.verbose() {
                    eprintln!("CACHE SET: cache id: {:x}", self.id);
                }
                let df = input.execute(state)?;
                let cache = state.get_df_cache(self.id, self.count);
                cache.1.set(df).expect("should be empty");
                Ok(DataFrame::empty())
            },
        }
    }
}

pub struct CachePrefiller {
    pub caches: PlIndexMap<usize, Box<dyn Executor>>,
    pub phys_plan: Box<dyn Executor>,
}

impl Executor for CachePrefiller {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        for cache in self.caches.values_mut().rev() {
            let _df = cache.execute(state)?;
        }

        self.phys_plan.execute(state)
    }
}
