use std::sync::atomic::Ordering;

use super::*;

pub struct CacheExec {
    pub input: Option<Box<dyn Executor>>,
    pub id: usize,
    /// `(cache_hits_before_drop - 1)`
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
                let out = cache.1.get().expect("prefilled").clone();
                let previous = cache.0.fetch_sub(1, Ordering::Relaxed);
                if previous == 0 {
                    if state.verbose() {
                        eprintln!("CACHE DROP: cache id: {:x}", self.id);
                    }
                    state.remove_df_cache(self.id);
                }

                Ok(out)
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
    pub caches: PlIndexMap<usize, Box<CacheExec>>,
    pub phys_plan: Box<dyn Executor>,
}

impl Executor for CachePrefiller {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        if state.verbose() {
            eprintln!("PREFILL CACHES")
        }
        // Ensure we traverse in discovery order. This will ensure that caches aren't dependent on each
        // other.
        for cache in self.caches.values_mut() {
            let mut state = state.split();
            state.branch_idx += 1;
            let _df = cache.execute(&mut state)?;
        }
        if state.verbose() {
            eprintln!("EXECUTE PHYS PLAN")
        }
        self.phys_plan.execute(state)
    }
}
