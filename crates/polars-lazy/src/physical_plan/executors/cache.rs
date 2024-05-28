use std::sync::atomic::Ordering;

use super::*;

pub struct CacheExec {
    pub input: Box<dyn Executor>,
    pub id: usize,
    pub count: u32,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let cache = state.get_df_cache(self.id, self.count);
        let mut cache_hit = true;
        let previous = cache.0.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous >= 0);

        let df = cache.1.get_or_try_init(|| {
            cache_hit = false;
            self.input.execute(state)
        })?;

        // Decrement count on cache hits.
        if cache_hit && previous == 0 {
            state.remove_df_cache(self.id);
        }

        if state.verbose() {
            if cache_hit {
                eprintln!("CACHE HIT: cache id: {:x}", self.id);
            } else {
                eprintln!("CACHE SET: cache id: {:x}", self.id);
            }
        }

        Ok(df.clone())
    }
}
