use super::*;

pub struct CacheExec {
    pub input: Box<dyn Executor>,
    pub id: usize,
    pub count: usize,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        // skip cache and always re-execute
        if self.count == 0 {
            if state.verbose() {
                println!("CACHE IGNORE: cache id: {:x}", self.id);
            }
            return self.input.execute(state);
        }

        let cache = state.get_df_cache(self.id);
        let mut cache_hit = true;

        let df = cache.get_or_try_init(|| {
            cache_hit = false;
            self.input.execute(state)
        })?;

        // decrement count on cache hits
        if cache_hit {
            self.count -= 1;
        }

        if state.verbose() {
            if cache_hit {
                println!("CACHE HIT: cache id: {:x}", self.id);
            } else {
                println!("CACHE SET: cache id: {:x}", self.id);
            }
        }

        Ok(df.clone())
    }
}
