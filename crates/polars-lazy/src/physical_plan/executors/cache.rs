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
        let read_lock = cache.read().unwrap();
        let mut cache_hit = true;

        let out = match read_lock.as_ref() {
            None => {
                drop(read_lock);
                let mut write_lock = cache.write().unwrap();
                let out = self.input.execute(state)?;
                cache_hit = false;
                *write_lock = Some(out.clone());
                out
            },
            Some(df) => df.clone(),
        };

        // decrement count on cache hits
        if cache_hit {
            self.count = self.count.saturating_sub(1);

            if self.count == 0 {
                // clear cache
                let mut write_lock = cache.write().unwrap();
                *write_lock = None;
            }
        }

        if state.verbose() {
            if cache_hit {
                println!("CACHE HIT: cache id: {:x}", self.id);
            } else {
                println!("CACHE SET: cache id: {:x}", self.id);
            }
        }

        Ok(out)
    }
}
