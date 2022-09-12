use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct CacheExec {
    pub input: Box<dyn Executor>,
    pub id: usize,
    pub count: usize,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        if self.count > 0 {
            if let Some(df) = state.cache_hit(&self.id) {
                if state.verbose() {
                    println!("CACHE HIT: cache id: {}", self.id);
                }
                self.count -= 0;
                return Ok(df);
            }
        }

        // cache miss
        let df = self.input.execute(state)?;
        state.store_cache(self.id, df.clone());
        if state.verbose() {
            println!("CACHE SET: cache id: {}", self.id);
        }
        Ok(df)
    }
}
