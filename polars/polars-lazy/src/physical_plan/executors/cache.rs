use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct CacheExec {
    pub id: usize,
    pub input: Box<dyn Executor>,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        if let Some(df) = state.cache_hit(&self.id) {
            if state.verbose() {
                println!("CACHE HIT: cache id: {}", self.id);
            }
            return Ok(df);
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
