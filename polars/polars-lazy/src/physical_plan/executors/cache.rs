use crate::physical_plan::executors::POLARS_VERBOSE;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub struct CacheExec {
    pub key: String,
    pub input: Box<dyn Executor>,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        if let Some(df) = state.cache_hit(&self.key) {
            return Ok(df);
        }

        // cache miss
        let df = self.input.execute(state)?;
        state.store_cache(std::mem::take(&mut self.key), df.clone());
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("cache set {:?}", self.key);
        }
        Ok(df)
    }
}
