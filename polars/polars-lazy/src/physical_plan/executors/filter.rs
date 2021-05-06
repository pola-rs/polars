use crate::physical_plan::executors::POLARS_VERBOSE;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub struct FilterExec {
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) input: Box<dyn Executor>,
}

impl FilterExec {
    pub fn new(predicate: Arc<dyn PhysicalExpr>, input: Box<dyn Executor>) -> Self {
        Self { predicate, input }
    }
}

impl Executor for FilterExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        let s = self.predicate.evaluate(&df, state)?;
        let mask = s.bool().expect("filter predicate wasn't of type boolean");
        let df = df.filter(mask)?;
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("dataframe filtered");
        }
        Ok(df)
    }
}
