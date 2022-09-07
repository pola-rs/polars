use std::borrow::Cow;

use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

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
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run FilterExec")
            }
        }
        let df = self.input.execute(state)?;
        let s = self.predicate.evaluate(&df, state)?;
        let mask = s.bool().expect("filter predicate wasn't of type boolean");

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".filter({})", &self.predicate.as_ref()))
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                let df = df.filter(mask)?;
                if state.verbose() {
                    eprintln!("dataframe filtered");
                }
                Ok(df)
            },
            profile_name,
        )
    }
}
