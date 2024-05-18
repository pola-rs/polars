use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::polars_err;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub(crate) struct FilterOperator {
    pub(crate) predicate: Arc<dyn PhysicalPipedExpr>,
}

impl Operator for FilterOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let s = self.predicate.evaluate(chunk, &context.execution_state)?;
        let mask = s.bool().map_err(|_| {
            polars_err!(
                ComputeError: "filter predicate must be of type `Boolean`, got `{}`", s.dtype()
            )
        })?;
        // the filter is sequential as they are already executed on different threads
        // we don't want to increase contention and data copies
        let df = chunk.data._filter_seq(mask)?;

        Ok(OperatorResult::Finished(chunk.with_data(df)))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
    fn fmt(&self) -> &str {
        "filter"
    }
}
