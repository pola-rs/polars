use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub(crate) struct FastProjectionOperator {
    columns: Arc<[Arc<str>]>,
    input_schema: SchemaRef,
}

impl FastProjectionOperator {
    pub(crate) fn new(columns: Arc<[Arc<str>]>, input_schema: SchemaRef) -> Self {
        Self {
            columns,
            input_schema,
        }
    }
}

impl Operator for FastProjectionOperator {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let chunk = chunk.with_data(
            chunk
                .data
                .select_with_schema_unchecked(self.columns.as_ref(), &self.input_schema)?,
        );
        Ok(OperatorResult::Finished(chunk))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
    fn fmt(&self) -> &str {
        "fast_projection"
    }
}

#[derive(Clone)]
pub(crate) struct ProjectionOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
}

impl Operator for ProjectionOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let mut has_literals = false;
        let mut has_empty = false;
        let mut projected = self
            .exprs
            .iter()
            .map(|e| {
                let s = e.evaluate(chunk, context.execution_state.as_any())?;
                if s.len() == 1 {
                    has_literals = true;
                }
                if s.len() == 0 {
                    has_empty = true;
                }
                Ok(s)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        if has_empty {
            for s in &mut projected {
                *s = s.clear();
            }
        } else if has_literals {
            let height = projected.iter().map(|s| s.len()).max().unwrap();
            for s in &mut projected {
                let len = s.len();
                if len == 1 && len != height {
                    *s = s.new_from_index(0, height)
                }
            }
        }

        let chunk = chunk.with_data(DataFrame::new_no_checks(projected));
        Ok(OperatorResult::Finished(chunk))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
    fn fmt(&self) -> &str {
        "projection"
    }
}

#[derive(Clone)]
pub(crate) struct HstackOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
    pub(crate) input_schema: SchemaRef,
}

impl Operator for HstackOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let projected = self
            .exprs
            .iter()
            .map(|e| e.evaluate(chunk, context.execution_state.as_any()))
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut df = chunk.data.clone();
        let schema = &*self.input_schema;
        df._add_columns(projected, schema)?;

        let chunk = chunk.with_data(df);
        Ok(OperatorResult::Finished(chunk))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
    fn fmt(&self) -> &str {
        "hstack"
    }
}
