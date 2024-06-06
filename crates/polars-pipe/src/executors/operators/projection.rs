use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_plan::prelude::ProjectionOptions;
use smartstring::alias::String as SmartString;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub(crate) struct SimpleProjectionOperator {
    columns: Arc<[SmartString]>,
    input_schema: SchemaRef,
}

impl SimpleProjectionOperator {
    pub(crate) fn new(columns: Arc<[SmartString]>, input_schema: SchemaRef) -> Self {
        Self {
            columns,
            input_schema,
        }
    }
}

impl Operator for SimpleProjectionOperator {
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
    pub(crate) options: ProjectionOptions,
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
                #[allow(unused_mut)]
                let mut s = e.evaluate(chunk, &context.execution_state)?;

                has_literals |= s.len() == 1;
                has_empty |= s.len() == 0;

                Ok(s)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        if has_empty {
            for s in &mut projected {
                *s = s.clear();
            }
        } else if has_literals && self.options.should_broadcast {
            let height = projected.iter().map(|s| s.len()).max().unwrap();
            for s in &mut projected {
                let len = s.len();
                if len == 1 && len != height {
                    *s = s.new_from_index(0, height)
                }
            }
        }

        let chunk = chunk.with_data(unsafe { DataFrame::new_no_checks(projected) });
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
    pub(crate) options: ProjectionOptions,
}

impl Operator for HstackOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // add temporary cse column to the chunk
        let width = chunk.data.width();
        let projected = self
            .exprs
            .iter()
            .map(|e| e.evaluate(chunk, &context.execution_state))
            .collect::<PolarsResult<Vec<_>>>()?;

        let columns = chunk.data.get_columns()[..width].to_vec();
        let mut df = unsafe { DataFrame::new_no_checks(columns) };

        let schema = &*self.input_schema;
        if self.options.should_broadcast {
            df._add_columns(projected, schema)?;
        } else {
            debug_assert!(
                projected
                    .iter()
                    .all(|column| column.name().starts_with("__POLARS_CSER_0x")),
                "non-broadcasting hstack should only be used for CSE columns"
            );
            // Safety: this case only appears as a result of CSE
            // optimization, and the usage there produces new, unique
            // column names. It is immediately followed by a
            // projection which pulls out the possibly mismatching
            // column lengths.
            unsafe { df.get_columns_mut().extend(projected) }
        }

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
