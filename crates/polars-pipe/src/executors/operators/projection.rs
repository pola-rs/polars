use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use smartstring::alias::String as SmartString;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub(crate) struct FastProjectionOperator {
    columns: Arc<[SmartString]>,
    input_schema: SchemaRef,
}

impl FastProjectionOperator {
    pub(crate) fn new(columns: Arc<[SmartString]>, input_schema: SchemaRef) -> Self {
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
    pub(crate) cse_exprs: Option<HstackOperator>,
}

impl Operator for ProjectionOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // add temporary cse column to the chunk
        let cse_owned_chunk;
        let chunk = if let Some(hstack) = &mut self.cse_exprs {
            let OperatorResult::Finished(out) = hstack.execute(context, chunk)? else {
                unreachable!()
            };
            cse_owned_chunk = out;
            &cse_owned_chunk
        } else {
            chunk
        };

        let mut has_literals = false;
        let mut has_empty = false;
        let mut projected = self
            .exprs
            .iter()
            .map(|e| {
                #[allow(unused_mut)]
                let mut s = e.evaluate(chunk, context.execution_state.as_any())?;

                has_literals |= s.len() == 1;
                has_empty |= s.len() == 0;

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
        if self.cse_exprs.is_some() {
            "projection[cse]"
        } else {
            "projection"
        }
    }
}

#[derive(Clone)]
pub(crate) struct HstackOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
    pub(crate) input_schema: SchemaRef,
    pub(crate) cse_exprs: Option<Box<Self>>,
    // add columns without any checks
    // this is needed for cse, as the temporary columns
    // may have a different size
    pub(crate) unchecked: bool,
}

impl Operator for HstackOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // add temporary cse column to the chunk
        let width = chunk.data.width();
        let cse_owned_chunk;
        let chunk = if let Some(hstack) = &mut self.cse_exprs {
            let OperatorResult::Finished(out) = hstack.execute(context, chunk)? else {
                unreachable!()
            };
            cse_owned_chunk = out;
            &cse_owned_chunk
        } else {
            chunk
        };

        let projected = self
            .exprs
            .iter()
            .map(|e| e.evaluate(chunk, context.execution_state.as_any()))
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut df = DataFrame::new_no_checks(chunk.data.get_columns()[..width].to_vec());

        let schema = &*self.input_schema;
        if self.unchecked {
            unsafe { df.get_columns_mut().extend(projected) }
        } else {
            df._add_columns(projected, schema)?;
        }

        let chunk = chunk.with_data(df);
        Ok(OperatorResult::Finished(chunk))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
    fn fmt(&self) -> &str {
        if self.cse_exprs.is_some() {
            "hstack[cse]"
        } else {
            "hstack"
        }
    }
}
