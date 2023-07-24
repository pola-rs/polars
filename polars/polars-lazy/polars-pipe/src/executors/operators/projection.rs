use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
#[cfg(feature = "cse")]
use polars_plan::utils::rename_cse_tmp_series;

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
    #[cfg(feature = "cse")]
    pub(crate) cse_exprs: Option<HstackOperator>,
}

impl Operator for ProjectionOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // add temporary cse column to the chunk
        #[cfg(feature = "cse")]
        let cse_owned_chunk;
        #[cfg(feature = "cse")]
        let chunk = if let Some(hstack) = &mut self.cse_exprs {
            let OperatorResult::Finished(out) = hstack.execute(context, chunk)? else { unreachable!() };
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

                // correct the cse name
                #[cfg(feature = "cse")]
                if self.cse_exprs.is_some() {
                    rename_cse_tmp_series(&mut s);
                }

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
        #[cfg(feature = "cse")]
        {
            if self.cse_exprs.is_some() {
                "projection[cse]"
            } else {
                "projection"
            }
        }
        #[cfg(not(feature = "cse"))]
        "projection"
    }
}

#[derive(Clone)]
pub(crate) struct HstackOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
    pub(crate) input_schema: SchemaRef,
    #[cfg(feature = "cse")]
    pub(crate) cse_exprs: Option<Box<Self>>,
}

impl Operator for HstackOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // add temporary cse column to the chunk
        #[cfg(feature = "cse")]
        let width = chunk.data.width();
        #[cfg(feature = "cse")]
        let cse_owned_chunk;
        #[cfg(feature = "cse")]
        let chunk = if let Some(hstack) = &mut self.cse_exprs {
            let OperatorResult::Finished(out) = hstack.execute(context, chunk)? else { unreachable!() };
            cse_owned_chunk = out;
            &cse_owned_chunk
        } else {
            chunk
        };

        let projected = self
            .exprs
            .iter()
            .map(|e| {
                #[allow(unused_mut)]
                let mut res = e.evaluate(chunk, context.execution_state.as_any());

                #[cfg(feature = "cse")]
                if self.cse_exprs.is_some() {
                    res = res.map(|mut s| {
                        rename_cse_tmp_series(&mut s);
                        s
                    })
                }
                res
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        #[cfg(feature = "cse")]
        let mut df = DataFrame::new_no_checks(chunk.data.get_columns()[..width].to_vec());
        #[cfg(not(feature = "cse"))]
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
        #[cfg(feature = "cse")]
        {
            if self.cse_exprs.is_some() {
                "hstack[cse]"
            } else {
                "hstack"
            }
        }
        #[cfg(not(feature = "cse"))]
        "hstack"
    }
}
