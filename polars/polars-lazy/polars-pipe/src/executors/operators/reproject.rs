use polars_core::frame::DataFrame;
use polars_core::prelude::SchemaRef;
use polars_core::schema::Schema;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext, PolarsResult};

/// An operator that will ensure we keep the schema order
pub(crate) struct ReProjectOperator {
    schema: SchemaRef,
    operator: Box<dyn Operator>,
    // cache the positions
    positions: Vec<usize>,
}

impl ReProjectOperator {
    pub(crate) fn new(schema: SchemaRef, operator: Box<dyn Operator>) -> Self {
        ReProjectOperator {
            schema,
            operator,
            positions: vec![],
        }
    }
}

pub(crate) fn reproject_chunk(
    chunk: &mut DataChunk,
    positions: &mut Vec<usize>,
    schema: &Schema,
) -> PolarsResult<()> {
    let out = if positions.is_empty() {
        let out = chunk.data.select(schema.iter_names())?;
        *positions = out
            .get_columns()
            .iter()
            .map(|s| schema.get_full(s.name()).unwrap().0)
            .collect();
        out
    } else {
        let columns = chunk.data.get_columns();
        let cols = positions.iter().map(|i| columns[*i].clone()).collect();
        DataFrame::new_no_checks(cols)
    };
    *chunk = chunk.with_data(out);
    Ok(())
}

impl Operator for ReProjectOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let (mut chunk, finished) = match self.operator.execute(context, chunk)? {
            OperatorResult::Finished(chunk) => (chunk, true),
            OperatorResult::HaveMoreOutPut(chunk) => (chunk, false),
            OperatorResult::NeedsNewData => return Ok(OperatorResult::NeedsNewData),
        };
        reproject_chunk(&mut chunk, &mut self.positions, self.schema.as_ref())?;
        Ok(if finished {
            OperatorResult::Finished(chunk)
        } else {
            OperatorResult::HaveMoreOutPut(chunk)
        })
    }

    fn split(&self, thread_no: usize) -> Box<dyn Operator> {
        let operator = self.operator.split(thread_no);
        Box::new(Self {
            schema: self.schema.clone(),
            positions: self.positions.clone(),
            operator,
        })
    }

    fn fmt(&self) -> &str {
        "re-project-operator"
    }
}
