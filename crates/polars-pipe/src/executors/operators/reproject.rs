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
        // use the chunk schema to cache
        // the positions for subsequent calls
        let chunk_schema = chunk.data.schema();

        let out = chunk
            .data
            .select_with_schema_unchecked(schema.iter_names(), &chunk_schema)?;

        *positions = out
            .get_columns()
            .iter()
            .map(|s| chunk_schema.get_full(s.name()).unwrap().0)
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

#[cfg(test)]
mod test {
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn test_reproject_chunk() {
        let df = df![
            "a" => [1, 2],
            "b" => [1, 2],
            "c" => [1, 2],
            "d" => [1, 2],
        ]
        .unwrap();

        let mut chunk1 = DataChunk::new(0, df.clone());
        let mut chunk2 = DataChunk::new(1, df);

        let mut positions = vec![];

        let mut out_schema = Schema::new();
        out_schema.with_column("c".into(), DataType::Int32);
        out_schema.with_column("b".into(), DataType::Int32);
        out_schema.with_column("d".into(), DataType::Int32);
        out_schema.with_column("a".into(), DataType::Int32);

        reproject_chunk(&mut chunk1, &mut positions, &out_schema).unwrap();
        // second call cached the positions
        reproject_chunk(&mut chunk2, &mut positions, &out_schema).unwrap();
        assert_eq!(&chunk1.data.schema(), &out_schema);
        assert_eq!(&chunk2.data.schema(), &out_schema);
    }
}
