use std::any::Any;

use polars_core::prelude::sort::_broadcast_descending;
use polars_core::prelude::sort::arg_sort_multiple::_get_rows_encoded_compat_array;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_plan::prelude::*;
use polars_row::SortField;

use super::*;
use crate::operators::{
    DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult, Source, SourceResult,
};
const POLARS_SORT_COLUMN: &str = "__POLARS_SORT_COLUMN";

fn get_sort_fields(sort_idx: &[usize], sort_args: &SortArguments) -> Vec<SortField> {
    let mut descending = sort_args.descending.clone();
    _broadcast_descending(sort_idx.len(), &mut descending);
    descending
        .into_iter()
        .map(|descending| SortField {
            descending,
            nulls_last: sort_args.nulls_last,
        })
        .collect()
}

fn finalize_dataframe(df: &mut DataFrame, sort_idx: &[usize], sort_args: &SortArguments) {
    unsafe {
        let cols = df.get_columns_mut();
        // pop the encoded sort column
        let _ = cols.pop();

        let first_sort_col = &mut cols[sort_idx[0]];
        let flag = if sort_args.descending[0] {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        first_sort_col.set_sorted_flag(flag)
    }
}

/// This struct will dispatch all sorting to `SortSink`
/// But before it does that it will encode the column that we
/// must sort by and set that row-encoded column as last
/// column in the data chunks.
///
/// Once the sorting is finished it adapts the result so that
/// the encoded column is removed
pub struct SortSinkMultiple {
    sort_idx: Arc<[usize]>,
    sort_sink: Box<dyn Sink>,
    sort_args: SortArguments,
    // Needed for encoding
    sort_fields: Arc<[SortField]>,
    // amortize allocs
    sort_column: Vec<ArrayRef>,
}

impl SortSinkMultiple {
    pub(crate) fn new(sort_args: SortArguments, schema: &Schema, sort_idx: Vec<usize>) -> Self {
        let mut schema = schema.clone();
        schema.with_column(POLARS_SORT_COLUMN.into(), DataType::Binary);
        let sort_fields = get_sort_fields(&sort_idx, &sort_args);

        // don't set descending and nulls last as this
        // will be solved by the row encoding
        let sort_sink = Box::new(SortSink::new(
            // we will set the last column as sort column
            schema.len() - 1,
            SortArguments {
                descending: vec![false],
                nulls_last: false,
                slice: sort_args.slice,
            },
            Arc::new(schema),
        ));

        SortSinkMultiple {
            sort_sink,
            sort_args,
            sort_idx: Arc::from(sort_idx),
            sort_fields: Arc::from(sort_fields),
            sort_column: vec![],
        }
    }

    fn encode(&mut self, chunk: &mut DataChunk) -> PolarsResult<()> {
        let df = &chunk.data;
        let cols = df.get_columns();

        self.sort_column.clear();

        for i in self.sort_idx.iter() {
            let s = &cols[*i];
            let arr = _get_rows_encoded_compat_array(s)?;
            self.sort_column.push(arr);
        }
        let rows_encoded = polars_row::convert_columns(&self.sort_column, &self.sort_fields);
        let column = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                POLARS_SORT_COLUMN,
                vec![Box::new(rows_encoded.into_array())],
                &DataType::Binary,
            )
        };

        // Safety: length is correct
        unsafe { chunk.data.with_column_unchecked(column) };
        Ok(())
    }
}

impl Sink for SortSinkMultiple {
    fn sink(
        &mut self,
        context: &PExecutionContext,
        mut chunk: DataChunk,
    ) -> PolarsResult<SinkResult> {
        self.encode(&mut chunk)?;
        self.sort_sink.sink(context, chunk)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        self.sort_sink.combine(&mut *other.sort_sink)
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        let sort_sink = self.sort_sink.split(thread_no);
        Box::new(Self {
            sort_idx: self.sort_idx.clone(),
            sort_sink,
            sort_fields: self.sort_fields.clone(),
            sort_args: self.sort_args.clone(),
            sort_column: vec![],
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let out = self.sort_sink.finalize(context)?;

        // we must adapt the finalized sink result so that the sort encoded column is dropped
        match out {
            FinalizedSink::Finished(mut df) => {
                finalize_dataframe(&mut df, self.sort_idx.as_ref(), &self.sort_args);
                Ok(FinalizedSink::Finished(df))
            }
            FinalizedSink::Source(source) => Ok(FinalizedSink::Source(Box::new(DropEncoded {
                source,
                sort_idx: self.sort_idx.clone(),
                sort_args: std::mem::take(&mut self.sort_args),
            }))),
            // SortSink should not produce this branch
            FinalizedSink::Operator(_) => unreachable!(),
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "sort_multiple"
    }
}

struct DropEncoded {
    source: Box<dyn Source>,
    sort_idx: Arc<[usize]>,
    sort_args: SortArguments,
}

impl Source for DropEncoded {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        let mut result = self.source.get_batches(context);
        if let Ok(SourceResult::GotMoreData(data)) = &mut result {
            for chunk in data {
                finalize_dataframe(&mut chunk.data, self.sort_idx.as_ref(), &self.sort_args)
            }
        };
        result
    }

    fn fmt(&self) -> &str {
        "sort_multiple_source"
    }
}
