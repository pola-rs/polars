use std::any::Any;

use arrow::array::BinaryArray;
use polars_core::prelude::sort::_broadcast_bools;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_row::decode::decode_rows_from_binary;
use polars_row::{RowEncodingContext, RowEncodingOptions};

use self::row_encode::get_row_encoding_context;
use super::*;
use crate::operators::{
    DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult, Source, SourceResult,
};
const POLARS_SORT_COLUMN: &str = "__POLARS_SORT_COLUMN";

fn get_sort_fields(
    sort_idx: &[usize],
    sort_options: &SortMultipleOptions,
) -> Vec<RowEncodingOptions> {
    let mut descending = sort_options.descending.clone();
    let mut nulls_last = sort_options.nulls_last.clone();

    _broadcast_bools(sort_idx.len(), &mut descending);
    _broadcast_bools(sort_idx.len(), &mut nulls_last);

    descending
        .into_iter()
        .zip(nulls_last)
        .map(|(descending, nulls_last)| RowEncodingOptions::new_sorted(descending, nulls_last))
        .collect()
}

fn sort_by_idx<V: Clone>(values: &[V], idx: &[usize]) -> Vec<V> {
    assert_eq!(values.len(), idx.len());

    let mut tmp = values
        .iter()
        .cloned()
        .zip(idx.iter().copied())
        .collect::<Vec<_>>();
    tmp.sort_unstable_by_key(|k| k.1);
    tmp.into_iter().map(|k| k.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn finalize_dataframe(
    df: &mut DataFrame,
    sort_idx: &[usize],
    sort_options: &SortMultipleOptions,
    sort_dtypes: Option<&[ArrowDataType]>,
    rows: &mut Vec<&'static [u8]>,
    sort_opts: &[RowEncodingOptions],
    sort_dicts: &[Option<RowEncodingContext>],
    schema: &Schema,
) {
    // pop the encoded sort column
    // SAFETY: We only pop a value
    let encoded = unsafe { df.get_columns_mut() }.pop().unwrap();

    // we decode the row-encoded binary column
    // this will be decoded into multiple columns
    // these are the columns we sorted by
    // those need to be inserted at the `sort_idx` position
    // in the `DataFrame`.
    let sort_dtypes = sort_dtypes.expect("should be set if 'can_decode'");

    let encoded = encoded.binary_offset().unwrap();
    assert_eq!(encoded.chunks().len(), 1);
    let arr = encoded.downcast_iter().next().unwrap();

    // SAFETY:
    // temporary extend lifetime
    // this is safe as the lifetime in rows stays bound to this scope
    let arrays = unsafe {
        let arr = std::mem::transmute::<&'_ BinaryArray<i64>, &'static BinaryArray<i64>>(arr);
        decode_rows_from_binary(arr, sort_opts, sort_dicts, sort_dtypes, rows)
    };
    rows.clear();

    let arrays = sort_by_idx(&arrays, sort_idx);
    let mut sort_idx = sort_idx.to_vec();
    sort_idx.sort_unstable();

    for (&sort_idx, arr) in sort_idx.iter().zip(arrays) {
        let (name, logical_dtype) = schema.get_at_index(sort_idx).unwrap();
        assert_eq!(
            logical_dtype.to_physical(),
            DataType::from_arrow_dtype(arr.dtype())
        );
        let col = unsafe {
            Series::from_chunks_and_dtype_unchecked(name.clone(), vec![arr], logical_dtype)
        }
        .into_column();

        // SAFETY: col has the same length as the df height because it was popped from df.
        unsafe { df.get_columns_mut() }.insert(sort_idx, col);
        df.clear_schema();
    }

    // SAFETY: We just change the sorted flag.
    let first_sort_col = &mut unsafe { df.get_columns_mut() }[sort_idx[0]];
    let flag = if sort_options.descending[0] {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    first_sort_col.set_sorted_flag(flag)
}

/// This struct will dispatch all sorting to `SortSink`
/// But before it does that it will encode the column that we
/// must sort by and set that row-encoded column as last
/// column in the data chunks.
///
/// Once the sorting is finished it adapts the result so that
/// the encoded column is removed
pub struct SortSinkMultiple {
    output_schema: SchemaRef,
    sort_idx: Arc<[usize]>,
    sort_sink: Box<dyn Sink>,
    slice: Option<(i64, usize)>,
    sort_options: SortMultipleOptions,
    // Needed for encoding
    sort_opts: Arc<[RowEncodingOptions]>,
    sort_dicts: Arc<[Option<RowEncodingContext>]>,
    sort_dtypes: Option<Arc<[DataType]>>,
    // amortize allocs
    sort_column: Vec<ArrayRef>,
}

impl SortSinkMultiple {
    pub(crate) fn new(
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
        output_schema: SchemaRef,
        sort_idx: Vec<usize>,
    ) -> PolarsResult<Self> {
        let mut schema = (*output_schema).clone();

        let sort_ctxts = sort_idx
            .iter()
            .map(|i| {
                let (_, dtype) = schema.get_at_index(*i).unwrap();
                get_row_encoding_context(dtype, true)
            })
            .collect::<Vec<_>>();

        polars_ensure!(sort_idx.iter().collect::<PlHashSet::<_>>().len() == sort_idx.len(), ComputeError: "only supports sorting by unique columns");

        let mut dtypes = vec![DataType::Null; sort_idx.len()];

        // we remove columns by index, but then the indices aren't correct anymore
        // so we do it in the proper order and keep track of the indices removed
        let mut sorted_sort_idx = sort_idx.iter().copied().enumerate().collect::<Vec<_>>();
        // Sort by `sort_idx`.
        sorted_sort_idx.sort_unstable_by_key(|k| k.1);
        // remove the sort indices as we will encode them into the sort binary
        for (iterator_i, (original_idx, sort_i)) in sorted_sort_idx.iter().enumerate() {
            dtypes[*original_idx] = schema.shift_remove_index(*sort_i - iterator_i).unwrap().1;
        }
        let sort_dtypes = Some(dtypes.into());
        schema.with_column(POLARS_SORT_COLUMN.into(), DataType::BinaryOffset);
        let sort_fields = get_sort_fields(&sort_idx, &sort_options);

        // don't set descending and nulls last as this
        // will be solved by the row encoding
        let sort_sink = Box::new(SortSink::new(
            // we will set the last column as sort column
            schema.len() - 1,
            slice,
            sort_options
                .clone()
                .with_order_descending(false)
                .with_nulls_last(false)
                .with_maintain_order(false),
            Arc::new(schema),
        ));

        Ok(SortSinkMultiple {
            sort_sink,
            slice,
            sort_options,
            sort_idx: Arc::from(sort_idx),
            sort_opts: Arc::from(sort_fields),
            sort_dicts: Arc::from(sort_ctxts),
            sort_dtypes,
            sort_column: vec![],
            output_schema,
        })
    }

    fn encode(&mut self, chunk: &mut DataChunk) -> PolarsResult<()> {
        let df = &mut chunk.data;

        self.sort_column.clear();

        for i in self.sort_idx.iter() {
            let s = &df.get_columns()[*i].as_materialized_series();
            let arr = s.to_physical_repr().rechunk().chunks()[0].to_boxed();
            self.sort_column.push(arr);
        }

        // we remove columns by index, but then the aren't correct anymore
        // so we do it in the proper order and keep track of the indices removed
        let mut sorted_sort_idx = self.sort_idx.to_vec();
        sorted_sort_idx.sort_unstable();

        // SAFETY: We do not adjust the names or lengths or columns.
        let cols = unsafe { df.get_columns_mut() };

        sorted_sort_idx
            .into_iter()
            .enumerate()
            .for_each(|(i, sort_idx)| {
                // shifts all columns right from removed one to the left so
                // therefore we subtract `i` as the shifted count
                let _ = cols.remove(sort_idx - i);
            });

        df.clear_schema();
        let name = PlSmallStr::from_static(POLARS_SORT_COLUMN);
        let column = if chunk.data.height() == 0 && chunk.data.width() > 0 {
            Column::new_empty(name, &DataType::BinaryOffset)
        } else {
            let rows_encoded = polars_row::convert_columns(
                self.sort_column[0].len(), // @NOTE: does not work for ZFS
                &self.sort_column,
                &self.sort_opts,
                &self.sort_dicts,
            );
            let series = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    name,
                    vec![Box::new(rows_encoded.into_array())],
                    &DataType::BinaryOffset,
                )
            };
            debug_assert_eq!(series.chunks().len(), 1);
            series.into()
        };

        // SAFETY: length is correct
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
            sort_opts: self.sort_opts.clone(),
            slice: self.slice,
            sort_options: self.sort_options.clone(),
            sort_dicts: self.sort_dicts.clone(),
            sort_column: vec![],
            sort_dtypes: self.sort_dtypes.clone(),
            output_schema: self.output_schema.clone(),
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let out = self.sort_sink.finalize(context)?;

        let sort_dtypes = self.sort_dtypes.take().map(|arr| {
            arr.iter()
                .map(|dt| dt.to_physical().to_arrow(CompatLevel::newest()))
                .collect::<Vec<_>>()
        });

        // we must adapt the finalized sink result so that the sort encoded column is dropped
        match out {
            FinalizedSink::Finished(mut df) => {
                finalize_dataframe(
                    &mut df,
                    self.sort_idx.as_ref(),
                    &self.sort_options,
                    sort_dtypes.as_deref(),
                    &mut vec![],
                    self.sort_opts.as_ref(),
                    self.sort_dicts.as_ref(),
                    &self.output_schema,
                );
                Ok(FinalizedSink::Finished(df))
            },
            FinalizedSink::Source(source) => Ok(FinalizedSink::Source(Box::new(DropEncoded {
                source,
                sort_idx: self.sort_idx.clone(),
                sort_options: self.sort_options.clone(),
                sort_dtypes,
                rows: vec![],
                sort_opts: self.sort_opts.clone(),
                sort_dicts: self.sort_dicts.clone(),
                output_schema: self.output_schema.clone(),
            }))),
            // SortSink should not produce this branch
            FinalizedSink::Operator => unreachable!(),
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
    sort_options: SortMultipleOptions,
    sort_dtypes: Option<Vec<ArrowDataType>>,
    rows: Vec<&'static [u8]>,
    sort_opts: Arc<[RowEncodingOptions]>,
    sort_dicts: Arc<[Option<RowEncodingContext>]>,
    output_schema: SchemaRef,
}

impl Source for DropEncoded {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        let mut result = self.source.get_batches(context);
        if let Ok(SourceResult::GotMoreData(data)) = &mut result {
            for chunk in data {
                finalize_dataframe(
                    &mut chunk.data,
                    self.sort_idx.as_ref(),
                    &self.sort_options,
                    self.sort_dtypes.as_deref(),
                    &mut self.rows,
                    self.sort_opts.as_ref(),
                    self.sort_dicts.as_ref(),
                    &self.output_schema,
                )
            }
        };
        result
    }

    fn fmt(&self) -> &str {
        "sort_multiple_source"
    }
}
