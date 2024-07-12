use std::io::{Read, Write};
use std::sync::Arc;

use arrow::array::new_empty_array;
use arrow::record_batch::RecordBatch;
use polars_core::prelude::*;

use crate::options::RowIndex;
#[cfg(any(feature = "ipc", feature = "avro", feature = "ipc_streaming",))]
use crate::predicates::PhysicalIoExpr;

pub trait SerReader<R>
where
    R: Read,
{
    /// Create a new instance of the `[SerReader]`
    fn new(reader: R) -> Self;

    /// Make sure that all columns are contiguous in memory by
    /// aggregating the chunks into a single array.
    #[must_use]
    fn set_rechunk(self, _rechunk: bool) -> Self
    where
        Self: Sized,
    {
        self
    }

    /// Take the SerReader and return a parsed DataFrame.
    fn finish(self) -> PolarsResult<DataFrame>;
}

pub trait SerWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self
    where
        Self: Sized;
    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()>;
}

pub trait WriteDataFrameToFile {
    fn write_df_to_file<W: std::io::Write>(&self, df: DataFrame, file: W) -> PolarsResult<()>;
}

pub trait ArrowReader {
    fn next_record_batch(&mut self) -> PolarsResult<Option<RecordBatch>>;
}

#[cfg(any(feature = "ipc", feature = "avro", feature = "ipc_streaming",))]
pub(crate) fn finish_reader<R: ArrowReader>(
    mut reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    arrow_schema: &ArrowSchema,
    row_index: Option<RowIndex>,
) -> PolarsResult<DataFrame> {
    use polars_core::utils::accumulate_dataframes_vertical_unchecked;

    let mut num_rows = 0;
    let mut parsed_dfs = Vec::with_capacity(1024);

    while let Some(batch) = reader.next_record_batch()? {
        let current_num_rows = num_rows as IdxSize;
        num_rows += batch.len();
        let mut df = DataFrame::try_from((batch, arrow_schema.fields.as_slice()))?;

        if let Some(rc) = &row_index {
            df.with_row_index_mut(&rc.name, Some(current_num_rows + rc.offset));
        }

        if let Some(predicate) = &predicate {
            let s = predicate.evaluate_io(&df)?;
            let mask = s.bool().expect("filter predicates was not of type boolean");
            df = df.filter(mask)?;
        }

        if let Some(n) = n_rows {
            if num_rows >= n {
                let len = n - parsed_dfs
                    .iter()
                    .map(|df: &DataFrame| df.height())
                    .sum::<usize>();
                if polars_core::config::verbose() {
                    eprintln!("sliced off {} rows of the 'DataFrame'. These lines were read because they were in a single chunk.", df.height().saturating_sub(n))
                }
                parsed_dfs.push(df.slice(0, len));
                break;
            }
        }
        parsed_dfs.push(df);
    }

    let mut df = {
        if parsed_dfs.is_empty() {
            // Create an empty dataframe with the correct data types
            let empty_cols = arrow_schema
                .fields
                .iter()
                .map(|fld| {
                    Series::try_from((fld.name.as_str(), new_empty_array(fld.data_type.clone())))
                })
                .collect::<PolarsResult<_>>()?;
            DataFrame::new(empty_cols)?
        } else {
            // If there are any rows, accumulate them into a df
            accumulate_dataframes_vertical_unchecked(parsed_dfs)
        }
    };

    if rechunk {
        df.as_single_chunk_par();
    }
    Ok(df)
}

pub(crate) fn schema_to_arrow_checked(
    schema: &Schema,
    compat_level: CompatLevel,
    _file_name: &str,
) -> PolarsResult<ArrowSchema> {
    let fields = schema.iter_fields().map(|field| {
        #[cfg(feature = "object")]
        polars_ensure!(!matches!(field.data_type(), DataType::Object(_, _)), ComputeError: "cannot write 'Object' datatype to {}", _file_name);
        Ok(field.data_type().to_arrow_field(field.name().as_str(), compat_level))
    }).collect::<PolarsResult<Vec<_>>>()?;
    Ok(ArrowSchema::from(fields))
}
