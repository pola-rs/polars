#![cfg_attr(docsrs, feature(doc_auto_cfg))]

#[cfg(feature = "avro")]
pub mod avro;
#[cfg(feature = "async")]
mod cloud;
#[cfg(any(feature = "csv-file", feature = "json"))]
pub mod csv;
#[cfg(feature = "parquet")]
pub mod export;
#[cfg(any(feature = "ipc", feature = "ipc_streaming"))]
pub mod ipc;
#[cfg(feature = "json")]
pub mod json;
#[cfg(feature = "json")]
pub mod ndjson_core;
#[cfg(feature = "async")]
pub use crate::cloud::glob as async_glob;

#[cfg(any(
    feature = "csv-file",
    feature = "parquet",
    feature = "ipc",
    feature = "json"
))]
pub mod mmap;
mod options;
#[cfg(feature = "parquet")]
pub mod parquet;
#[cfg(feature = "private")]
pub mod predicates;
#[cfg(not(feature = "private"))]
pub(crate) mod predicates;
pub mod prelude;
#[cfg(all(test, feature = "csv-file"))]
mod tests;
pub(crate) mod utils;

#[cfg(feature = "partition")]
pub mod partition;

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

#[allow(unused)] // remove when updating to rust nightly >= 1.61
use arrow::array::new_empty_array;
use arrow::error::Result as ArrowResult;
pub use options::*;
use polars_core::config::verbose;
use polars_core::frame::ArrowChunk;
use polars_core::prelude::*;

#[cfg(any(
    feature = "ipc",
    feature = "json",
    feature = "avro",
    feature = "ipc_streaming",
))]
use crate::predicates::PhysicalIoExpr;

pub trait SerReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self;

    /// Rechunk to a single chunk after Reading file.
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

pub trait WriterFactory {
    fn create_writer<W: Write + 'static>(&self, writer: W) -> Box<dyn SerWriter<W>>;
    fn extension(&self) -> PathBuf;
}

pub trait ArrowReader {
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>>;
}

#[cfg(any(
    feature = "ipc",
    feature = "json",
    feature = "avro",
    feature = "ipc_streaming",
))]
pub(crate) fn finish_reader<R: ArrowReader>(
    mut reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    arrow_schema: &ArrowSchema,
    row_count: Option<RowCount>,
) -> PolarsResult<DataFrame> {
    use polars_core::utils::accumulate_dataframes_vertical;

    let mut num_rows = 0;
    let mut parsed_dfs = Vec::with_capacity(1024);

    while let Some(batch) = reader.next_record_batch()? {
        let current_num_rows = num_rows as IdxSize;
        num_rows += batch.len();
        let mut df = DataFrame::try_from((batch, arrow_schema.fields.as_slice()))?;

        if let Some(rc) = &row_count {
            df.with_row_count_mut(&rc.name, Some(current_num_rows + rc.offset));
        }

        if let Some(predicate) = &predicate {
            let s = predicate.evaluate(&df)?;
            let mask = s.bool().expect("filter predicates was not of type boolean");
            df = df.filter(mask)?;
        }

        if let Some(n) = n_rows {
            if num_rows >= n {
                let len = n - parsed_dfs
                    .iter()
                    .map(|df: &DataFrame| df.height())
                    .sum::<usize>();
                if verbose() {
                    eprintln!("sliced off {} rows of the 'DataFrame'. These lines were read because they were in a single chunk.", df.height() - n)
                }
                parsed_dfs.push(df.slice(0, len));
                break;
            }
        }
        parsed_dfs.push(df);
    }

    let df = {
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
            accumulate_dataframes_vertical(parsed_dfs)?
        }
    };

    match rechunk {
        true => Ok(df.agg_chunks()),
        false => Ok(df),
    }
}

/// Check if the path is a cloud url.
pub fn is_cloud_url<P: AsRef<Path>>(p: P) -> bool {
    p.as_ref().starts_with("s3://")
        || p.as_ref().starts_with("file://")
        || p.as_ref().starts_with("gcs://")
}
