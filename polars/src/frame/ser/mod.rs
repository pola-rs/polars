pub mod csv;
pub(crate) mod fork;
pub mod ipc;
pub mod json;
#[cfg(feature = "parquet")]
#[doc(cfg(feature = "parquet"))]
pub mod parquet;

#[cfg(feature = "lazy")]
use crate::lazy::prelude::PhysicalExpr;
use crate::prelude::*;
use crate::series::implementations::Wrap;
use crate::utils::accumulate_dataframes_vertical;
use arrow::array::ArrayRef;
use arrow::{
    csv::Reader as ArrowCsvReader, error::Result as ArrowResult, json::Reader as ArrowJsonReader,
    record_batch::RecordBatch,
};
use std::io::{Read, Seek, Write};
use std::sync::Arc;

#[cfg(not(feature = "lazy"))]
pub trait PhysicalExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series>;
}

pub trait SerReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self;

    /// Rechunk to a single chunk after Reading file.
    fn set_rechunk(self, _rechunk: bool) -> Self
    where
        Self: std::marker::Sized,
    {
        self
    }

    /// Take the SerReader and return a parsed DataFrame.
    fn finish(self) -> Result<DataFrame>;
}

pub trait SerWriter<'a, W>
where
    W: Write,
{
    fn new(writer: &'a mut W) -> Self;
    fn finish(self, df: &mut DataFrame) -> Result<()>;
}

pub trait ArrowReader {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>>;

    fn schema(&self) -> Arc<Schema>;
}

impl<R: Read> ArrowReader for ArrowCsvReader<R> {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

impl<R: Read> ArrowReader for ArrowJsonReader<R> {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next()
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

fn init_ca<T>(arr: &ArrayRef, field: &Field) -> ChunkedArray<T>
where
    T: PolarsDataType,
{
    ChunkedArray::new_from_chunks(field.name(), vec![arr.clone()])
}

fn arr_to_series(arr: &ArrayRef, field: &Field) -> Series {
    let s: Wrap<_> = (field.name().as_str(), arr.clone()).into();
    Series(s.0)
}

pub fn finish_reader<R: ArrowReader>(
    mut reader: R,
    rechunk: bool,
    stop_after_n_rows: Option<usize>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
) -> Result<DataFrame> {
    let mut n_rows = 0;
    let mut parsed_dfs = Vec::with_capacity(1024);

    while let Some(batch) = reader.next_record_batch()? {
        n_rows += batch.num_rows();

        let columns = batch
            .columns()
            .iter()
            .zip(reader.schema().fields())
            .map(|(arr, field)| arr_to_series(arr, field))
            .collect();

        let mut df = DataFrame::new_no_checks(columns);

        if let Some(predicate) = &predicate {
            let s = predicate.evaluate(&df)?;
            let mask = s.bool().expect("filter predicates was not of type boolean");
            df = df.filter(mask)?;
        }
        parsed_dfs.push(df);
        if let Some(n) = stop_after_n_rows {
            if n_rows >= n {
                break;
            }
        }
    }
    let df = accumulate_dataframes_vertical(parsed_dfs)?;
    match rechunk {
        true => Ok(df.agg_chunks()),
        false => Ok(df),
    }
}
