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
    match arr.data_type() {
        ArrowDataType::UInt8 => Series::UInt8(init_ca(arr, field)),
        ArrowDataType::UInt16 => Series::UInt16(init_ca(arr, field)),
        ArrowDataType::UInt32 => Series::UInt32(init_ca(arr, field)),
        ArrowDataType::UInt64 => Series::UInt64(init_ca(arr, field)),
        ArrowDataType::Int8 => Series::Int8(init_ca(arr, field)),
        ArrowDataType::Int16 => Series::Int16(init_ca(arr, field)),
        ArrowDataType::Int32 => Series::Int32(init_ca(arr, field)),
        ArrowDataType::Int64 => Series::Int64(init_ca(arr, field)),
        ArrowDataType::Float32 => Series::Float32(init_ca(arr, field)),
        ArrowDataType::Float64 => Series::Float64(init_ca(arr, field)),
        ArrowDataType::Utf8 => Series::Utf8(init_ca(arr, field)),
        ArrowDataType::Boolean => Series::Bool(init_ca(arr, field)),
        ArrowDataType::Date32(DateUnit::Day) => Series::Date32(init_ca(arr, field)),
        ArrowDataType::Date64(DateUnit::Millisecond) => Series::Date64(init_ca(arr, field)),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => {
            Series::DurationNanosecond(init_ca(arr, field))
        }
        ArrowDataType::Duration(TimeUnit::Microsecond) => {
            Series::DurationMicrosecond(init_ca(arr, field))
        }
        ArrowDataType::Duration(TimeUnit::Millisecond) => {
            Series::DurationMillisecond(init_ca(arr, field))
        }
        ArrowDataType::Duration(TimeUnit::Second) => Series::DurationSecond(init_ca(arr, field)),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => {
            Series::Time64Nanosecond(init_ca(arr, field))
        }
        ArrowDataType::Time64(TimeUnit::Microsecond) => {
            Series::Time64Microsecond(init_ca(arr, field))
        }
        ArrowDataType::Time32(TimeUnit::Millisecond) => {
            Series::Time32Millisecond(init_ca(arr, field))
        }
        ArrowDataType::Time32(TimeUnit::Second) => Series::Time32Second(init_ca(arr, field)),
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
            Series::TimestampNanosecond(init_ca(arr, field))
        }
        ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
            Series::TimestampMicrosecond(init_ca(arr, field))
        }
        ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
            Series::TimestampMillisecond(init_ca(arr, field))
        }
        ArrowDataType::Timestamp(TimeUnit::Second, _) => {
            Series::TimestampSecond(init_ca(arr, field))
        }
        ArrowDataType::List(_) => Series::List(init_ca(arr, field)),
        t => panic!(format!("Arrow datatype {:?} is not supported", t)),
    }
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
