pub mod csv;
pub mod ipc;
pub mod json;
use crate::prelude::*;
use arrow::{
    csv::Reader as ArrowCsvReader, error::Result as ArrowResult, json::Reader as ArrowJsonReader,
    record_batch::RecordBatch,
};
use std::io::{Read, Seek, Write};
use std::sync::Arc;

pub trait SerReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self;

    /// Rechunk to a single chunk after Reading file.
    fn set_rechunk(self, rechunk: bool) -> Self;

    /// Take the SerReader and return a parsed DataFrame.
    fn finish(self) -> Result<DataFrame>;
}

pub trait SerWriter<'a, W>
where
    W: Write,
{
    fn new(writer: &'a mut W) -> Self;
    fn finish(self, df: &DataFrame) -> Result<()>;
}

pub trait ArrowReader {
    fn next(&mut self) -> ArrowResult<Option<RecordBatch>>;

    fn schema(&self) -> Arc<Schema>;
}

impl<R: Read> ArrowReader for ArrowCsvReader<R> {
    fn next(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next()
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

impl<R: Read> ArrowReader for ArrowJsonReader<R> {
    fn next(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next()
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

pub fn finish_reader<R: ArrowReader>(mut reader: R, rechunk: bool) -> Result<DataFrame> {
    let mut columns = reader
        .schema()
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            ArrowDataType::UInt32 => {
                Series::UInt32(UInt32Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Int32 => {
                Series::Int32(Int32Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Int64 => {
                Series::Int64(Int64Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Float32 => {
                Series::Float32(Float32Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Float64 => {
                Series::Float64(Float64Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Utf8 => Series::Utf8(Utf8Chunked::new_from_chunks(field.name(), vec![])),
            ArrowDataType::Boolean => {
                Series::Bool(BooleanChunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Date32(DateUnit::Millisecond) => {
                Series::Date32(Date32Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                Series::Date64(Date64Chunked::new_from_chunks(field.name(), vec![]))
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => Series::DurationNanosecond(
                DurationNanosecondChunked::new_from_chunks(field.name(), vec![]),
            ),
            ArrowDataType::Time64(TimeUnit::Nanosecond) => Series::Time64Nanosecond(
                Time64NanosecondChunked::new_from_chunks(field.name(), vec![]),
            ),
            _ => unimplemented!(),
        })
        .collect::<Vec<_>>();

    while let Some(batch) = reader.next()? {
        batch
            .columns()
            .into_iter()
            .zip(&mut columns)
            .map(|(arr, ser)| ser.append_array(arr.clone()))
            .collect::<Result<Vec<_>>>()?;
    }

    if rechunk {
        columns = columns
            .into_iter()
            .map(|s| {
                let s = s.rechunk(None)?;
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;
    }

    Ok(DataFrame {
        schema: reader.schema(),
        columns,
    })
}
