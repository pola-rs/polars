pub mod csv;
pub mod ipc;
pub mod json;
#[cfg(feature = "parquet")]
#[doc(cfg(feature = "parquet"))]
pub mod parquet;
use crate::prelude::*;
use arrow::{
    csv::Reader as ArrowCsvReader, error::ArrowError, error::Result as ArrowResult,
    json::Reader as ArrowJsonReader, record_batch::RecordBatch,
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

    /// Continue with next batch when a ParserError is encountered.
    fn with_ignore_parser_error(self) -> Self;

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

pub fn finish_reader<R: ArrowReader>(
    mut reader: R,
    rechunk: bool,
    ignore_parser_error: bool,
    stop_after_n_rows: Option<usize>,
) -> Result<DataFrame> {
    fn init_ca<T>(field: &Field) -> ChunkedArray<T>
    where
        T: PolarsDataType,
    {
        ChunkedArray::new_from_chunks(field.name(), vec![])
    }

    let mut columns = reader
        .schema()
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            ArrowDataType::UInt8 => Series::UInt8(init_ca(field)),
            ArrowDataType::UInt16 => Series::UInt16(init_ca(field)),
            ArrowDataType::UInt32 => Series::UInt32(init_ca(field)),
            ArrowDataType::UInt64 => Series::UInt64(init_ca(field)),
            ArrowDataType::Int8 => Series::Int8(init_ca(field)),
            ArrowDataType::Int16 => Series::Int16(init_ca(field)),
            ArrowDataType::Int32 => Series::Int32(init_ca(field)),
            ArrowDataType::Int64 => Series::Int64(init_ca(field)),
            ArrowDataType::Float32 => Series::Float32(init_ca(field)),
            ArrowDataType::Float64 => Series::Float64(init_ca(field)),
            ArrowDataType::Utf8 => Series::Utf8(init_ca(field)),
            ArrowDataType::Boolean => Series::Bool(init_ca(field)),
            ArrowDataType::Date32(DateUnit::Millisecond) => Series::Date32(init_ca(field)),
            ArrowDataType::Date64(DateUnit::Millisecond) => Series::Date64(init_ca(field)),
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                Series::DurationNanosecond(init_ca(field))
            }
            ArrowDataType::Duration(TimeUnit::Microsecond) => {
                Series::DurationMicrosecond(init_ca(field))
            }
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                Series::DurationMillisecond(init_ca(field))
            }
            ArrowDataType::Duration(TimeUnit::Second) => Series::DurationSecond(init_ca(field)),
            ArrowDataType::Time64(TimeUnit::Nanosecond) => Series::Time64Nanosecond(init_ca(field)),
            ArrowDataType::Time64(TimeUnit::Microsecond) => {
                Series::Time64Microsecond(init_ca(field))
            }
            ArrowDataType::Time32(TimeUnit::Millisecond) => {
                Series::Time32Millisecond(init_ca(field))
            }
            ArrowDataType::Time32(TimeUnit::Second) => Series::Time32Second(init_ca(field)),
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Series::TimestampNanosecond(init_ca(field))
            }
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
                Series::TimestampMicrosecond(init_ca(field))
            }
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
                Series::TimestampMillisecond(init_ca(field))
            }
            ArrowDataType::Timestamp(TimeUnit::Second, _) => {
                Series::TimestampSecond(init_ca(field))
            }
            ArrowDataType::List(_) => Series::List(init_ca(field)),
            t => panic!(format!("Arrow datatype {:?} is not supported", t)),
        })
        .collect::<Vec<_>>();

    let mut n_rows = 0;
    loop {
        let batch = match reader.next_record_batch() {
            Err(ArrowError::ParseError(s)) => {
                if ignore_parser_error {
                    continue;
                } else {
                    return Err(PolarsError::ArrowError(ArrowError::ParseError(s)));
                }
            }
            Err(e) => return Err(PolarsError::ArrowError(e)),
            Ok(None) => break,
            Ok(Some(batch)) => batch,
        };
        n_rows += batch.num_rows();
        batch
            .columns()
            .into_iter()
            .zip(&mut columns)
            .map(|(arr, ser)| ser.append_array(arr.clone()))
            .collect::<Result<Vec<_>>>()?;
        if let Some(n) = stop_after_n_rows {
            if n_rows > n {
                break;
            }
        }
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

    Ok(DataFrame { columns })
}
