//! # Reading Apache parquet files.
//!
//! ## Example
//!
//! ```rust
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> Result<DataFrame> {
//!     let r = File::open("some_file.parquet").unwrap();
//!     let reader = ParquetReader::new(r);
//!     reader.finish()
//! }
//! ```
//!
use super::{finish_reader, ArrowReader, ArrowResult, RecordBatch};
use crate::prelude::*;
use crate::utils::to_arrow_compatible_df;
use crate::{PhysicalIoExpr, ScanAggregation};
use arrow::{compute::cast, record_batch::RecordBatchReader};
use parquet_lib::file::reader::{FileReader, SerializedFileReader};
pub use parquet_lib::file::serialized_reader::SliceableCursor;
use parquet_lib::{
    arrow::{
        arrow_reader::ParquetRecordBatchReader, arrow_writer::ArrowWriter as ParquetArrowWriter,
        ArrowReader as ParquetArrowReader, ParquetFileArrowReader,
    },
    file::writer::TryClone,
};
use polars_core::prelude::*;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

fn set_batch_size(max_rows: usize, stop_after_n_rows: Option<usize>) -> usize {
    let mut batch_size = max_rows;
    if let Some(n) = stop_after_n_rows {
        // set batch size exactly to n_rows
        batch_size = std::cmp::min(batch_size, n);
        batch_size = std::cmp::max(batch_size, n);
    }
    batch_size
}

/// Read Apache parquet format into a DataFrame.
pub struct ParquetReader<R> {
    reader: R,
    rechunk: bool,
    stop_after_n_rows: Option<usize>,
}

impl<R> ParquetReader<R>
where
    R: 'static + Read + Seek + parquet_lib::file::reader::ChunkReader,
{
    #[cfg(feature = "lazy")]
    // todo! hoist to lazy crate
    pub fn finish_with_scan_ops(
        mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        aggregate: Option<&[ScanAggregation]>,
        projection: Option<&[usize]>,
    ) -> Result<DataFrame> {
        let rechunk = self.rechunk;

        let file_reader = Arc::new(SerializedFileReader::new(self.reader)?);
        let rows_in_file = file_reader.metadata().file_metadata().num_rows() as usize;

        if let Some(stop_after_n_rows) = self.stop_after_n_rows {
            if stop_after_n_rows > rows_in_file {
                self.stop_after_n_rows = Some(rows_in_file)
            }
        }

        let batch_size = match predicate {
            Some(_) => 512 * 1024,
            None => rows_in_file,
        };
        let batch_size = set_batch_size(batch_size, self.stop_after_n_rows);

        let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
        let record_reader = match projection {
            Some(projection) => {
                arrow_reader.get_record_reader_by_columns(projection.iter().copied(), batch_size)
            }
            None => arrow_reader.get_record_reader(batch_size),
        }?;
        finish_reader(
            record_reader,
            rechunk,
            self.stop_after_n_rows,
            predicate,
            aggregate,
        )
    }

    /// Stop parsing when `n` rows are parsed. By settings this parameter the csv will be parsed
    /// sequentially.
    pub fn with_stop_after_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.stop_after_n_rows = num_rows;
        self
    }

    pub fn schema(self) -> Result<Schema> {
        let file_reader = Arc::new(SerializedFileReader::new(self.reader)?);
        let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
        let schema = arrow_reader.get_schema()?;
        Ok(schema.into())
    }
}

impl ArrowReader for ParquetRecordBatchReader {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new((&*<Self as RecordBatchReader>::schema(self)).into())
    }
}

impl<R> ParquetReader<R> {}

impl<R> SerReader<R> for ParquetReader<R>
where
    R: 'static + Read + Seek + parquet_lib::file::reader::ChunkReader,
{
    fn new(reader: R) -> Self {
        ParquetReader {
            reader,
            rechunk: false,
            stop_after_n_rows: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let file_reader = Arc::new(SerializedFileReader::new(self.reader)?);
        let n_rows = file_reader.metadata().file_metadata().num_rows() as usize;
        let batch_size = set_batch_size(n_rows, self.stop_after_n_rows);
        let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
        let record_reader = arrow_reader.get_record_reader(batch_size)?;
        finish_reader(record_reader, rechunk, self.stop_after_n_rows, None, None)
    }
}

/// Write a DataFrame to parquet format
///
/// # Example
///
///
pub struct ParquetWriter<W> {
    writer: W,
}

impl<W> ParquetWriter<W>
where
    W: 'static + Write + Seek + TryClone,
{
    /// Create a new writer
    pub fn new(writer: W) -> Self
    where
        W: 'static + Write + Seek + TryClone,
    {
        ParquetWriter { writer }
    }

    /// Write the given DataFrame in the the writer `W`.
    pub fn finish(self, df: &DataFrame) -> Result<()> {
        let df = to_arrow_compatible_df(df);
        let mut fields = df.schema().to_arrow().fields().clone();

        // date64 is not supported by parquet and will be be truncated to date32
        // We coerce these to timestamp(ms)
        let date64_columns = df
            .get_columns()
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s.dtype() {
                DataType::Date64 => {
                    fields[i] = ArrowField::new(
                        s.name(),
                        ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                        s.null_count() > 0,
                    );
                    Some(i)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let column_names = df
            .get_columns()
            .iter()
            .map(|s| s.name().to_string())
            .collect::<Vec<_>>();

        let iter = df.iter_record_batches().map(|rb| {
            if !date64_columns.is_empty() {
                let mut columns = rb.columns().to_vec();
                for i in &date64_columns {
                    let array = cast(&columns[*i], &ArrowDataType::Int64).unwrap();
                    let array = cast(
                        &array,
                        &ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                    )
                    .unwrap();
                    columns[*i] = array;
                }
                RecordBatch::try_from_iter(column_names.iter().zip(columns)).unwrap()
            } else {
                rb
            }
        });

        let mut parquet_writer =
            ParquetArrowWriter::try_new(self.writer, Arc::new(ArrowSchema::new(fields)), None)?;

        for batch in iter {
            parquet_writer.write(&batch)?
        }
        let _ = parquet_writer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use parquet_lib::file::writer::InMemoryWriteableCursor;
    use polars_core::{df, prelude::*};
    use std::fs::File;

    #[test]
    fn test_parquet() {
        // In CI: This test will be skipped because the file does not exist.
        if let Ok(r) = File::open("data/simple.parquet") {
            let reader = ParquetReader::new(r);
            let df = reader.finish().unwrap();
            assert_eq!(df.get_column_names(), ["a", "b"]);
            assert_eq!(df.shape(), (3, 2));
        }
    }

    #[test]
    #[cfg(all(feature = "dtype-date64", feature = "parquet"))]
    fn test_parquet_date64_round_trip() -> Result<()> {
        let f: InMemoryWriteableCursor = Default::default();

        let mut df = df![
            "date64" => [Some(191845729i64), Some(89107598), None, Some(3158971092)]
        ]?;

        df.may_apply("date64", |s| s.cast::<Date64Type>())?;

        ParquetWriter::new(f.clone()).finish(&df)?;
        let data = f.data();

        let f = SliceableCursor::new(data);

        let read = ParquetReader::new(f).finish()?;
        assert!(read.frame_equal_missing(&df));
        Ok(())
    }
}
