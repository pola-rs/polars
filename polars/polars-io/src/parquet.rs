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
use crate::{PhysicalIoExpr, ScanAggregation};
use arrow::datatypes::PhysicalType;
use arrow::io::parquet::write::{array_to_pages, DynIter, Encoding};
use arrow::io::parquet::{read, write};
use polars_core::prelude::*;
use rayon::prelude::*;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Read Apache parquet format into a DataFrame.
pub struct ParquetReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    stop_after_n_rows: Option<usize>,
}

impl<R> ParquetReader<R>
where
    R: Read + Seek,
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

        let reader = read::RecordReader::try_new(
            &mut self.reader,
            projection.map(|x| x.to_vec()),
            self.stop_after_n_rows,
            None,
            None,
        )?;

        finish_reader(
            reader,
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

    pub fn schema(mut self) -> Result<Schema> {
        let metadata = read::read_metadata(&mut self.reader)?;

        let schema = read::get_schema(&metadata)?;
        Ok(schema.into())
    }
}

impl<R: Read + Seek> ArrowReader for read::RecordReader<R> {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new((&*self.schema().clone()).into())
    }
}

impl<R> SerReader<R> for ParquetReader<R>
where
    R: Read + Seek,
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

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;

        let reader = read::RecordReader::try_new(
            &mut self.reader,
            None,
            self.stop_after_n_rows,
            None,
            None,
        )?;
        finish_reader(reader, rechunk, self.stop_after_n_rows, None, None)
    }
}

/// Write a DataFrame to parquet format
///
/// # Example
///
///
pub struct ParquetWriter<W> {
    writer: W,
    compression: write::Compression,
}

pub use write::Compression;

impl<W> ParquetWriter<W>
where
    W: Write + Seek,
{
    /// Create a new writer
    pub fn new(writer: W) -> Self
    where
        W: Write + Seek,
    {
        ParquetWriter {
            writer,
            compression: write::Compression::Snappy,
        }
    }

    /// Set the compression used. Defaults to `Snappy`.
    pub fn with_compression(mut self, compression: write::Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Write the given DataFrame in the the writer `W`.
    pub fn finish(mut self, df: &DataFrame) -> Result<()> {
        let fields = df.schema().to_arrow().fields().clone();
        let rb_iter = df.iter_record_batches();

        let options = write::WriteOptions {
            write_statistics: false,
            compression: self.compression,
            version: write::Version::V2,
        };
        let schema = ArrowSchema::new(fields);
        let parquet_schema = write::to_parquet_schema(&schema)?;
        let encodings = schema
            .fields()
            .iter()
            .map(|field| match field.data_type().to_physical_type() {
                // delta encoding
                // Not yet supported by pyarrow
                // PhysicalType::LargeUtf8 => Encoding::DeltaLengthByteArray,
                // dictionaries are kept dict-encoded
                PhysicalType::Dictionary(_) => Encoding::RleDictionary,
                // remaining is plain
                _ => Encoding::Plain,
            })
            .collect::<Vec<_>>();

        // clone is needed because parquet schema is moved into `write_file`
        let parquet_schema_iter = parquet_schema.clone();
        let row_group_iter = rb_iter.map(|batch| {
            let columns = batch
                .columns()
                .par_iter()
                .zip(parquet_schema_iter.columns().par_iter())
                .zip(encodings.par_iter())
                .map(|((array, descriptor), encoding)| {
                    let array = array.clone();

                    let pages =
                        array_to_pages(array, descriptor.clone(), options, *encoding).unwrap();
                    pages.collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let out = write::DynIter::new(columns.into_iter().map(|column| {
                // one parquet page per array.
                // we could use `array.slice()` to split it based on some number of rows.
                Ok(DynIter::new(column.into_iter()))
            }));
            ArrowResult::Ok(out)
        });

        write::write_file(
            &mut self.writer,
            row_group_iter,
            &schema,
            parquet_schema,
            options,
            None,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
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
    #[cfg(all(feature = "dtype-datetime", feature = "parquet"))]
    fn test_parquet_datetime_round_trip() -> Result<()> {
        use std::io::{Cursor, Seek, SeekFrom};

        let mut f = Cursor::new(vec![]);

        let mut df = df![
            "datetime" => [Some(191845729i64), Some(89107598), None, Some(3158971092)]
        ]?;

        df.may_apply("datetime", |s| s.cast(&DataType::Datetime))?;

        ParquetWriter::new(&mut f).finish(&df)?;

        f.seek(SeekFrom::Start(0))?;

        let read = ParquetReader::new(f).finish()?;
        assert!(read.frame_equal_missing(&df));
        Ok(())
    }
}
