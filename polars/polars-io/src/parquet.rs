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
use arrow::error::ArrowError;
use arrow::io::parquet::write::{array_to_pages, DynIter, DynStreamingIterator, Encoding};
use arrow::io::parquet::{
    read,
    write::{self, *},
};
use polars_core::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Read Apache parquet format into a DataFrame.
pub struct ParquetReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
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
            self.n_rows,
            None,
            None,
        )?;

        finish_reader(reader, rechunk, self.n_rows, predicate, aggregate)
    }

    /// Stop parsing when `n` rows are parsed. By settings this parameter the csv will be parsed
    /// sequentially.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
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
            n_rows: None,
            columns: None,
            projection: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = read::read_metadata(&mut self.reader)?;
        let schema = read::schema::get_schema(&metadata)?;

        if let Some(cols) = self.columns {
            let mut prj = Vec::with_capacity(cols.len());
            for col in cols.iter() {
                let i = schema.index_of(col)?;
                prj.push(i);
            }

            self.projection = Some(prj);
        }

        let reader = read::RecordReader::try_new(
            &mut self.reader,
            self.projection,
            self.n_rows,
            None,
            None,
        )?;
        finish_reader(reader, rechunk, self.n_rows, None, None)
    }
}

struct Bla {
    columns: VecDeque<CompressedPage>,
    current: Option<CompressedPage>,
}

impl Bla {
    pub fn new(columns: VecDeque<CompressedPage>) -> Self {
        Self {
            columns,
            current: None,
        }
    }
}

impl FallibleStreamingIterator for Bla {
    type Item = CompressedPage;
    type Error = ArrowError;

    fn advance(&mut self) -> ArrowResult<()> {
        self.current = self.columns.pop_front();
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
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

pub use write::Compression as ParquetCompression;

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
                    let encoded_pages =
                        array_to_pages(array.as_ref(), descriptor.clone(), options, *encoding)?;
                    encoded_pages
                        .map(|page| {
                            compress(page?, vec![], options.compression).map_err(|x| x.into())
                        })
                        .collect::<ArrowResult<VecDeque<_>>>()
                })
                .collect::<ArrowResult<Vec<VecDeque<CompressedPage>>>>()?;

            let row_group = DynIter::new(
                columns
                    .into_iter()
                    .map(|column| Ok(DynStreamingIterator::new(Bla::new(column)))),
            );
            ArrowResult::Ok(row_group)
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
    use std::io::Cursor;

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

    #[test]
    fn test_read_parquet_with_projection() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        ParquetWriter::new(&mut buf)
            .finish(&df)
            .expect("parquet writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = ParquetReader::new(buf)
            .with_projection(Some(vec![1, 2]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }

    #[test]
    fn test_read_parquet_with_columns() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        ParquetWriter::new(&mut buf)
            .finish(&df)
            .expect("parquet writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = ParquetReader::new(buf)
            .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }
}
