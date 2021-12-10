//! # (De)serializing Arrows IPC format.
//!
//! Arrow IPC is a [binary format format](https://arrow.apache.org/docs/python/ipc.html).
//! It is the recommended way to serialize and deserialize Polars DataFrames as this is most true
//! to the data schema.
//!
//! ## Example
//!
//! ```rust
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::io::Cursor;
//!
//!
//! let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
//! let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
//! let mut df = DataFrame::new(vec![s0, s1]).unwrap();
//!
//! // Create an in memory file handler.
//! // Vec<u8>: Read + Write
//! // Cursor<T>: Seek
//!
//! let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
//!
//! // write to the in memory buffer
//! IpcWriter::new(&mut buf).finish(&mut df).expect("ipc writer");
//!
//! // reset the buffers index after writing to the beginning of the buffer
//! buf.set_position(0);
//!
//! // read the buffer into a DataFrame
//! let df_read = IpcReader::new(buf).finish().unwrap();
//! assert!(df.frame_equal(&df_read));
//! ```
use super::{finish_reader, ArrowReader, ArrowResult, RecordBatch};
use crate::prelude::*;
use crate::{PhysicalIoExpr, ScanAggregation};
use ahash::AHashMap;
use arrow::io::ipc::write::WriteOptions;
use arrow::io::ipc::{read, write};
use polars_core::prelude::*;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Read Arrows IPC format into a DataFrame
///
/// # Example
/// ```
/// use polars_core::prelude::*;
/// use std::fs::File;
/// use polars_io::ipc::IpcReader;
/// use polars_io::SerReader;
///
/// fn example() -> Result<DataFrame> {
///     let file = File::open("file.ipc").expect("file not found");
///
///     IpcReader::new(file)
///         .finish()
/// }
/// ```
pub struct IpcReader<R> {
    /// File or Stream object
    reader: R,
    /// Aggregates chunks afterwards to a single chunk.
    rechunk: bool,
    n_rows: Option<usize>,
    projection: Option<Vec<usize>>,
    columns: Option<Vec<String>>,
}

impl<R: Read + Seek> IpcReader<R> {
    /// Get schema of the Ipc File
    pub fn schema(&mut self) -> Result<Schema> {
        let metadata = read::read_file_metadata(&mut self.reader)?;
        Ok((&**metadata.schema()).into())
    }

    /// Get arrow schema of the Ipc File, this is faster than a polars schema.
    pub fn arrow_schema(&mut self) -> Result<Arc<ArrowSchema>> {
        let metadata = read::read_file_metadata(&mut self.reader)?;
        Ok(metadata.schema().clone())
    }
    /// Stop reading when `n` rows are read.
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

    #[cfg(feature = "lazy")]
    // todo! hoist to lazy crate
    pub fn finish_with_scan_ops(
        mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        aggregate: Option<&[ScanAggregation]>,
        projection: Option<&[usize]>,
    ) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = read::read_file_metadata(&mut self.reader)?;
        let reader = read::FileReader::new(
            &mut self.reader,
            metadata,
            projection.map(|x| {
                let mut x = x.to_vec();
                x.sort_unstable();
                x
            }),
        );

        finish_reader(reader, rechunk, self.n_rows, predicate, aggregate)
    }
}

impl<R> ArrowReader for read::FileReader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new((&**self.schema()).into())
    }
}

impl<R> SerReader<R> for IpcReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        IpcReader {
            reader,
            rechunk: true,
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
        let metadata = read::read_file_metadata(&mut self.reader)?;
        let schema = metadata.schema();

        if let Some(cols) = self.columns {
            let mut prj = Vec::with_capacity(cols.len());
            if cols.len() > 100 {
                let mut column_names = AHashMap::with_capacity(schema.fields().len());
                schema.fields().iter().enumerate().for_each(|(i, c)| {
                    column_names.insert(c.name(), i);
                });

                for column in cols.iter() {
                    if let Some(i) = column_names.get(&column) {
                        prj.push(*i);
                    } else {
                        let valid_fields: Vec<String> =
                            schema.fields().iter().map(|f| f.name().clone()).collect();
                        return Err(PolarsError::NotFound(format!(
                            "Unable to get field named \"{}\". Valid fields: {:?}",
                            column, valid_fields
                        )));
                    }
                }
            } else {
                for column in cols.iter() {
                    let i = schema.index_of(column)?;
                    prj.push(i);
                }
            }

            // Ipc reader panics if the projection is not in increasing order, so sorting is the safer way.
            prj.sort_unstable();
            self.projection = Some(prj);
        }

        let ipc_reader = read::FileReader::new(&mut self.reader, metadata, self.projection);
        finish_reader(ipc_reader, rechunk, self.n_rows, None, None)
    }
}

/// Write a DataFrame to Arrow's IPC format
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::ipc::IpcWriter;
/// use std::fs::File;
/// use polars_io::SerWriter;
///
/// fn example(df: &mut DataFrame) -> Result<()> {
///     let mut file = File::create("file.ipc").expect("could not create file");
///
///     IpcWriter::new(&mut file)
///         .finish(df)
/// }
///
/// ```
pub struct IpcWriter<W> {
    writer: W,
    compression: Option<write::Compression>,
}

pub use write::Compression as IpcCompression;

impl<W> IpcWriter<W>
where
    W: Write + Seek,
{
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<write::Compression>) -> Self {
        self.compression = compression;
        self
    }
}

impl<W> SerWriter<W> for IpcWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self {
        IpcWriter {
            writer,
            compression: None,
        }
    }

    fn finish(mut self, df: &DataFrame) -> Result<()> {
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            &df.schema().to_arrow(),
            WriteOptions {
                compression: self.compression,
            },
        )?;

        let iter = df.iter_record_batches();

        for batch in iter {
            ipc_writer.write(&batch)?
        }
        let _ = ipc_writer.finish()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use arrow::io::ipc::write;
    use polars_core::df;
    use polars_core::prelude::*;
    use std::io::Cursor;

    #[test]
    fn write_and_read_ipc() {
        // Vec<T> : Write + Read
        // Cursor<Vec<_>>: Seek
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let df = create_df();

        IpcWriter::new(&mut buf).finish(&df).expect("ipc writer");

        buf.set_position(0);

        let df_read = IpcReader::new(buf).finish().unwrap();
        assert!(df.frame_equal(&df_read));
    }

    #[test]
    fn test_read_ipc_with_projection() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        IpcWriter::new(&mut buf).finish(&df).expect("ipc writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = IpcReader::new(buf)
            .with_projection(Some(vec![1, 2]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }

    #[test]
    fn test_read_ipc_with_columns() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        IpcWriter::new(&mut buf).finish(&df).expect("ipc writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = IpcReader::new(buf)
            .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }

    #[test]
    fn test_write_with_compression() {
        let df = create_df();

        let compressions = vec![
            None,
            Some(write::Compression::LZ4),
            Some(write::Compression::ZSTD),
        ];

        for compression in compressions.into_iter() {
            let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
            IpcWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&df)
                .expect("ipc writer");
            buf.set_position(0);

            let df_read = IpcReader::new(buf)
                .finish()
                .expect(&format!("IPC reader: {:?}", compression));
            assert!(df.frame_equal(&df_read));
        }
    }
}
