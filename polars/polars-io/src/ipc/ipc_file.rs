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
use std::io::{Read, Seek, Write};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::io::ipc::write::WriteOptions;
use arrow::io::ipc::{read, write};
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{finish_reader, ArrowReader, ArrowResult};
use crate::predicates::PhysicalIoExpr;
use crate::prelude::*;
use crate::WriterFactory;

/// Read Arrows IPC format into a DataFrame
///
/// # Example
/// ```
/// use polars_core::prelude::*;
/// use std::fs::File;
/// use polars_io::ipc::IpcReader;
/// use polars_io::SerReader;
///
/// fn example() -> PolarsResult<DataFrame> {
///     let file = File::open("file.ipc").expect("file not found");
///
///     IpcReader::new(file)
///         .finish()
/// }
/// ```
#[must_use]
pub struct IpcReader<R: MmapBytesReader> {
    /// File or Stream object
    pub(super) reader: R,
    /// Aggregates chunks afterwards to a single chunk.
    rechunk: bool,
    pub(super) n_rows: Option<usize>,
    pub(super) projection: Option<Vec<usize>>,
    pub(crate) columns: Option<Vec<String>>,
    pub(super) row_count: Option<RowCount>,
    memmap: bool,
    metadata: Option<read::FileMetadata>,
}

impl<R: MmapBytesReader> IpcReader<R> {
    #[doc(hidden)]
    /// A very bad estimate of the number of rows
    /// This estimation will be entirely off if the file is compressed.
    /// And will be varying off depending on the data types.
    pub fn _num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata()?;
        let n_cols = metadata.schema.fields.len();
        // this magic number 10 is computed from the yellow trip dataset
        Ok((metadata.size as usize) / n_cols / 10)
    }
    fn get_metadata(&mut self) -> PolarsResult<&read::FileMetadata> {
        if self.metadata.is_none() {
            self.metadata = Some(read::read_file_metadata(&mut self.reader)?);
        }
        Ok(self.metadata.as_ref().unwrap())
    }

    /// Get schema of the Ipc File
    pub fn schema(&mut self) -> PolarsResult<Schema> {
        let metadata = self.get_metadata()?;
        Ok(metadata.schema.fields.iter().into())
    }

    /// Get arrow schema of the Ipc File, this is faster than creating a polars schema.
    pub fn arrow_schema(&mut self) -> PolarsResult<ArrowSchema> {
        let metadata = read::read_file_metadata(&mut self.reader)?;
        Ok(metadata.schema)
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

    /// Add a `row_count` column.
    pub fn with_row_count(mut self, row_count: Option<RowCount>) -> Self {
        self.row_count = row_count;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Set if the file is to be memory_mapped. Only works with uncompressed files.
    pub fn memory_mapped(mut self, toggle: bool) -> Self {
        self.memmap = toggle;
        self
    }

    // todo! hoist to lazy crate
    #[cfg(feature = "lazy")]
    pub fn finish_with_scan_ops(
        mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        verbose: bool,
    ) -> PolarsResult<DataFrame> {
        if self.memmap && self.reader.to_file().is_some() {
            if verbose {
                eprintln!("memory map ipc file")
            }
            match self.finish_memmapped(predicate.clone()) {
                Ok(df) => return Ok(df),
                Err(err) => {
                    match err {
                        PolarsError::ArrowError(e) => match e.as_ref() {
                            arrow::error::Error::NotYetImplemented(s)
                                if s == "mmap can only be done on uncompressed IPC files" =>
                            {
                                if verbose {
                                    eprint!("could not mmap compressed IPC file, defaulting to normal read")
                                }
                            }
                            _ => return Err(PolarsError::ArrowError(e)),
                        },
                        err => return Err(err),
                    }
                }
            }
        }
        let rechunk = self.rechunk;
        let metadata = read::read_file_metadata(&mut self.reader)?;

        let schema = if let Some(projection) = &self.projection {
            apply_projection(&metadata.schema, projection)
        } else {
            metadata.schema.clone()
        };

        let reader = read::FileReader::new(self.reader, metadata, self.projection, self.n_rows);

        finish_reader(reader, rechunk, None, predicate, &schema, self.row_count)
    }
}

impl<R: MmapBytesReader> ArrowReader for read::FileReader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }
}

impl<R: MmapBytesReader> SerReader<R> for IpcReader<R> {
    fn new(reader: R) -> Self {
        IpcReader {
            reader,
            rechunk: true,
            n_rows: None,
            columns: None,
            projection: None,
            row_count: None,
            memmap: true,
            metadata: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        if self.memmap && self.reader.to_file().is_some() {
            match self.finish_memmapped(None) {
                Ok(df) => return Ok(df),
                Err(err) => match err {
                    PolarsError::ArrowError(e) => match e.as_ref() {
                        arrow::error::Error::NotYetImplemented(s)
                            if s == "mmap can only be done on uncompressed IPC files" =>
                        {
                            eprint!("could not mmap compressed IPC file, defaulting to normal read")
                        }
                        _ => return Err(PolarsError::ArrowError(e)),
                    },
                    err => return Err(err),
                },
            }
        }
        let rechunk = self.rechunk;
        let metadata = read::read_file_metadata(&mut self.reader)?;
        let schema = &metadata.schema;

        if let Some(columns) = &self.columns {
            let prj = columns_to_projection(columns, schema)?;
            self.projection = Some(prj);
        }

        let schema = if let Some(projection) = &self.projection {
            apply_projection(&metadata.schema, projection)
        } else {
            metadata.schema.clone()
        };

        let ipc_reader =
            read::FileReader::new(self.reader, metadata.clone(), self.projection, self.n_rows);
        finish_reader(ipc_reader, rechunk, None, None, &schema, self.row_count)
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
/// fn example(df: &mut DataFrame) -> PolarsResult<()> {
///     let mut file = File::create("file.ipc").expect("could not create file");
///
///     IpcWriter::new(&mut file)
///         .finish(df)
/// }
///
/// ```
#[must_use]
pub struct IpcWriter<W> {
    writer: W,
    compression: Option<IpcCompression>,
}

use polars_core::frame::ArrowChunk;

use crate::mmap::MmapBytesReader;
use crate::RowCount;

impl<W: Write> IpcWriter<W> {
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<IpcCompression>) -> Self {
        self.compression = compression;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let mut writer = write::FileWriter::new(
            self.writer,
            schema.to_arrow(),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        );
        writer.start()?;

        Ok(BatchedWriter { writer })
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

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            df.schema().to_arrow(),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        )?;
        df.rechunk();
        let iter = df.iter_chunks();

        for batch in iter {
            ipc_writer.write(&batch, None)?
        }
        ipc_writer.finish()?;
        Ok(())
    }
}

pub struct BatchedWriter<W: Write> {
    writer: write::FileWriter<W>,
}

impl<W: Write> BatchedWriter<W> {
    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let iter = df.iter_chunks();
        for batch in iter {
            self.writer.write(&batch, None)?
        }
        Ok(())
    }

    /// Writes the footer of the IPC file.
    pub fn finish(&mut self) -> PolarsResult<()> {
        self.writer.finish()?;
        Ok(())
    }
}

/// Compression codec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IpcCompression {
    /// LZ4 (framed)
    LZ4,
    /// ZSTD
    #[default]
    ZSTD,
}

impl From<IpcCompression> for write::Compression {
    fn from(value: IpcCompression) -> Self {
        match value {
            IpcCompression::LZ4 => write::Compression::LZ4,
            IpcCompression::ZSTD => write::Compression::ZSTD,
        }
    }
}

pub struct IpcWriterOption {
    compression: Option<IpcCompression>,
    extension: PathBuf,
}

impl IpcWriterOption {
    pub fn new() -> Self {
        Self {
            compression: None,
            extension: PathBuf::from(".ipc"),
        }
    }

    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<IpcCompression>) -> Self {
        self.compression = compression;
        self
    }

    /// Set the extension. Defaults to ".ipc".
    pub fn with_extension(mut self, extension: PathBuf) -> Self {
        self.extension = extension;
        self
    }
}

impl Default for IpcWriterOption {
    fn default() -> Self {
        Self::new()
    }
}

impl WriterFactory for IpcWriterOption {
    fn create_writer<W: Write + 'static>(&self, writer: W) -> Box<dyn SerWriter<W>> {
        Box::new(IpcWriter::new(writer).with_compression(self.compression))
    }

    fn extension(&self) -> PathBuf {
        self.extension.to_owned()
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use polars_core::df;
    use polars_core::prelude::*;

    use crate::prelude::*;

    #[test]
    fn write_and_read_ipc() {
        // Vec<T> : Write + Read
        // Cursor<Vec<_>>: Seek
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = create_df();

        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");

        buf.set_position(0);

        let df_read = IpcReader::new(buf).finish().unwrap();
        assert!(df.frame_equal(&df_read));
    }

    #[test]
    fn test_read_ipc_with_projection() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
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
        let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = IpcReader::new(buf)
            .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
            .finish()
            .unwrap();
        df_read.frame_equal(&expected);

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df![
            "letters" => ["x", "y", "z"],
            "ints" => [123, 456, 789],
            "floats" => [4.5, 10.0, 10.0],
            "other" => ["misc", "other", "value"],
        ]
        .unwrap();
        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);
        let expected = df![
            "letters" => ["x", "y", "z"],
            "floats" => [4.5, 10.0, 10.0],
            "other" => ["misc", "other", "value"],
            "ints" => [123, 456, 789],
        ]
        .unwrap();
        let df_read = IpcReader::new(&mut buf)
            .with_columns(Some(vec![
                "letters".to_string(),
                "floats".to_string(),
                "other".to_string(),
                "ints".to_string(),
            ]))
            .finish()
            .unwrap();
        assert!(df_read.frame_equal(&expected));
    }

    #[test]
    fn test_write_with_compression() {
        let mut df = create_df();

        let compressions = vec![None, Some(IpcCompression::LZ4), Some(IpcCompression::ZSTD)];

        for compression in compressions.into_iter() {
            let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
            IpcWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut df)
                .expect("ipc writer");
            buf.set_position(0);

            let df_read = IpcReader::new(buf)
                .finish()
                .expect(&format!("IPC reader: {:?}", compression));
            assert!(df.frame_equal(&df_read));
        }
    }

    #[test]
    fn write_and_read_ipc_empty_series() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let chunked_array = Float64Chunked::new("empty", &[0_f64; 0]);
        let mut df = DataFrame::new(vec![chunked_array.into_series()]).unwrap();
        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");

        buf.set_position(0);

        let df_read = IpcReader::new(buf).finish().unwrap();
        assert!(df.frame_equal(&df_read));
    }
}
