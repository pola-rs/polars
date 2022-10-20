//! # (De)serializing Arrows Streaming IPC format.
//!
//! Arrow Streaming IPC is a [binary format format](https://arrow.apache.org/docs/python/ipc.html).
//! It used for sending an arbitrary length sequence of record batches.
//! The format must be processed from start to end, and does not support random access.
//! It is different than IPC, if you can't deserialize a file with `IpcReader::new`, it's probably an IPC Stream File.
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
//! IpcStreamWriter::new(&mut buf).finish(&mut df).expect("ipc writer");
//!
//! // reset the buffers index after writing to the beginning of the buffer
//! buf.set_position(0);
//!
//! // read the buffer into a DataFrame
//! let df_read = IpcStreamReader::new(buf).finish().unwrap();
//! assert!(df.frame_equal(&df_read));
//! ```
use std::io::{Read, Seek, Write};
use std::path::PathBuf;

use arrow::io::ipc::read::{StreamMetadata, StreamState};
use arrow::io::ipc::write::WriteOptions;
use arrow::io::ipc::{read, write};
use polars_core::prelude::*;

use crate::prelude::*;
use crate::{finish_reader, ArrowReader, ArrowResult, WriterFactory};

/// Read Arrows Stream IPC format into a DataFrame
///
/// # Example
/// ```
/// use polars_core::prelude::*;
/// use std::fs::File;
/// use polars_io::ipc::IpcStreamReader;
/// use polars_io::SerReader;
///
/// fn example() -> PolarsResult<DataFrame> {
///     let file = File::open("file.ipc").expect("file not found");
///
///     IpcStreamReader::new(file)
///         .finish()
/// }
/// ```
#[must_use]
pub struct IpcStreamReader<R> {
    /// File or Stream object
    reader: R,
    /// Aggregates chunks afterwards to a single chunk.
    rechunk: bool,
    n_rows: Option<usize>,
    projection: Option<Vec<usize>>,
    columns: Option<Vec<String>>,
    row_count: Option<RowCount>,
    metadata: Option<StreamMetadata>,
}

impl<R: Read + Seek> IpcStreamReader<R> {
    /// Get schema of the Ipc Stream File
    pub fn schema(&mut self) -> PolarsResult<Schema> {
        Ok((self.metadata()?.schema.fields.iter()).into())
    }

    /// Get arrow schema of the Ipc Stream File, this is faster than creating a polars schema.
    pub fn arrow_schema(&mut self) -> PolarsResult<ArrowSchema> {
        Ok(self.metadata()?.schema)
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

    fn metadata(&mut self) -> PolarsResult<StreamMetadata> {
        match &self.metadata {
            None => {
                let metadata = read::read_stream_metadata(&mut self.reader)?;
                self.metadata = Option::from(metadata.clone());
                Ok(metadata)
            }
            Some(md) => Ok(md.clone()),
        }
    }
}

impl<R> ArrowReader for read::StreamReader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>> {
        self.next().map_or(Ok(None), |v| match v {
            Ok(stream_state) => match stream_state {
                StreamState::Waiting => Ok(None),
                StreamState::Some(chunk) => Ok(Some(chunk)),
            },
            Err(err) => Err(err),
        })
    }
}

impl<R> SerReader<R> for IpcStreamReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        IpcStreamReader {
            reader,
            rechunk: true,
            n_rows: None,
            columns: None,
            projection: None,
            row_count: None,
            metadata: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = self.metadata()?;
        let schema = &metadata.schema;

        if let Some(columns) = self.columns {
            let prj = columns_to_projection(&columns, schema)?;
            self.projection = Some(prj);
        }

        let sorted_projection = self.projection.clone().map(|mut proj| {
            proj.sort_unstable();
            proj
        });

        let schema = if let Some(projection) = &sorted_projection {
            apply_projection(&metadata.schema, projection)
        } else {
            metadata.schema.clone()
        };

        let include_row_count = self.row_count.is_some();
        let ipc_reader =
            read::StreamReader::new(&mut self.reader, metadata.clone(), sorted_projection);
        finish_reader(
            ipc_reader,
            rechunk,
            self.n_rows,
            None,
            &schema,
            self.row_count,
        )
        .map(|df| fix_column_order(df, self.projection, include_row_count))
    }
}

fn fix_column_order(df: DataFrame, projection: Option<Vec<usize>>, row_count: bool) -> DataFrame {
    if let Some(proj) = projection {
        let offset = usize::from(row_count);
        let mut args = (0..proj.len()).zip(proj).collect::<Vec<_>>();
        // first el of tuple is argument index
        // second el is the projection index
        args.sort_unstable_by_key(|tpl| tpl.1);
        let cols = df.get_columns();

        let iter = args.iter().map(|tpl| cols[tpl.0 + offset].clone());
        let cols = if row_count {
            let mut new_cols = vec![df.get_columns()[0].clone()];
            new_cols.extend(iter);
            new_cols
        } else {
            iter.collect()
        };

        DataFrame::new_no_checks(cols)
    } else {
        df
    }
}

/// Write a DataFrame to Arrow's Streaming IPC format
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::ipc::IpcStreamWriter;
/// use std::fs::File;
/// use polars_io::SerWriter;
///
/// fn example(df: &mut DataFrame) -> PolarsResult<()> {
///     let mut file = File::create("file.ipc").expect("could not create file");
///
///     IpcStreamWriter::new(&mut file)
///         .finish(df)
/// }
///
/// ```
#[must_use]
pub struct IpcStreamWriter<W> {
    writer: W,
    compression: Option<write::Compression>,
}

use polars_core::frame::ArrowChunk;
pub use write::Compression as IpcCompression;

use crate::RowCount;

impl<W> IpcStreamWriter<W> {
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<write::Compression>) -> Self {
        self.compression = compression;
        self
    }
}

impl<W> SerWriter<W> for IpcStreamWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self {
        IpcStreamWriter {
            writer,
            compression: None,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let mut ipc_stream_writer = write::StreamWriter::new(
            &mut self.writer,
            WriteOptions {
                compression: self.compression,
            },
        );

        ipc_stream_writer.start(&df.schema().to_arrow(), None)?;

        df.rechunk();
        let iter = df.iter_chunks();

        for batch in iter {
            ipc_stream_writer.write(&batch, None)?
        }
        ipc_stream_writer.finish()?;
        Ok(())
    }
}

pub struct IpcStreamWriterOption {
    compression: Option<write::Compression>,
    extension: PathBuf,
}

impl IpcStreamWriterOption {
    pub fn new() -> Self {
        Self {
            compression: None,
            extension: PathBuf::from(".ipc"),
        }
    }

    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<write::Compression>) -> Self {
        self.compression = compression;
        self
    }

    /// Set the extension. Defaults to ".ipc".
    pub fn with_extension(mut self, extension: PathBuf) -> Self {
        self.extension = extension;
        self
    }
}

impl Default for IpcStreamWriterOption {
    fn default() -> Self {
        Self::new()
    }
}

impl WriterFactory for IpcStreamWriterOption {
    fn create_writer<W: Write + 'static>(&self, writer: W) -> Box<dyn SerWriter<W>> {
        Box::new(IpcStreamWriter::new(writer).with_compression(self.compression))
    }

    fn extension(&self) -> PathBuf {
        self.extension.to_owned()
    }
}
