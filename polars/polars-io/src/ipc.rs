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
        }
    }
    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = read::read_file_metadata(&mut self.reader)?;
        let ipc_reader = read::FileReader::new(&mut self.reader, metadata, None);
        finish_reader(ipc_reader, rechunk, None, None, None)
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
}

impl<W> SerWriter<W> for IpcWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self {
        IpcWriter { writer }
    }

    fn finish(mut self, df: &DataFrame) -> Result<()> {
        let mut ipc_writer = write::FileWriter::try_new(&mut self.writer, &df.schema().to_arrow())?;

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
}
