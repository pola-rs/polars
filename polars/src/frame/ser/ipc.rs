//! # (De)serializing Arrows IPC format.
//!
//! Arrow IPC is a [binary format format](https://arrow.apache.org/docs/python/ipc.html).
//! It is the recommended way to serialize and deserialize Polars DataFrames as this is most true
//! to the data schema.
//!
//! ## Example
//!
//! ```rust
//! use polars::prelude::*;
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
//! IPCWriter::new(&mut buf).finish(&mut df).expect("ipc writer");
//!
//! // reset the buffers index after writing to the beginning of the buffer
//! buf.set_position(0);
//!
//! // read the buffer into a DataFrame
//! let df_read = IPCReader::new(buf).finish().unwrap();
//! assert!(df.frame_equal(&df_read));
//! ```
use super::{finish_reader, ArrowReader, ArrowResult, RecordBatch};
use crate::prelude::*;
use arrow::ipc::{
    reader::FileReader as ArrowIPCFileReader, writer::FileWriter as ArrowIPCFileWriter,
};
use arrow::record_batch::RecordBatchReader;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Read Arrows IPC format into a DataFrame
pub struct IPCReader<R> {
    /// File or Stream object
    reader: R,
    /// Aggregates chunks afterwards to a single chunk.
    rechunk: bool,
}

impl<R> ArrowReader for ArrowIPCFileReader<R>
where
    R: Read + Seek,
{
    fn next(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next_batch()
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

impl<R> SerReader<R> for IPCReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        IPCReader {
            reader,
            rechunk: true,
        }
    }
    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let ipc_reader = ArrowIPCFileReader::try_new(self.reader)?;
        finish_reader(ipc_reader, rechunk)
    }
}

/// Write a DataFrame to Arrow's IPC format
pub struct IPCWriter<'a, W> {
    writer: &'a mut W,
    batch_size: usize,
}

impl<'a, W> IPCWriter<'a, W> {
    /// Set the size of the write buffer. Batch size is the amount of rows written at once.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl<'a, W> SerWriter<'a, W> for IPCWriter<'a, W>
where
    W: Write,
{
    fn new(writer: &'a mut W) -> Self {
        IPCWriter {
            writer,
            batch_size: 1000,
        }
    }

    fn finish(self, df: &mut DataFrame) -> Result<()> {
        let mut ipc_writer = ArrowIPCFileWriter::try_new(self.writer, &df.schema)?;

        let iter = df.iter_record_batches(self.batch_size);

        for batch in iter {
            ipc_writer.write(&batch)?
        }
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
        let mut df = create_df();

        IPCWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");

        buf.set_position(0);

        let df_read = IPCReader::new(buf).finish().unwrap();
        assert!(df.frame_equal(&df_read));
    }
}
