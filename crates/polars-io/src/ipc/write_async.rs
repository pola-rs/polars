use arrow::io::ipc::write::file_async::FileSink;
use arrow::io::ipc::write::WriteOptions;
use futures::{AsyncWrite, SinkExt};
use polars_core::prelude::*;

use crate::ipc::IpcWriter;

impl<W: AsyncWrite + Unpin + Send> IpcWriter<W> {
    pub fn new_async(writer: W) -> Self {
        IpcWriter {
            writer,
            compression: None,
            pl_flavor: false,
        }
    }

    pub fn batched_async(self, schema: &Schema) -> PolarsResult<BatchedWriterAsync<W>> {
        let writer = FileSink::new(
            self.writer,
            schema.to_arrow(self.pl_flavor),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        );

        Ok(BatchedWriterAsync { writer, pl_flavor: self.pl_flavor })
    }
}

pub struct BatchedWriterAsync<'a, W>
where
    W: AsyncWrite + Unpin + Send + 'a,
{
    writer: FileSink<'a, W>,
    pl_flavor: bool
}

impl<'a, W> BatchedWriterAsync<'a, W>
where
    W: AsyncWrite + Unpin + Send + 'a,
{
    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub async fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let iter = df.iter_chunks(self.pl_flavor);
        for batch in iter {
            let a = batch.into();
            self.writer.feed(a).await?;
        }
        Ok(())
    }

    /// Writes the footer of the IPC file.
    pub async fn finish(&mut self) -> PolarsResult<()> {
        self.writer.close().await?;
        Ok(())
    }
}
