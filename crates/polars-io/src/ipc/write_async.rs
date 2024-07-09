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
            compat_level: CompatLevel::oldest(),
        }
    }

    pub fn batched_async(self, schema: &Schema) -> PolarsResult<BatchedWriterAsync<W>> {
        let writer = FileSink::new(
            self.writer,
            schema.to_arrow(CompatLevel::oldest()),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        );

        Ok(BatchedWriterAsync { writer })
    }
}

pub struct BatchedWriterAsync<'a, W>
where
    W: AsyncWrite + Unpin + Send + 'a,
{
    writer: FileSink<'a, W>,
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
        let iter = df.iter_chunks(CompatLevel::oldest(), true);
        for batch in iter {
            self.writer.feed(batch.into()).await?;
        }
        Ok(())
    }

    /// Writes the footer of the IPC file.
    pub async fn finish(&mut self) -> PolarsResult<()> {
        self.writer.close().await?;
        Ok(())
    }
}
