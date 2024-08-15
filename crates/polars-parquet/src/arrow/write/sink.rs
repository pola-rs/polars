use std::pin::Pin;
use std::task::Poll;

use arrow::array::Array;
use arrow::datatypes::ArrowSchema;
use arrow::record_batch::RecordBatchT;
use futures::future::BoxFuture;
use futures::{AsyncWrite, AsyncWriteExt, FutureExt, Sink, TryFutureExt};
use polars_error::{polars_bail, to_compute_err, PolarsError, PolarsResult};
use polars_utils::aliases::PlHashMap;

use super::file::add_arrow_schema;
use super::{Encoding, SchemaDescriptor, WriteOptions};
use crate::parquet::metadata::KeyValue;
use crate::parquet::write::{FileStreamer, WriteOptions as ParquetWriteOptions};

/// Sink that writes array [`chunks`](RecordBatchT) as a Parquet file.
///
/// Any values in the sink's `metadata` field will be written to the file's footer
/// when the sink is closed.
pub struct FileSink<'a, W: AsyncWrite + Send + Unpin> {
    writer: Option<FileStreamer<W>>,
    task: Option<BoxFuture<'a, PolarsResult<Option<FileStreamer<W>>>>>,
    options: WriteOptions,
    encodings: Vec<Vec<Encoding>>,
    schema: ArrowSchema,
    parquet_schema: SchemaDescriptor,
    /// Key-value metadata that will be written to the file on close.
    pub metadata: PlHashMap<String, Option<String>>,
}

impl<'a, W> FileSink<'a, W>
where
    W: AsyncWrite + Send + Unpin + 'a,
{
    /// Create a new sink that writes arrays to the provided `writer`.
    ///
    /// # Error
    /// Iff
    /// * the Arrow schema can't be converted to a valid Parquet schema.
    /// * the length of the encodings is different from the number of fields in schema
    pub fn try_new(
        writer: W,
        schema: ArrowSchema,
        encodings: Vec<Vec<Encoding>>,
        options: WriteOptions,
    ) -> PolarsResult<Self> {
        if encodings.len() != schema.fields.len() {
            polars_bail!(InvalidOperation:
                "The number of encodings must equal the number of fields".to_string(),
            )
        }

        let parquet_schema = crate::arrow::write::to_parquet_schema(&schema)?;
        let created_by = Some("Polars".to_string());
        let writer = FileStreamer::new(
            writer,
            parquet_schema.clone(),
            ParquetWriteOptions {
                version: options.version,
                write_statistics: options.has_statistics(),
            },
            created_by,
        );
        Ok(Self {
            writer: Some(writer),
            task: None,
            options,
            schema,
            encodings,
            parquet_schema,
            metadata: PlHashMap::default(),
        })
    }

    /// The Arrow [`ArrowSchema`] for the file.
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }

    /// The Parquet [`SchemaDescriptor`] for the file.
    pub fn parquet_schema(&self) -> &SchemaDescriptor {
        &self.parquet_schema
    }

    /// The write options for the file.
    pub fn options(&self) -> &WriteOptions {
        &self.options
    }

    fn poll_complete(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<PolarsResult<()>> {
        if let Some(task) = &mut self.task {
            match futures::ready!(task.poll_unpin(cx)) {
                Ok(writer) => {
                    self.task = None;
                    self.writer = writer;
                    Poll::Ready(Ok(()))
                },
                Err(error) => {
                    self.task = None;
                    Poll::Ready(Err(error))
                },
            }
        } else {
            Poll::Ready(Ok(()))
        }
    }
}

impl<'a, W> Sink<RecordBatchT<Box<dyn Array>>> for FileSink<'a, W>
where
    W: AsyncWrite + Send + Unpin + 'a,
{
    type Error = PolarsError;

    fn start_send(
        self: Pin<&mut Self>,
        item: RecordBatchT<Box<dyn Array>>,
    ) -> Result<(), Self::Error> {
        if self.schema.fields.len() != item.arrays().len() {
            polars_bail!(InvalidOperation:
                "The number of arrays in the chunk must equal the number of fields in the schema"
            )
        }
        let this = self.get_mut();
        if let Some(mut writer) = this.writer.take() {
            let rows = crate::arrow::write::row_group_iter(
                item,
                this.encodings.clone(),
                this.parquet_schema.fields().to_vec(),
                this.options,
            );
            this.task = Some(Box::pin(async move {
                writer.write(rows).await?;
                Ok(Some(writer))
            }));
            Ok(())
        } else {
            let io_err = std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "writer closed".to_string(),
            );
            Err(PolarsError::from(io_err))
        }
    }

    fn poll_ready(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.get_mut().poll_complete(cx)
    }

    fn poll_flush(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.get_mut().poll_complete(cx)
    }

    fn poll_close(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        match futures::ready!(this.poll_complete(cx)) {
            Ok(()) => {
                let writer = this.writer.take();
                if let Some(mut writer) = writer {
                    let meta = std::mem::take(&mut this.metadata);
                    let metadata = if meta.is_empty() {
                        None
                    } else {
                        Some(
                            meta.into_iter()
                                .map(|(k, v)| KeyValue::new(k, v))
                                .collect::<Vec<_>>(),
                        )
                    };
                    let kv_meta = add_arrow_schema(&this.schema, metadata);

                    this.task = Some(Box::pin(async move {
                        writer.end(kv_meta).map_err(to_compute_err).await?;
                        writer.into_inner().close().map_err(to_compute_err).await?;
                        Ok(None)
                    }));
                    this.poll_complete(cx)
                } else {
                    Poll::Ready(Ok(()))
                }
            },
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}
