use ahash::AHashMap;
use std::{pin::Pin, task::Poll};

use futures::{future::BoxFuture, AsyncWrite, AsyncWriteExt, FutureExt, Sink, TryFutureExt};
use parquet2::metadata::KeyValue;
use parquet2::write::FileStreamer;
use parquet2::write::WriteOptions as ParquetWriteOptions;

use crate::{array::Array, chunk::Chunk, datatypes::Schema, error::Error};

use super::file::add_arrow_schema;
use super::{Encoding, SchemaDescriptor, WriteOptions};

/// Sink that writes array [`chunks`](Chunk) as a Parquet file.
///
/// Any values in the sink's `metadata` field will be written to the file's footer
/// when the sink is closed.
///
/// # Examples
///
/// ```
/// use futures::SinkExt;
/// use arrow2::array::{Array, Int32Array};
/// use arrow2::datatypes::{DataType, Field, Schema};
/// use arrow2::chunk::Chunk;
/// use arrow2::io::parquet::write::{Encoding, WriteOptions, CompressionOptions, Version};
/// # use arrow2::io::parquet::write::FileSink;
/// # futures::executor::block_on(async move {
///
/// let schema = Schema::from(vec![
///     Field::new("values", DataType::Int32, true),
/// ]);
/// let encoding = vec![vec![Encoding::Plain]];
/// let options = WriteOptions {
///     write_statistics: true,
///     compression: CompressionOptions::Uncompressed,
///     version: Version::V2,
///     data_pagesize_limit: None,
/// };
///
/// let mut buffer = vec![];
/// let mut sink = FileSink::try_new(
///     &mut buffer,
///     schema,
///     encoding,
///     options,
/// )?;
///
/// for i in 0..3 {
///     let values = Int32Array::from(&[Some(i), None]);
///     let chunk = Chunk::new(vec![values.boxed()]);
///     sink.feed(chunk).await?;
/// }
/// sink.metadata.insert(String::from("key"), Some(String::from("value")));
/// sink.close().await?;
/// # arrow2::error::Result::Ok(())
/// # }).unwrap();
/// ```
pub struct FileSink<'a, W: AsyncWrite + Send + Unpin> {
    writer: Option<FileStreamer<W>>,
    task: Option<BoxFuture<'a, Result<Option<FileStreamer<W>>, Error>>>,
    options: WriteOptions,
    encodings: Vec<Vec<Encoding>>,
    schema: Schema,
    parquet_schema: SchemaDescriptor,
    /// Key-value metadata that will be written to the file on close.
    pub metadata: AHashMap<String, Option<String>>,
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
        schema: Schema,
        encodings: Vec<Vec<Encoding>>,
        options: WriteOptions,
    ) -> Result<Self, Error> {
        if encodings.len() != schema.fields.len() {
            return Err(Error::InvalidArgumentError(
                "The number of encodings must equal the number of fields".to_string(),
            ));
        }

        let parquet_schema = crate::io::parquet::write::to_parquet_schema(&schema)?;
        let created_by = Some("Arrow2 - Native Rust implementation of Arrow".to_string());
        let writer = FileStreamer::new(
            writer,
            parquet_schema.clone(),
            ParquetWriteOptions {
                version: options.version,
                write_statistics: options.write_statistics,
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
            metadata: AHashMap::default(),
        })
    }

    /// The Arrow [`Schema`] for the file.
    pub fn schema(&self) -> &Schema {
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
    ) -> std::task::Poll<Result<(), Error>> {
        if let Some(task) = &mut self.task {
            match futures::ready!(task.poll_unpin(cx)) {
                Ok(writer) => {
                    self.task = None;
                    self.writer = writer;
                    Poll::Ready(Ok(()))
                }
                Err(error) => {
                    self.task = None;
                    Poll::Ready(Err(error))
                }
            }
        } else {
            Poll::Ready(Ok(()))
        }
    }
}

impl<'a, W> Sink<Chunk<Box<dyn Array>>> for FileSink<'a, W>
where
    W: AsyncWrite + Send + Unpin + 'a,
{
    type Error = Error;

    fn start_send(self: Pin<&mut Self>, item: Chunk<Box<dyn Array>>) -> Result<(), Self::Error> {
        if self.schema.fields.len() != item.arrays().len() {
            return Err(Error::InvalidArgumentError(
                "The number of arrays in the chunk must equal the number of fields in the schema"
                    .to_string(),
            ));
        }
        let this = self.get_mut();
        if let Some(mut writer) = this.writer.take() {
            let rows = crate::io::parquet::write::row_group_iter(
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
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "writer closed".to_string(),
            )))
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
                        writer.end(kv_meta).map_err(Error::from).await?;
                        writer.into_inner().close().map_err(Error::from).await?;
                        Ok(None)
                    }));
                    this.poll_complete(cx)
                } else {
                    Poll::Ready(Ok(()))
                }
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}
