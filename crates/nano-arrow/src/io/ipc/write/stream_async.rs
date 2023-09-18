//! `async` writing of arrow streams

use std::{pin::Pin, task::Poll};

use futures::{future::BoxFuture, AsyncWrite, AsyncWriteExt, FutureExt, Sink};

use super::super::IpcField;
pub use super::common::WriteOptions;
use super::common::{encode_chunk, DictionaryTracker, EncodedData};
use super::common_async::{write_continuation, write_message};
use super::{default_ipc_fields, schema_to_bytes, Record};

use crate::datatypes::*;
use crate::error::{Error, Result};

/// A sink that writes array [`chunks`](crate::chunk::Chunk) as an IPC stream.
///
/// The stream header is automatically written before writing the first chunk.
///
/// # Examples
///
/// ```
/// use futures::SinkExt;
/// use arrow2::array::{Array, Int32Array};
/// use arrow2::datatypes::{DataType, Field, Schema};
/// use arrow2::chunk::Chunk;
/// # use arrow2::io::ipc::write::stream_async::StreamSink;
/// # futures::executor::block_on(async move {
/// let schema = Schema::from(vec![
///     Field::new("values", DataType::Int32, true),
/// ]);
///
/// let mut buffer = vec![];
/// let mut sink = StreamSink::new(
///     &mut buffer,
///     &schema,
///     None,
///     Default::default(),
/// );
///
/// for i in 0..3 {
///     let values = Int32Array::from(&[Some(i), None]);
///     let chunk = Chunk::new(vec![values.boxed()]);
///     sink.feed(chunk.into()).await?;
/// }
/// sink.close().await?;
/// # arrow2::error::Result::Ok(())
/// # }).unwrap();
/// ```
pub struct StreamSink<'a, W: AsyncWrite + Unpin + Send + 'a> {
    writer: Option<W>,
    task: Option<BoxFuture<'a, Result<Option<W>>>>,
    options: WriteOptions,
    dictionary_tracker: DictionaryTracker,
    fields: Vec<IpcField>,
}

impl<'a, W> StreamSink<'a, W>
where
    W: AsyncWrite + Unpin + Send + 'a,
{
    /// Create a new [`StreamSink`].
    pub fn new(
        writer: W,
        schema: &Schema,
        ipc_fields: Option<Vec<IpcField>>,
        write_options: WriteOptions,
    ) -> Self {
        let fields = ipc_fields.unwrap_or_else(|| default_ipc_fields(&schema.fields));
        let task = Some(Self::start(writer, schema, &fields[..]));
        Self {
            writer: None,
            task,
            fields,
            dictionary_tracker: DictionaryTracker {
                dictionaries: Default::default(),
                cannot_replace: false,
            },
            options: write_options,
        }
    }

    fn start(
        mut writer: W,
        schema: &Schema,
        ipc_fields: &[IpcField],
    ) -> BoxFuture<'a, Result<Option<W>>> {
        let message = EncodedData {
            ipc_message: schema_to_bytes(schema, ipc_fields),
            arrow_data: vec![],
        };
        async move {
            write_message(&mut writer, message).await?;
            Ok(Some(writer))
        }
        .boxed()
    }

    fn write(&mut self, record: Record<'_>) -> Result<()> {
        let fields = record.fields().unwrap_or(&self.fields[..]);
        let (dictionaries, message) = encode_chunk(
            record.columns(),
            fields,
            &mut self.dictionary_tracker,
            &self.options,
        )?;

        if let Some(mut writer) = self.writer.take() {
            self.task = Some(
                async move {
                    for d in dictionaries {
                        write_message(&mut writer, d).await?;
                    }
                    write_message(&mut writer, message).await?;
                    Ok(Some(writer))
                }
                .boxed(),
            );
            Ok(())
        } else {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "writer closed".to_string(),
            )))
        }
    }

    fn poll_complete(&mut self, cx: &mut std::task::Context<'_>) -> Poll<Result<()>> {
        if let Some(task) = &mut self.task {
            match futures::ready!(task.poll_unpin(cx)) {
                Ok(writer) => {
                    self.writer = writer;
                    self.task = None;
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

impl<'a, W> Sink<Record<'_>> for StreamSink<'a, W>
where
    W: AsyncWrite + Unpin + Send,
{
    type Error = Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Result<()>> {
        self.get_mut().poll_complete(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Record<'_>) -> Result<()> {
        self.get_mut().write(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Result<()>> {
        self.get_mut().poll_complete(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Result<()>> {
        let this = self.get_mut();
        match this.poll_complete(cx) {
            Poll::Ready(Ok(())) => {
                if let Some(mut writer) = this.writer.take() {
                    this.task = Some(
                        async move {
                            write_continuation(&mut writer, 0).await?;
                            writer.flush().await?;
                            writer.close().await?;
                            Ok(None)
                        }
                        .boxed(),
                    );
                    this.poll_complete(cx)
                } else {
                    Poll::Ready(Ok(()))
                }
            }
            res => res,
        }
    }
}
