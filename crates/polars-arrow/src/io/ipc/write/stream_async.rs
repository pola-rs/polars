//! `async` writing of arrow streams

use std::pin::Pin;
use std::task::Poll;

use futures::future::BoxFuture;
use futures::{AsyncWrite, AsyncWriteExt, FutureExt, Sink};
use polars_error::{PolarsError, PolarsResult};

use super::super::IpcField;
pub use super::common::WriteOptions;
use super::common::{encode_chunk, DictionaryTracker, EncodedData};
use super::common_async::{write_continuation, write_message};
use super::{default_ipc_fields, schema_to_bytes, Record};
use crate::datatypes::*;

/// A sink that writes array [`chunks`](crate::record_batch::RecordBatchT) as an IPC stream.
///
/// The stream header is automatically written before writing the first chunk.
pub struct StreamSink<'a, W: AsyncWrite + Unpin + Send + 'a> {
    writer: Option<W>,
    task: Option<BoxFuture<'a, PolarsResult<Option<W>>>>,
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
        schema: &ArrowSchema,
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
        schema: &ArrowSchema,
        ipc_fields: &[IpcField],
    ) -> BoxFuture<'a, PolarsResult<Option<W>>> {
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

    fn write(&mut self, record: Record<'_>) -> PolarsResult<()> {
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
            let io_err = std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "writer closed".to_string(),
            );
            Err(PolarsError::from(io_err))
        }
    }

    fn poll_complete(&mut self, cx: &mut std::task::Context<'_>) -> Poll<PolarsResult<()>> {
        if let Some(task) = &mut self.task {
            match futures::ready!(task.poll_unpin(cx)) {
                Ok(writer) => {
                    self.writer = writer;
                    self.task = None;
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

impl<'a, W> Sink<Record<'_>> for StreamSink<'a, W>
where
    W: AsyncWrite + Unpin + Send,
{
    type Error = PolarsError;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<PolarsResult<()>> {
        self.get_mut().poll_complete(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Record<'_>) -> PolarsResult<()> {
        self.get_mut().write(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<PolarsResult<()>> {
        self.get_mut().poll_complete(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<PolarsResult<()>> {
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
            },
            res => res,
        }
    }
}
