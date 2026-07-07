use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use arrow::io::ipc::write::arrow_ipc_block;
use arrow::io::ipc::write::schema::serialize_schema;
use arrow::io::ipc::write2::footer::serialize_ipc_footer_and_magic_bytes;
use arrow::io::ipc::write2::message::finish_ipc_message_bytes;
use arrow::io::ipc::write2::schema::serialize_ipc_schema_message_bytes;
use arrow::io::ipc::{ARROW_MAGIC_V2_PADDED, IpcField};
use bytes::Bytes;
use polars_async::executor;
use polars_async::primitives::wait_group::WaitToken;
use polars_buffer::Buffer;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow;
use polars_error::{PolarsResult, to_compute_err};
use polars_io::ipc::IpcWriterOptions;
use polars_io::ipc::pl_ipc_metadata::{POLARS_IPC_METADATA_KEY, PlIpcMetadata};
use polars_io::schema_to_arrow_checked;
use polars_io::utils::bytes_bufferer::{BytesBufferer, BytesBuffererConfig};
use polars_io::utils::file::Writable;

use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;
use crate::nodes::io_sinks::writers::ipc::{IpcBatch, IpcBatchType};

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub ipc_batch_rx: tokio::sync::mpsc::Receiver<(
        executor::AbortOnDropHandle<PolarsResult<IpcBatch>>,
        Option<WaitToken>,
    )>,
    pub options: Arc<IpcWriterOptions>,
    pub schema: SchemaRef,
    pub ipc_fields: Vec<IpcField>,
    pub write_custom_pl_metadata: bool,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        let IOWriter {
            file,
            mut ipc_batch_rx,
            options,
            schema,
            ipc_fields,
            write_custom_pl_metadata,
        } = self;

        let (mut writable, sync_on_close) = file.await?;

        let mut writer = WritableWrap {
            writable: &mut writable,
        };

        let mut custom_pl_metadata = write_custom_pl_metadata.then_some(PlIpcMetadata::default());

        let mut record_blocks = vec![];
        let mut dictionary_blocks = vec![];

        writer
            .write_multiple_owned_unbuffered([Buffer::from_static(&ARROW_MAGIC_V2_PADDED)])
            .await?;

        let mut current_byte_offset: usize = ARROW_MAGIC_V2_PADDED.len();

        let schema = schema_to_arrow_checked(&schema, options.compat_level, "ipc")?;
        let serialized_ipc_schema = Box::new(serialize_schema(&schema, &ipc_fields, None));

        let mut schema_bytes = BytesBufferer::new(schema_bytes_bufferer_config());

        schema_bytes.push_owned(&serialize_ipc_schema_message_bytes(
            serialized_ipc_schema.clone(),
        ));
        finish_ipc_message_bytes(&mut schema_bytes)?;

        let schema_bytes_len = schema_bytes.len();

        writer.write_multiple_owned_unbuffered(schema_bytes).await?;

        current_byte_offset += schema_bytes_len;

        while let Some((batch, finish_write_token)) = ipc_batch_rx.recv().await {
            let IpcBatch {
                batch_type,
                num_rows,
                continuation_bytes,
                message_bytes,
                message_num_bytes,
                arrow_data_bytes,
                arrow_data_num_bytes,
                morsel_permit,
            } = batch.await?;

            let metadata_num_bytes = continuation_bytes.len() + message_num_bytes;

            let ipc_block = arrow_ipc_block(
                current_byte_offset,
                metadata_num_bytes,
                arrow_data_num_bytes,
            );

            match batch_type {
                IpcBatchType::Record => record_blocks.push(ipc_block),
                IpcBatchType::Dictionary => dictionary_blocks.push(ipc_block),
            };

            current_byte_offset += metadata_num_bytes + arrow_data_num_bytes;

            writer
                .write_multiple_owned_unbuffered(
                    std::iter::once(Buffer::from_vec(continuation_bytes))
                        .chain(message_bytes)
                        .chain(arrow_data_bytes),
                )
                .await?;

            if let Some(md) = custom_pl_metadata.as_mut() {
                if let Some(end_offset) =
                    num_rows.checked_add(md.record_batch_cum_len.last().copied().unwrap_or(0))
                {
                    md.record_batch_cum_len.push(end_offset);
                } else {
                    custom_pl_metadata = None;
                };
            }

            drop(morsel_permit);
            drop(finish_write_token);
        }

        let custom_metadata = if let Some(custom_pl_metadata) = custom_pl_metadata {
            Some(vec![(
                POLARS_IPC_METADATA_KEY.into(),
                serde_json::to_string(&custom_pl_metadata).map_err(to_compute_err)?,
            )])
        } else {
            None
        };

        let mut schema_bytes = BytesBufferer::new(schema_bytes_bufferer_config());

        serialize_ipc_footer_and_magic_bytes(
            &mut schema_bytes,
            serialized_ipc_schema,
            dictionary_blocks,
            record_blocks,
            custom_metadata,
        )?;

        writer.write_multiple_owned_unbuffered(schema_bytes).await?;
        writable.close(sync_on_close)?;

        Ok(())
    }
}

/// Use smaller limits for IPC schema messages.
fn schema_bytes_bufferer_config() -> &'static BytesBuffererConfig {
    &const {
        BytesBuffererConfig {
            target_size: NonZeroUsize::new(8192).unwrap()..NonZeroUsize::MAX,
            copy_buffer_reserve_size: NonZeroUsize::new(8192).unwrap()..NonZeroUsize::MAX,
        }
    }
}

struct WritableWrap<'a> {
    writable: &'a mut Writable,
}

impl<'a> Deref for WritableWrap<'a> {
    type Target = Writable;

    fn deref(&self) -> &Self::Target {
        self.writable
    }
}

impl<'a> DerefMut for WritableWrap<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.writable
    }
}

impl<'a> WritableWrap<'a> {
    async fn write_multiple_owned_unbuffered<I>(&mut self, bytes: I) -> PolarsResult<()>
    where
        I: IntoIterator<Item = Buffer<u8>>,
    {
        use polars_io::utils::file::Writable::*;

        match self.writable {
            Cloud(v) => {
                v.write_multiple_owned_unbuffered(bytes.into_iter().map(Bytes::from_owner))
                    .await?;
            },
            Dyn(_) | Local(_) => {
                for buffer in bytes {
                    self.write_all(Buffer::as_slice(&buffer))?;
                }
            },
        }

        Ok(())
    }
}
