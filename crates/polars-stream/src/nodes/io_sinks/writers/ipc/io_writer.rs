use std::sync::Arc;

use arrow::io::ipc::IpcField;
use arrow::io::ipc::write::common_sync::{push_footer, push_magic, push_message};
use arrow::io::ipc::write::schema::schema_to_bytes;
use arrow::io::ipc::write::{EncodedDataBytes, arrow_ipc_block};
use bytes::Bytes;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow;
use polars_error::{PolarsResult, to_compute_err};
use polars_io::ipc::pl_ipc_metadata::{POLARS_IPC_METADATA_KEY, PlIpcMetadata};
use polars_io::ipc::{IpcWriter, IpcWriterOptions};
use polars_io::utils::file::Writeable;
use polars_io::{SerWriter, schema_to_arrow_checked};

use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;
use crate::nodes::io_sinks::writers::ipc::IpcBatch;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub ipc_batch_rx: tokio::sync::mpsc::Receiver<IpcBatch>,
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

        let (file, sync_on_close) = file.await?;

        let mut custom_pl_metadata = write_custom_pl_metadata.then_some(PlIpcMetadata::default());

        match file {
            Writeable::Cloud(cloudwriter) => {
                // The zero-copy implementation takes ownership of the encoded data and, after
                // framing and aligning, passes it to the object_store via the BufWriter::put() method.
                let mut cloud_writer = cloudwriter.into_cloud_writer().await?;
                let mut record_blocks = vec![];
                let mut dictionary_blocks = vec![];
                let mut block_offsets = 0;
                let mut sink_queue = Vec::new();

                // Start with header.
                let offset = push_magic(&mut sink_queue, true);
                for bytes in sink_queue.drain(..) {
                    cloud_writer.write_all_owned(bytes).await?;
                }
                block_offsets += offset;

                // Add schema message.
                let schema = schema_to_arrow_checked(&schema, options.compat_level, "ipc")?;
                let encoded_data = EncodedDataBytes {
                    ipc_message: Bytes::from(schema_to_bytes(&schema, &ipc_fields, None)),
                    arrow_data: Bytes::new(),
                };
                let (meta, data) = push_message(&mut sink_queue, encoded_data);
                for bytes in sink_queue.drain(..) {
                    cloud_writer.write_all_owned(bytes).await?;
                }
                block_offsets += meta + data;

                // Process each incoming batch as a message.
                while let Some(batch) = ipc_batch_rx.recv().await {
                    match batch {
                        IpcBatch::Record {
                            encoded_data,
                            morsel_permit,
                            num_rows,
                        } => {
                            let encoded_data = encoded_data.await;
                            let encoded_data = EncodedDataBytes {
                                ipc_message: Bytes::from(encoded_data.ipc_message),
                                arrow_data: Bytes::from(encoded_data.arrow_data),
                            };

                            let (meta, data) = push_message(&mut sink_queue, encoded_data);
                            for bytes in sink_queue.drain(..) {
                                cloud_writer.write_all_owned(bytes).await?;
                            }

                            let block = arrow_ipc_block(block_offsets, meta, data);
                            record_blocks.push(block);
                            block_offsets += meta + data;

                            if let Some(md) = custom_pl_metadata.as_mut() {
                                if let Some(end_offset) = num_rows.checked_add(
                                    md.record_batch_cum_len.last().copied().unwrap_or(0),
                                ) {
                                    md.record_batch_cum_len.push(end_offset);
                                } else {
                                    custom_pl_metadata = None;
                                };
                            }

                            drop(morsel_permit);
                        },
                        IpcBatch::Dictionary(dictionary_data) => {
                            let encoded_dictionary = EncodedDataBytes {
                                ipc_message: Bytes::from(dictionary_data.ipc_message),
                                arrow_data: Bytes::from(dictionary_data.arrow_data),
                            };

                            let (meta, data) = push_message(&mut sink_queue, encoded_dictionary);
                            for bytes in sink_queue.drain(..) {
                                cloud_writer.write_all_owned(bytes).await?;
                            }

                            let block = arrow_ipc_block(block_offsets, meta, data);
                            dictionary_blocks.push(block);
                            block_offsets += meta + data;
                        },
                    }
                }

                // Write footer.
                push_footer(
                    &mut sink_queue,
                    &schema,
                    &ipc_fields,
                    dictionary_blocks,
                    record_blocks,
                    if let Some(custom_pl_metadata) = custom_pl_metadata {
                        Some(vec![(
                            POLARS_IPC_METADATA_KEY.into(),
                            serde_json::to_string(&custom_pl_metadata).map_err(to_compute_err)?,
                        )])
                    } else {
                        None
                    },
                    None,
                );
                push_magic(&mut sink_queue, false);
                for bytes in sink_queue.drain(..) {
                    cloud_writer.write_all_owned(bytes).await?;
                }

                // Finish.
                cloud_writer.finish().await?;
            },
            mut file => {
                let mut buffered_file = file.as_buffered();

                let mut ipc_writer = IpcWriter::new(&mut *buffered_file)
                    .with_compression(options.compression)
                    .with_compat_level(options.compat_level)
                    .with_parallel(false)
                    .batched(&schema, ipc_fields)?;

                while let Some(batch) = ipc_batch_rx.recv().await {
                    match batch {
                        IpcBatch::Record {
                            encoded_data,
                            morsel_permit,
                            num_rows,
                        } => {
                            let encoded_data = encoded_data.await;
                            ipc_writer.write_encoded(&[], &encoded_data)?;

                            if let Some(md) = custom_pl_metadata.as_mut() {
                                if let Some(end_offset) = num_rows.checked_add(
                                    md.record_batch_cum_len.last().copied().unwrap_or(0),
                                ) {
                                    md.record_batch_cum_len.push(end_offset);
                                } else {
                                    custom_pl_metadata = None;
                                };
                            }

                            drop(encoded_data);
                            drop(morsel_permit);
                        },
                        IpcBatch::Dictionary(dictionary_data) => {
                            ipc_writer.write_encoded_dictionaries(&[dictionary_data])?
                        },
                    }
                }

                if let Some(custom_pl_metadata) = custom_pl_metadata {
                    ipc_writer.set_custom_metadata(vec![(
                        POLARS_IPC_METADATA_KEY.into(),
                        serde_json::to_string(&custom_pl_metadata).map_err(to_compute_err)?,
                    )]);
                }

                ipc_writer.finish()?;

                drop(ipc_writer);
                drop(buffered_file);
                file.close(sync_on_close)?;
            },
        }

        Ok(())
    }
}
