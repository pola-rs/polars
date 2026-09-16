use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::Field as ArrowField;
use arrow::io::ipc::write2::array::{IpcBatchSerializationContext, write_array};
use arrow::io::ipc::write2::message::{
    finish_encode_ipc_dictionary_batch, finish_encode_ipc_record_batch,
};
use polars_async::executor::{self, TaskPriority};
use polars_async::primitives::connector;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_async::primitives::wait_group::{WaitGroup, WaitToken};
use polars_core::prelude::CompatLevel;
use polars_core::series::arrow_export::ToArrowConverter;
use polars_core::utils::arrow;
use polars_core::utils::arrow::io::ipc::write::{WriteOptions, schema};
use polars_error::PolarsResult;
use polars_io::utils::bytes_bufferer::{BytesBufferer, BytesBuffererConfig};
use polars_utils::IdxSize;

use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::writers::interface::IPC_RW_RECORD_BATCH_FLAGS_KEY;
use crate::nodes::io_sinks::writers::ipc::{IpcBatch, IpcBatchType};

pub struct RecordBatchEncoder {
    pub morsel_rx: connector::Receiver<SinkMorsel>,
    pub ipc_batch_tx: tokio::sync::mpsc::Sender<(
        executor::AbortOnDropHandle<PolarsResult<IpcBatch>>,
        Option<WaitToken>,
    )>,
    pub arrow_converters: Vec<(ToArrowConverter, ArrowField)>,
    pub compat_level: CompatLevel,
    pub dictionary_id_offsets: Arc<[usize]>,
    pub write_options: WriteOptions,
    // Unstable.
    pub write_statistics_flags: bool,
    pub bytes_bufferer_config: BytesBuffererConfig,
    pub finish_record_batch_write_wg: WaitGroup,
}

impl RecordBatchEncoder {
    pub async fn run(self) -> PolarsResult<()> {
        let RecordBatchEncoder {
            mut morsel_rx,
            ipc_batch_tx,
            mut arrow_converters,
            compat_level,
            dictionary_id_offsets,
            write_options,
            write_statistics_flags,
            bytes_bufferer_config,
            finish_record_batch_write_wg,
        } = self;

        let compression = write_options.compression;

        while let Ok(morsel) = morsel_rx.recv().await {
            let (df, permit) = morsel.into_inner();
            let height = df.height();
            let columns = df.into_columns();
            let flags = write_statistics_flags.then(|| {
                columns
                    .iter()
                    .map(|c| c.get_flags().bits())
                    .collect::<Vec<_>>()
            });
            let custom_metadata = flags.map(|flags| {
                vec![schema::key_value(
                    IPC_RW_RECORD_BATCH_FLAGS_KEY,
                    serde_json::to_string(&flags).unwrap(),
                )]
            });

            assert_eq!(arrow_converters.len(), columns.len());

            let mut record_batch_arrow_arrays: Vec<Box<dyn Array>> =
                Vec::with_capacity(arrow_converters.len());

            // Rechunk and convert to arrow in parallel.
            for fut in parallelize_first_to_local(
                TaskPriority::High,
                columns.into_iter().zip(arrow_converters.drain(..)).map(
                    |(column, (mut arrow_converter, arrow_field))| async move {
                        let rechunked = column.as_materialized_series().rechunk();
                        let dtype = rechunked.dtype();

                        let array: Box<dyn Array> = arrow_converter
                            .array_to_arrow(
                                rechunked.chunks()[0].as_ref(),
                                dtype,
                                Cow::Borrowed(&arrow_field),
                            )
                            .unwrap();

                        (array, (arrow_converter, arrow_field))
                    },
                ),
            ) {
                let (array, arrow_converter_and_field) = fut.await;
                arrow_converters.push(arrow_converter_and_field);
                record_batch_arrow_arrays.push(array);
            }

            let bytes_bufferer_config = bytes_bufferer_config.clone();

            let serialize_handle =
                executor::AbortOnDropHandle::new(executor::spawn(TaskPriority::High, async move {
                    let mut arrow_data = BytesBufferer::new(&bytes_bufferer_config);
                    let mut ipc_message = BytesBufferer::new(&bytes_bufferer_config);
                    let mut ctx = IpcBatchSerializationContext::new(
                        &mut ipc_message,
                        &mut arrow_data,
                        compression,
                    );

                    for array in record_batch_arrow_arrays {
                        write_array(&mut ctx, array.as_ref())?;
                    }

                    let continuation_bytes =
                        finish_encode_ipc_record_batch(&mut ctx, height, custom_metadata)?;

                    let message_num_bytes = ipc_message.len();
                    let arrow_data_num_bytes = arrow_data.len();

                    Ok(IpcBatch {
                        batch_type: IpcBatchType::Record,
                        num_rows: height as IdxSize,
                        continuation_bytes,
                        message_bytes: ipc_message.into_iter(),
                        message_num_bytes,
                        arrow_data_bytes: arrow_data.into_iter(),
                        arrow_data_num_bytes,
                        morsel_permit: Some(permit),
                    })
                }));

            // Wait -> acquire here applies backpressure from slow I/O.
            // However, if I/O is faster than record batch encoding, then acquiring a token here
            // will unnecessarily restrict record batch encoding parallelism to 2 threads.
            let token = if compression.is_none() {
                // Uncompressed: Encoding should be 0-copy and complete instantly.
                finish_record_batch_write_wg.wait().await;
                Some(finish_record_batch_write_wg.token())
            } else {
                // Comperssed: Encoding will require CPU effort, allow parallelism across all threads.
                None
            };

            if ipc_batch_tx.send((serialize_handle, token)).await.is_err() {
                return Ok(());
            }
        }

        for encode_fut in arrow_converters
            .into_iter()
            .zip(dictionary_id_offsets.iter().copied())
            .filter(|((arrow_converter, _), _)| {
                !arrow_converter.categorical_converter.converters.is_empty()
            })
            .flat_map(|((arrow_converter, _), dictionary_id_offset)| {
                let bytes_bufferer_config = bytes_bufferer_config.clone();

                arrow_converter
                    .categorical_converter
                    .converters
                    .into_iter()
                    .enumerate()
                    .map(move |(i, (_, categorical_converter))| {
                        let bytes_bufferer_config = bytes_bufferer_config.clone();

                        async move {
                            let mut arrow_data = BytesBufferer::new(&bytes_bufferer_config);
                            let mut ipc_message = BytesBufferer::new(&bytes_bufferer_config);
                            let mut ctx = IpcBatchSerializationContext::new(
                                &mut ipc_message,
                                &mut arrow_data,
                                compression,
                            );

                            let categorical_values_array = categorical_converter
                                .build_values_array(compat_level.uses_binview_types());

                            write_array(&mut ctx, categorical_values_array.as_ref())?;

                            let continuation_bytes = finish_encode_ipc_dictionary_batch(
                                &mut ctx,
                                categorical_values_array.len(),
                                i64::try_from(i + dictionary_id_offset).unwrap(),
                            )?;

                            let message_num_bytes = ipc_message.len();
                            let arrow_data_num_bytes = arrow_data.len();

                            Ok(IpcBatch {
                                batch_type: IpcBatchType::Dictionary,
                                num_rows: categorical_values_array.len() as IdxSize,
                                continuation_bytes,
                                message_bytes: ipc_message.into_iter(),
                                message_num_bytes,
                                arrow_data_bytes: arrow_data.into_iter(),
                                arrow_data_num_bytes,
                                morsel_permit: None,
                            })
                        }
                    })
            })
        {
            let encode_handle =
                executor::AbortOnDropHandle::new(executor::spawn(TaskPriority::High, encode_fut));

            if ipc_batch_tx.send((encode_handle, None)).await.is_err() {
                return Ok(());
            }
        }

        Ok(())
    }
}
