use std::sync::Arc;

use arrow::array::Array;
use arrow::io::ipc::write::encode_dictionary_values;
use polars_core::series::ToArrowConverter;
use polars_core::utils::arrow;
use polars_core::utils::arrow::io::ipc::write::{
    EncodedData, WriteOptions, commit_encoded_arrays, encode_array, schema,
};
use polars_error::PolarsResult;
use polars_utils::concat_vec::ConcatVec as _;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::opt_spawned_future::parallelize_first_to_local;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::writers::interface::IPC_RW_RECORD_BATCH_FLAGS_KEY;
use crate::nodes::io_sinks::writers::ipc::IpcBatch;

pub struct RecordBatchEncoder {
    pub morsel_rx: connector::Receiver<SinkMorsel>,
    pub ipc_batch_tx: tokio::sync::mpsc::Sender<IpcBatch>,
    pub arrow_converters: Vec<ToArrowConverter>,
    pub dictionary_id_offsets: Arc<[usize]>,
    pub write_options: WriteOptions,
    // Unstable.
    pub write_statistics_flags: bool,
}

impl RecordBatchEncoder {
    pub async fn run(self) -> PolarsResult<()> {
        let RecordBatchEncoder {
            mut morsel_rx,
            ipc_batch_tx,
            mut arrow_converters,
            dictionary_id_offsets,
            write_options,
            write_statistics_flags,
        } = self;

        let mut record_batch_arrow_arrays: Vec<Box<dyn Array>> =
            Vec::with_capacity(arrow_converters.len());

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

            assert!(record_batch_arrow_arrays.is_empty());
            assert_eq!(arrow_converters.len(), columns.len());

            // Rechunk and convert to arrow in parallel.
            for fut in parallelize_first_to_local(
                TaskPriority::High,
                columns.into_iter().zip(arrow_converters.drain(..)).map(
                    |(column, mut arrow_converter)| async move {
                        let rechunked = column.take_materialized_series().rechunk();
                        let dtype = rechunked.dtype();

                        let array: Box<dyn Array> = arrow_converter
                            .array_to_arrow(rechunked.chunks()[0].as_ref(), dtype, None)
                            .unwrap();

                        (array, arrow_converter)
                    },
                ),
            ) {
                let (array, arrow_converter) = fut.await;
                arrow_converters.push(arrow_converter);
                record_batch_arrow_arrays.push(array);
            }

            // Construct the iterator here so that the loop retains ownership of `record_batch_arrow_arrays`.
            let array_encode_fut_iter = parallelize_first_to_local(
                TaskPriority::High,
                record_batch_arrow_arrays.drain(..).map(|array| async move {
                    let mut out = EncodedArrayData::default();

                    let EncodedArrayData {
                        variadic_buffer_counts,
                        buffers,
                        arrow_data,
                        nodes,
                        offset,
                    } = &mut out;

                    encode_array(
                        &array,
                        &write_options,
                        variadic_buffer_counts,
                        buffers,
                        arrow_data,
                        nodes,
                        offset,
                    );

                    out
                }),
            );

            let array_combine_handle = async_executor::AbortOnDropHandle::new(
                async_executor::spawn(TaskPriority::High, async move {
                    let mut buffers: Vec<arrow::io::ipc::format::ipc::Buffer> = vec![];
                    let mut buffer_offset: i64 = 0;

                    let num_results = array_encode_fut_iter.len();
                    let mut variadic_buffer_counts: Vec<Vec<i64>> = Vec::with_capacity(num_results);
                    let mut arrow_data: Vec<Vec<u8>> = Vec::with_capacity(num_results);
                    let mut nodes: Vec<Vec<arrow::io::ipc::format::ipc::FieldNode>> =
                        Vec::with_capacity(num_results);

                    for fut in array_encode_fut_iter {
                        let v: EncodedArrayData = fut.await;

                        variadic_buffer_counts.push(v.variadic_buffer_counts);
                        arrow_data.push(v.arrow_data);
                        nodes.push(v.nodes);

                        buffers.extend(v.buffers.into_iter().map(|mut b| {
                            b.offset += buffer_offset;
                            b
                        }));

                        buffer_offset += v.offset;
                    }

                    let variadic_buffer_counts = variadic_buffer_counts.concat_vec();
                    let arrow_data = arrow_data.concat_vec();
                    let nodes = nodes.concat_vec();

                    let mut encoded_data = EncodedData {
                        ipc_message: Vec::new(),
                        arrow_data,
                    };

                    commit_encoded_arrays(
                        height,
                        &write_options,
                        variadic_buffer_counts,
                        buffers,
                        nodes,
                        custom_metadata,
                        &mut encoded_data,
                    );

                    encoded_data
                }),
            );

            if ipc_batch_tx
                .send(IpcBatch::Record(array_combine_handle, permit))
                .await
                .is_err()
            {
                return Ok(());
            }
        }

        for fut in parallelize_first_to_local(
            TaskPriority::High,
            arrow_converters
                .into_iter()
                .zip(dictionary_id_offsets.iter().copied())
                .filter(|(arrow_converter, _)| {
                    !arrow_converter.categorical_converter.converters.is_empty()
                })
                .flat_map(|(arrow_converter, dictionary_id_offset)| {
                    let ipc_batch_tx = ipc_batch_tx.clone();

                    arrow_converter
                        .categorical_converter
                        .converters
                        .into_iter()
                        .enumerate()
                        .map(move |(i, (_, categorical_converter))| {
                            let ipc_batch_tx = ipc_batch_tx.clone();

                            async move {
                                let encoded_data = encode_dictionary_values(
                                    i64::try_from(i + dictionary_id_offset).unwrap(),
                                    categorical_converter
                                        .build_values_array(arrow_converter.compat_level)
                                        .as_ref(),
                                    &write_options,
                                )?;

                                // Dictionary batches can be placed anywhere in an IPC file, so we
                                // can have each task send their encoded data as soon as it's ready.
                                let _ = ipc_batch_tx.send(IpcBatch::Dictionary(encoded_data)).await;

                                PolarsResult::Ok(())
                            }
                        })
                }),
        ) {
            fut.await?;
        }

        Ok(())
    }
}

#[derive(Default)]
struct EncodedArrayData {
    variadic_buffer_counts: Vec<i64>,
    buffers: Vec<arrow::io::ipc::format::ipc::Buffer>,
    arrow_data: Vec<u8>,
    nodes: Vec<arrow::io::ipc::format::ipc::FieldNode>,
    offset: i64,
}
