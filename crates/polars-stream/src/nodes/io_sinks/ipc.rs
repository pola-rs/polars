use std::cmp::Reverse;
use std::io::BufWriter;

use polars_core::schema::{SchemaExt, SchemaRef};
use polars_core::utils::arrow;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::io::ipc::write::{
    DictionaryTracker, EncodedData, WriteOptions, commit_encoded_arrays, default_ipc_fields,
    encode_array, encode_new_dictionaries,
};
use polars_error::PolarsResult;
use polars_io::SerWriter;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::{IpcWriter, IpcWriterOptions};
use polars_plan::dsl::{SinkOptions, SinkTarget};
use polars_utils::priority::Priority;

use super::{
    DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_SINK_LINEARIZER_BUFFER_SIZE, SinkInputPort,
    SinkNode, buffer_and_distribute_columns_task,
};
use crate::async_executor::spawn;
use crate::async_primitives::connector::{Receiver, connector};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::execute::StreamingExecutionState;
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::{JoinHandle, TaskPriority};

pub struct IpcSinkNode {
    target: SinkTarget,

    input_schema: SchemaRef,
    write_options: IpcWriterOptions,
    sink_options: SinkOptions,
    cloud_options: Option<CloudOptions>,
}

impl IpcSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        target: SinkTarget,
        sink_options: SinkOptions,
        write_options: IpcWriterOptions,
        cloud_options: Option<CloudOptions>,
    ) -> Self {
        Self {
            target,

            input_schema,
            write_options,
            sink_options,
            cloud_options,
        }
    }
}

impl SinkNode for IpcSinkNode {
    fn name(&self) -> &str {
        "ipc-sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        false
    }
    fn do_maintain_order(&self) -> bool {
        self.sink_options.maintain_order
    }

    fn spawn_sink(
        &mut self,
        recv_port_rx: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        // Buffer task -> Encode tasks
        let (dist_tx, dist_rxs) =
            distributor_channel(state.num_pipelines, *DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE);
        // Encode tasks -> Collect task
        let (mut lin_rx, lin_txs) =
            Linearizer::new(state.num_pipelines, *DEFAULT_SINK_LINEARIZER_BUFFER_SIZE);
        // Collect task -> IO task
        let (mut io_tx, mut io_rx) = connector::<(Vec<EncodedData>, EncodedData)>();

        let options = WriteOptions {
            compression: self.write_options.compression.map(Into::into),
        };

        let chunk_size = self.write_options.chunk_size;

        let ipc_fields = self
            .input_schema
            .iter_fields()
            .map(|f| f.to_arrow(self.write_options.compat_level))
            .collect::<Vec<_>>();
        let ipc_fields = default_ipc_fields(ipc_fields.iter());

        // Buffer task.
        join_handles.push(buffer_and_distribute_columns_task(
            recv_port_rx,
            dist_tx,
            chunk_size as usize,
            self.input_schema.clone(),
        ));

        // Encoding tasks.
        //
        // Task encodes the buffered record batch and sends it to be written to the file.
        join_handles.extend(
            dist_rxs
                .into_iter()
                .zip(lin_txs)
                .map(|(mut dist_rx, mut lin_tx)| {
                    let write_options = self.write_options;
                    spawn(TaskPriority::High, async move {
                        while let Ok((seq, col_idx, column)) = dist_rx.recv().await {
                            let mut variadic_buffer_counts = Vec::new();
                            let mut buffers = Vec::new();
                            let mut arrow_data = Vec::new();
                            let mut nodes = Vec::new();
                            let mut offset = 0;

                            // We want to rechunk for two reasons:
                            // 1. the IPC writer expects aligned column chunks
                            // 2. the IPC writer turns chunks / record batches into chunks in the file,
                            //    so we want to respect the given `chunk_size`.
                            //
                            // This also properly sets the inner types of the record batches, which is
                            // important for dictionary and nested type encoding.
                            let array = column.rechunk_to_arrow(write_options.compat_level);

                            // Encode array.
                            encode_array(
                                &array,
                                &options,
                                &mut variadic_buffer_counts,
                                &mut buffers,
                                &mut arrow_data,
                                &mut nodes,
                                &mut offset,
                            );

                            // Send the encoded data to the IO task.
                            let msg = Priority(
                                Reverse(seq),
                                (
                                    col_idx,
                                    array,
                                    variadic_buffer_counts,
                                    buffers,
                                    arrow_data,
                                    nodes,
                                    offset,
                                ),
                            );
                            if lin_tx.insert(msg).await.is_err() {
                                return Ok(());
                            }
                        }

                        PolarsResult::Ok(())
                    })
                }),
        );

        // Collect Task.
        //
        // Collects all the encoded data and packs it together for the IO task to write it.
        let input_schema = self.input_schema.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut dictionary_tracker = DictionaryTracker {
                dictionaries: Default::default(),
                cannot_replace: false,
            };

            struct CurrentColumn {
                array: Box<dyn Array>,
                variadic_buffer_counts: Vec<i64>,
                buffers: Vec<arrow::io::ipc::format::ipc::Buffer>,
                arrow_data: Vec<u8>,
                nodes: Vec<arrow::io::ipc::format::ipc::FieldNode>,
                offset: i64,
            }
            struct Current {
                seq: usize,
                height: usize,
                num_columns_seen: usize,
                columns: Vec<Option<CurrentColumn>>,
                encoded_dictionaries: Vec<EncodedData>,
            }

            let mut current = Current {
                seq: 0,
                height: 0,
                num_columns_seen: 0,
                columns: (0..input_schema.len()).map(|_| None).collect(),
                encoded_dictionaries: Vec::new(),
            };

            // Linearize from all the Encoder tasks.
            while let Some(Priority(
                Reverse(seq),
                (i, array, variadic_buffer_counts, buffers, arrow_data, nodes, offset),
            )) = lin_rx.get().await
            {
                if current.num_columns_seen == 0 {
                    current.seq = seq;
                    current.height = array.len();
                }

                debug_assert_eq!(current.seq, seq);
                debug_assert_eq!(current.height, array.len());
                debug_assert!(current.columns[i].is_none());
                current.columns[i] = Some(CurrentColumn {
                    array,
                    variadic_buffer_counts,
                    buffers,
                    arrow_data,
                    nodes,
                    offset,
                });
                current.num_columns_seen += 1;

                if current.num_columns_seen == input_schema.len() {
                    // @Optimize: Keep track of these sizes so we can correctly preallocate
                    // them.
                    let mut variadic_buffer_counts = Vec::new();
                    let mut buffers = Vec::new();
                    let mut arrow_data = Vec::new();
                    let mut nodes = Vec::new();
                    let mut offset = 0;

                    for (i, column) in current.columns.iter_mut().enumerate() {
                        let column = column.take().unwrap();

                        // @Optimize: It would be nice to do this on the Encode Tasks, but it is
                        // difficult to centralize the dictionary tracker like that.
                        //
                        // If there are dictionaries, we might need to emit the original dictionary
                        // definitions or dictionary deltas. We have precomputed which columns contain
                        // dictionaries and only check those columns.
                        encode_new_dictionaries(
                            &ipc_fields[i],
                            column.array.as_ref(),
                            &options,
                            &mut dictionary_tracker,
                            &mut current.encoded_dictionaries,
                        )?;

                        variadic_buffer_counts.extend(column.variadic_buffer_counts);
                        buffers.extend(column.buffers.into_iter().map(|mut b| {
                            // @NOTE: We need to offset all the buffers by the prefix sum of the
                            // column offsets.
                            b.offset += offset;
                            b
                        }));
                        arrow_data.extend(column.arrow_data);
                        nodes.extend(column.nodes);

                        offset += column.offset;
                    }

                    let mut encoded_data = EncodedData {
                        ipc_message: Vec::new(),
                        arrow_data,
                    };
                    commit_encoded_arrays(
                        current.height,
                        &options,
                        variadic_buffer_counts,
                        buffers,
                        nodes,
                        &mut encoded_data,
                    );

                    if io_tx
                        .send((
                            std::mem::take(&mut current.encoded_dictionaries),
                            encoded_data,
                        ))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    current.num_columns_seen = 0;
                }
            }

            Ok(())
        }));

        // IO task.
        //
        // Task that will actually do write to the target file.
        let target = self.target.clone();
        let sink_options = self.sink_options.clone();
        let write_options = self.write_options;
        let cloud_options = self.cloud_options.clone();
        let input_schema = self.input_schema.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            let mut file = target
                .open_into_writeable_async(&sink_options, cloud_options.as_ref())
                .await?;
            let writer = BufWriter::new(&mut *file);
            let mut writer = IpcWriter::new(writer)
                .with_compression(write_options.compression)
                .with_parallel(false)
                .batched(&input_schema)?;

            while let Ok((dicts, record_batch)) = io_rx.recv().await {
                // @TODO: At the moment this is a sync write, this is not ideal because we can only
                // have so many blocking threads in the tokio threadpool.
                writer.write_encoded(dicts.as_slice(), &record_batch)?;
            }

            writer.finish()?;
            drop(writer);

            file.sync_on_close(sink_options.sync_on_close)?;
            file.close()?;

            PolarsResult::Ok(())
        });
        join_handles.push(spawn(TaskPriority::Low, async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }));
    }
}
