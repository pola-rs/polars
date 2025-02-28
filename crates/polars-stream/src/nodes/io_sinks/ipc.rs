use std::cmp::Reverse;
use std::io::BufWriter;
use std::path::PathBuf;

use polars_core::prelude::CompatLevel;
use polars_core::schema::{SchemaExt, SchemaRef};
use polars_core::utils::arrow;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::io::ipc::write::{
    commit_encoded_arrays, default_ipc_fields, encode_array, encode_new_dictionaries,
    DictionaryTracker, EncodedData, WriteOptions,
};
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::ipc::{IpcWriter, IpcWriterOptions};
use polars_io::SerWriter;
use polars_utils::priority::Priority;

use super::{
    buffer_and_distribute_columns_task, SinkNode, DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE,
    DEFAULT_SINK_LINEARIZER_BUFFER_SIZE,
};
use crate::async_executor::spawn;
use crate::async_primitives::connector::connector;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::{JoinHandle, TaskPriority};

pub struct IpcSinkNode {
    path: PathBuf,

    input_schema: SchemaRef,
    write_options: IpcWriterOptions,

    compat_level: CompatLevel,

    chunk_size: usize,
}

impl IpcSinkNode {
    pub fn new(input_schema: SchemaRef, path: PathBuf, write_options: IpcWriterOptions) -> Self {
        Self {
            path,

            input_schema,
            write_options,

            compat_level: CompatLevel::newest(), // @TODO: make this accessible from outside

            chunk_size: get_ideal_morsel_size(), // @TODO: change to something more appropriate
        }
    }
}

impl SinkNode for IpcSinkNode {
    fn name(&self) -> &str {
        "ipc_sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        false
    }

    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: super::SinkRecvPort,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        // .. -> Buffer task
        let buffer_rx = recv_ports_recv.serial(join_handles);
        // Buffer task -> Encode tasks
        let (dist_tx, dist_rxs) =
            distributor_channel(num_pipelines, DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE);
        // Encode tasks -> Collect task
        let (mut lin_rx, lin_txs) =
            Linearizer::new(num_pipelines, DEFAULT_SINK_LINEARIZER_BUFFER_SIZE);
        // Collect task -> IO task
        let (mut io_tx, mut io_rx) = connector::<(Vec<EncodedData>, EncodedData)>();

        let options = WriteOptions {
            compression: self.write_options.compression.map(Into::into),
        };

        let compat_level = self.compat_level;
        let chunk_size = self.chunk_size;

        let ipc_fields = self
            .input_schema
            .iter_fields()
            .map(|f| f.to_arrow(compat_level))
            .collect::<Vec<_>>();
        let ipc_fields = default_ipc_fields(ipc_fields.iter());

        // Buffer task.
        join_handles.push(buffer_and_distribute_columns_task(
            buffer_rx,
            dist_tx,
            chunk_size,
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
                            let array = column.rechunk_to_arrow(compat_level);

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
        let path = self.path.clone();
        let write_options = self.write_options;
        let input_schema = self.input_schema.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::fs::OpenOptions;

            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await?;
            let writer = BufWriter::new(file.into_std().await);
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

            PolarsResult::Ok(())
        });
        join_handles.push(spawn(TaskPriority::Low, async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }));
    }
}
