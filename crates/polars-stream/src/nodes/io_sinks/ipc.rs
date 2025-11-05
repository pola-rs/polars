use std::cmp::Reverse;
use std::io::BufWriter;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use arrow::io::ipc::IpcField;
use arrow::io::ipc::write::encode_dictionary_values;
use polars_core::prelude::{CategoricalMapping, Column, DataType};
use polars_core::schema::SchemaRef;
use polars_core::series::ToArrowConverter;
use polars_core::series::categorical_to_arrow::CategoricalToArrowConverter;
use polars_core::utils::arrow;
use polars_core::utils::arrow::io::ipc::write::{
    EncodedData, WriteOptions, commit_encoded_arrays, encode_array,
};
use polars_error::PolarsResult;
use polars_io::SerWriter;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::{IpcWriter, IpcWriterOptions};
use polars_plan::dsl::{SinkOptions, SinkTarget};
use polars_utils::UnitVec;
use polars_utils::priority::Priority;

use super::{
    DEFAULT_SINK_LINEARIZER_BUFFER_SIZE, SinkInputPort, SinkNode,
    buffer_and_distribute_columns_task,
};
use crate::async_executor::spawn;
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::linearizer::Linearizer;
use crate::execute::StreamingExecutionState;
use crate::nodes::io_sinks::SendBufferedMorsel;
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::{JoinHandle, TaskPriority};
use crate::utils::task_handles_ext::AbortOnDropHandle;

pub struct IpcSinkNode {
    target: SinkTarget,

    input_schema: SchemaRef,
    write_options: IpcWriterOptions,
    sink_options: SinkOptions,
    cloud_options: Option<CloudOptions>,

    /// mpsc - each column encode worker may need to send in dictionary batches.
    io_tx: Option<tokio::sync::mpsc::Sender<IpcBatch>>,
    io_task: Option<AbortOnDropHandle<PolarsResult<()>>>,
    /// Arrow converters per-column.
    ///
    /// Categorical arrays that share the same underlying mapping will share the
    /// same dictionary in the IPC file if they are under the same top-level column.
    arrow_converters: Option<Vec<ToArrowConverter>>,
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

            io_tx: None,
            io_task: None,
            arrow_converters: None,
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

    fn initialize(&mut self, _state: &StreamingExecutionState) -> PolarsResult<()> {
        // Collect task -> IO task
        let (io_tx, mut io_rx) = tokio::sync::mpsc::channel::<IpcBatch>(1);

        // IO task.
        //
        // Task that will actually do write to the target file.
        let target = self.target.clone();
        let sink_options = self.sink_options.clone();
        let write_options = self.write_options;
        let cloud_options = self.cloud_options.clone();
        let input_schema = self.input_schema.clone();

        let compat_level = self.write_options.compat_level;
        let arrow_converters: Vec<ToArrowConverter> = self
            .input_schema
            .iter_values()
            .map(|dtype| {
                let mut categorical_converter = CategoricalToArrowConverter {
                    converters: Default::default(),
                    persist_remap: true,
                    output_keys_only: true,
                };
                categorical_converter.initialize(dtype);
                ToArrowConverter {
                    compat_level,
                    categorical_converter,
                }
            })
            .collect();

        // Note: `arrow_converters` must not be mutated after this point. This is to ensure that
        // the dictionary IDs used by the encoders matches those assigned to the fields here.
        let ipc_fields: Vec<IpcField> = self
            .input_schema
            .iter_values()
            .zip(&arrow_converters)
            .zip(dictionary_id_offsets_iter(&arrow_converters))
            .map(|((dtype, arrow_converter), dictionary_id_offset)| {
                IpcFieldConverter {
                    get_dictionary_id: |mapping: &Arc<CategoricalMapping>| {
                        let converter_key: usize = Arc::as_ptr(mapping) as *const () as _;
                        let converter_index: usize = arrow_converter
                            .categorical_converter
                            .converters
                            .get_index_of(&converter_key)
                            .unwrap();

                        i64::try_from(dictionary_id_offset + converter_index).unwrap()
                    },
                }
                .dtype_to_ipc_field(dtype)
            })
            .collect();

        self.arrow_converters = Some(arrow_converters);

        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            let mut file = target
                .open_into_writeable_async(&sink_options, cloud_options.as_ref())
                .await?;
            let writer = BufWriter::new(&mut *file);
            let mut writer = IpcWriter::new(writer)
                .with_compression(write_options.compression)
                .with_compat_level(write_options.compat_level)
                .with_parallel(false)
                .batched(&input_schema, ipc_fields)?;

            while let Some(encoded_data) = io_rx.recv().await {
                // @TODO: At the moment this is a sync write, this is not ideal because we can only
                // have so many blocking threads in the tokio threadpool.
                match encoded_data {
                    IpcBatch::Record(encoded_data) => writer.write_encoded(&[], &encoded_data)?,
                    IpcBatch::Dictionary(dictionaries) => {
                        writer.write_encoded_dictionaries(&dictionaries)?
                    },
                }
            }

            writer.finish()?;
            drop(writer);

            file.sync_on_close(sink_options.sync_on_close)?;
            file.close()?;

            PolarsResult::Ok(())
        });

        self.io_tx = Some(io_tx);
        self.io_task = Some(AbortOnDropHandle(io_task));

        Ok(())
    }

    fn spawn_sink(
        &mut self,
        recv_port_rx: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let channel_capacity = state.num_pipelines.div_ceil(self.input_schema.len().max(1));

        // Buffer task -> Encode tasks
        #[expect(clippy::type_complexity)]
        let (col_txs, col_rxs): (
            // (seq_id, column)
            Vec<tokio::sync::mpsc::Sender<(usize, Column)>>,
            Vec<tokio::sync::mpsc::Receiver<(usize, Column)>>,
        ) = (0..self.input_schema.len())
            .map(|_| tokio::sync::mpsc::channel(channel_capacity))
            .unzip();
        // Encode tasks -> Collect task
        let (mut lin_rx, lin_txs) = Linearizer::new(
            self.input_schema.len(),
            *DEFAULT_SINK_LINEARIZER_BUFFER_SIZE,
        );
        // Collect task -> IO task
        let io_tx = self
            .io_tx
            .take()
            .expect("not initialized / spawn called more than once");
        let arrow_converters = self.arrow_converters.take().unwrap();

        let options = WriteOptions {
            compression: self.write_options.compression.map(Into::into),
        };

        let chunk_size = self.write_options.chunk_size;

        // Buffer task.
        join_handles.push(buffer_and_distribute_columns_task(
            recv_port_rx,
            SendBufferedMorsel::PerColumn(col_txs),
            chunk_size as usize,
            self.input_schema.clone(),
            Arc::new(Mutex::new(None)),
        ));

        let dictionary_id_offsets: Vec<usize> =
            dictionary_id_offsets_iter(&arrow_converters).collect();

        // Encoding tasks.
        //
        // Task encodes the buffered record batch and sends it to be written to the file.
        join_handles.extend(
            col_rxs
                .into_iter()
                .zip(lin_txs)
                .zip(arrow_converters)
                .zip(dictionary_id_offsets)
                .enumerate()
                .map(
                    |(
                        col_idx,
                        (((mut dist_rx, mut lin_tx), mut arrow_converter), dictionary_id_offset),
                    )| {
                        let io_tx = io_tx.clone();

                        spawn(TaskPriority::High, async move {
                            while let Some((seq, column)) = dist_rx.recv().await {
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
                                let rechunked = column.take_materialized_series().rechunk();

                                let array = arrow_converter.array_to_arrow(
                                    rechunked.chunks()[0].as_ref(),
                                    rechunked.dtype(),
                                );

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

                            // Finished consuming all incoming morsels. Now construct the full
                            // dictionaries for the categorical mappings present in this column.
                            if !arrow_converter.categorical_converter.converters.is_empty() {
                                let mut encoded_dictionaries: UnitVec<EncodedData> =
                                    UnitVec::with_capacity(
                                        arrow_converter.categorical_converter.converters.len(),
                                    );

                                for (i, converter) in arrow_converter
                                    .categorical_converter
                                    .converters
                                    .values()
                                    .enumerate()
                                {
                                    encoded_dictionaries.push(encode_dictionary_values(
                                        i64::try_from(i + dictionary_id_offset).unwrap(),
                                        converter
                                            .build_values_array(arrow_converter.compat_level)
                                            .as_ref(),
                                        &options,
                                    )?);
                                }

                                let _ =
                                    io_tx.send(IpcBatch::Dictionary(encoded_dictionaries)).await;
                            }

                            PolarsResult::Ok(())
                        })
                    },
                ),
        );

        // Collect Task.
        //
        // Collects all the encoded data and packs it together for the IO task to write it.
        let input_schema = self.input_schema.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            struct CurrentColumn {
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
            }

            let mut current = Current {
                seq: 0,
                height: 0,
                num_columns_seen: 0,
                columns: (0..input_schema.len()).map(|_| None).collect(),
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

                    for column in current.columns.iter_mut() {
                        let column = column.take().unwrap();

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

                    if io_tx.send(IpcBatch::Record(encoded_data)).await.is_err() {
                        return Ok(());
                    }
                    current.num_columns_seen = 0;
                }
            }

            Ok(())
        }));
    }

    fn finalize(
        &mut self,
        _state: &StreamingExecutionState,
    ) -> Option<Pin<Box<dyn Future<Output = PolarsResult<()>> + Send>>> {
        // If we were never spawned, we need to make sure that the `tx` is taken. This signals to
        // the IO task that it is done and prevents deadlocks.
        drop(self.io_tx.take());

        let io_task = self
            .io_task
            .take()
            .expect("not initialized / finish called more than once");

        // Wait for the IO task to complete.
        Some(Box::pin(async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }))
    }
}

/// Cumulative sum, excluding the current element.
///
/// Indicates total number of dictionaries in the columns to the left of the current one.
fn dictionary_id_offsets_iter(
    arrow_converters: &[ToArrowConverter],
) -> impl Iterator<Item = usize> {
    arrow_converters
        .iter()
        .scan(0, |acc: &mut usize, arrow_converter| {
            let out = *acc;
            *acc += arrow_converter.categorical_converter.converters.len();
            Some(out)
        })
}

enum IpcBatch {
    Record(EncodedData),
    Dictionary(UnitVec<EncodedData>),
}

struct IpcFieldConverter<F>
where
    F: Fn(&Arc<CategoricalMapping>) -> i64,
{
    get_dictionary_id: F,
}

impl<F> IpcFieldConverter<F>
where
    F: Fn(&Arc<CategoricalMapping>) -> i64,
{
    fn dtype_to_ipc_field(&self, dtype: &DataType) -> IpcField {
        use DataType::*;

        match dtype {
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, mapping) | Enum(_, mapping) => IpcField {
                fields: vec![self.dtype_to_ipc_field(&DataType::String)],
                dictionary_id: Some((self.get_dictionary_id)(mapping)),
            },
            List(inner) => IpcField {
                fields: vec![self.dtype_to_ipc_field(inner)],
                dictionary_id: None,
            },
            #[cfg(feature = "dtype-array")]
            Array(inner, _width) => IpcField {
                fields: vec![self.dtype_to_ipc_field(inner)],
                dictionary_id: None,
            },
            Struct(fields) => IpcField {
                fields: fields
                    .iter()
                    .map(|x| self.dtype_to_ipc_field(x.dtype()))
                    .collect(),
                dictionary_id: None,
            },
            _ => {
                assert!(!dtype.is_nested());
                IpcField {
                    fields: vec![],
                    dictionary_id: None,
                }
            },
        }
    }
}
