use std::cmp::Reverse;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use polars_core::frame::DataFrame;
use polars_core::prelude::CompatLevel;
use polars_core::schema::{SchemaExt, SchemaRef};
use polars_core::utils::arrow::io::ipc::write::{
    default_ipc_fields, dictionaries_to_encode, encode_dictionary, encode_record_batch,
    DictionaryTracker, EncodedData, WriteOptions,
};
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::ipc::{IpcWriter, IpcWriterOptions};
use polars_io::SerWriter;
use polars_utils::priority::Priority;

use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::{ComputeNode, JoinHandle, PortState, TaskPriority, TaskScope};
use crate::pipe::{RecvPort, SendPort};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub struct IpcSinkNode {
    path: PathBuf,

    input_schema: SchemaRef,
    write_options: IpcWriterOptions,

    compat_level: CompatLevel,

    num_encoders: usize,
    chunk_size: usize,
}

impl IpcSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        path: &Path,
        write_options: &IpcWriterOptions,
    ) -> PolarsResult<Self> {
        Ok(Self {
            path: path.to_path_buf(),

            input_schema,
            write_options: *write_options,

            compat_level: CompatLevel::newest(), // @TODO: make this accessible from outside

            num_encoders: 1,
            chunk_size: get_ideal_morsel_size(), // @TODO: change to something more appropriate
        })
    }
}

impl ComputeNode for IpcSinkNode {
    fn name(&self) -> &str {
        "ipc_sink"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_encoders = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        // We are always ready to receive, unless the sender is done, then we're
        // also done.
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1);
        assert!(send_ports.is_empty());

        // .. -> Buffer task
        let mut receiver = recv_ports[0].take().unwrap().serial();
        // Buffer task -> Encode tasks
        let (mut distribute, distribute_channels) =
            distributor_channel(self.num_encoders, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        // Encode tasks -> IO task
        let (mut linearizer, senders) = Linearizer::<
            Priority<Reverse<u64>, (Vec<EncodedData>, EncodedData)>,
        >::new(
            self.num_encoders, DEFAULT_LINEARIZER_BUFFER_SIZE
        );

        let options = WriteOptions {
            compression: self.write_options.compression.map(Into::into),
        };

        let slf = &*self;

        // Buffer task.
        //
        // This task linearizes and buffers morsels until a given a maximum chunk size is reached
        // and then sends the whole record batch to be encoded and written.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut seq = 0;
            let mut buffer = DataFrame::empty_with_schema(slf.input_schema.as_ref());
            let mut dictionary_tracker = DictionaryTracker {
                dictionaries: Default::default(),
                cannot_replace: false,
            };

            // Search for Dictionary fields and which need to handled in special ways when encoding
            // IPC.
            let ipc_fields = slf
                .input_schema
                .iter_fields()
                .map(|f| f.to_arrow(slf.compat_level))
                .collect::<Vec<_>>();
            let ipc_fields = default_ipc_fields(ipc_fields.iter());
            let dict_columns_idxs = ipc_fields
                .iter()
                .enumerate()
                .filter_map(|(i, f)| f.contains_dictionary().then_some(i))
                .collect::<Vec<_>>();

            let mut stop_requested = false;
            loop {
                if buffer.height() >= slf.chunk_size || (buffer.height() > 0 && stop_requested) {
                    let df;
                    (df, buffer) = buffer.split_at(buffer.height().min(slf.chunk_size) as i64);

                    // We want to rechunk for two reasons:
                    // 1. the IPC writer expects aligned column chunks
                    // 2. the IPC writer turns chunks / record batches into chunks in the file,
                    //    so we want to respect the given `chunk_size`.
                    //
                    // This also properly sets the inner types of the record batches, which is
                    // important for dictionary and nested type encoding.
                    let record_batch = df.rechunk_to_record_batch(slf.compat_level);

                    // If there are dictionaries, we might need to emit the original dictionary
                    // definitions or dictionary deltas. We have precomputed which columns contain
                    // dictionaries and only check those columns.
                    let mut dicts_to_encode = Vec::new();
                    for &i in &dict_columns_idxs {
                        dictionaries_to_encode(
                            &ipc_fields[i],
                            record_batch.arrays()[i].as_ref(),
                            &mut dictionary_tracker,
                            &mut dicts_to_encode,
                        )?;
                    }

                    // Send of the dictionaries and record batch to be encoded by an Encoder
                    // task. This is compute heavy, so distribute the chunks.
                    let msg = (seq, dicts_to_encode, record_batch);
                    seq += 1;
                    if distribute.send(msg).await.is_err() {
                        break;
                    }
                }

                // If we have no more rows to write and there are no more morsels coming, we can
                // stop this task.
                if buffer.is_empty() && stop_requested {
                    break;
                }

                let Ok(morsel) = receiver.recv().await else {
                    stop_requested = true;
                    continue;
                };

                let df = morsel.into_df();
                // @NOTE: This also performs schema validation.
                buffer.vstack_mut(&df)?;
            }

            PolarsResult::Ok(())
        }));

        // Encoding task.
        //
        // Task encodes the buffered record batch and sends it to be written to the file.
        for (mut receiver, mut sender) in distribute_channels.into_iter().zip(senders) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok((seq, dicts_to_encode, record_batch)) = receiver.recv().await {
                    let mut encoded_dictionaries = Vec::new();
                    let mut encoded_message = EncodedData::default();

                    // Encode the dictionaries and record batch.
                    for (dict_id, dict_array) in dicts_to_encode {
                        encode_dictionary(
                            dict_id,
                            dict_array.as_ref(),
                            &options,
                            &mut encoded_dictionaries,
                        )?;
                    }
                    encode_record_batch(&record_batch, &options, &mut encoded_message);

                    // Send the encoded data to the IO task.
                    let msg = Priority(Reverse(seq), (encoded_dictionaries, encoded_message));
                    if sender.insert(msg).await.is_err() {
                        break;
                    }
                }

                PolarsResult::Ok(())
            }));
        }

        // IO task.
        //
        // Task that will actually do write to the target file.
        let io_runtime = polars_io::pl_async::get_runtime();

        let path = self.path.clone();
        let write_options = self.write_options;
        let input_schema = self.input_schema.clone();

        let io_task = io_runtime.spawn(async move {
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

            // Linearize from all the Encoder tasks.
            while let Some(encoded_data) = linearizer.get().await {
                let (dicts, record_batch) = encoded_data.1;
                writer.write_encoded(dicts.as_slice(), &record_batch)?;
            }

            writer.finish()?;

            PolarsResult::Ok(())
        });
        join_handles
            .push(scope.spawn_task(TaskPriority::Low, async move { io_task.await.unwrap() }));
    }
}
