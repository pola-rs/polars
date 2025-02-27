use std::cmp::Reverse;
use std::io::BufWriter;
use std::path::PathBuf;

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

use super::SinkNode;
use crate::async_executor::spawn;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::{JoinHandle, TaskPriority};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

type Linearized = Priority<Reverse<u64>, (Vec<EncodedData>, EncodedData)>;
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
        let mut recv_ports_recv = recv_ports_recv.serial();
        // Buffer task -> Encode tasks
        let (mut distribute, distribute_channels) =
            distributor_channel(num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        // Encode tasks -> IO task
        let (mut linearizer, senders) =
            Linearizer::<Linearized>::new(num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);

        let options = WriteOptions {
            compression: self.write_options.compression.map(Into::into),
        };

        let input_schema = self.input_schema.clone();
        let compat_level = self.compat_level;
        let chunk_size = self.chunk_size;

        // Buffer task.
        //
        // This task linearizes and buffers morsels until a given a maximum chunk size is reached
        // and then sends the whole record batch to be encoded and written.
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut seq = 0;
            let mut buffer = DataFrame::empty_with_schema(input_schema.as_ref());
            let mut dictionary_tracker = DictionaryTracker {
                dictionaries: Default::default(),
                cannot_replace: false,
            };

            // Search for Dictionary fields and which need to handled in special ways when encoding
            // IPC.
            let ipc_fields = input_schema
                .iter_fields()
                .map(|f| f.to_arrow(compat_level))
                .collect::<Vec<_>>();
            let ipc_fields = default_ipc_fields(ipc_fields.iter());
            let dict_columns_idxs = ipc_fields
                .iter()
                .enumerate()
                .filter_map(|(i, f)| f.contains_dictionary().then_some(i))
                .collect::<Vec<_>>();

            while let Ok((outcome, port)) = recv_ports_recv.recv().await {
                let mut receiver = port.serial();
                while let Ok(morsel) = receiver.recv().await {
                    let df = morsel.into_df();
                    // @NOTE: This also performs schema validation.
                    buffer.vstack_mut(&df)?;

                    while buffer.height() >= chunk_size {
                        let df;
                        (df, buffer) = buffer.split_at(buffer.height().min(chunk_size) as i64);

                        // We want to rechunk for two reasons:
                        // 1. the IPC writer expects aligned column chunks
                        // 2. the IPC writer turns chunks / record batches into chunks in the file,
                        //    so we want to respect the given `chunk_size`.
                        //
                        // This also properly sets the inner types of the record batches, which is
                        // important for dictionary and nested type encoding.
                        let record_batch = df.rechunk_to_record_batch(compat_level);

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
                }

                outcome.stopped();
            }

            // Flush the remaining rows.
            while buffer.height() > 0 {
                let df;
                (df, buffer) = buffer.split_at(buffer.height().min(chunk_size) as i64);

                // We want to rechunk for two reasons:
                // 1. the IPC writer expects aligned column chunks
                // 2. the IPC writer turns chunks / record batches into chunks in the file,
                //    so we want to respect the given `chunk_size`.
                //
                // This also properly sets the inner types of the record batches, which is
                // important for dictionary and nested type encoding.
                let record_batch = df.rechunk_to_record_batch(compat_level);

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

            PolarsResult::Ok(())
        }));

        // Encoding task.
        //
        // Task encodes the buffered record batch and sends it to be written to the file.
        for (mut receiver, mut sender) in distribute_channels.into_iter().zip(senders) {
            join_handles.push(spawn(TaskPriority::High, async move {
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

            // Linearize from all the Encoder tasks.
            while let Some(encoded_data) = linearizer.get().await {
                let (dicts, record_batch) = encoded_data.1;

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
