use std::cmp::Reverse;
use std::future::Future;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use polars_core::frame::DataFrame;
use polars_core::prelude::{ArrowSchema, CompatLevel};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::parquet::write::BatchedWriter;
use polars_io::prelude::{get_encodings, ParquetWriteOptions};
use polars_io::schema_to_arrow_checked;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    array_to_columns, to_parquet_schema, CompressedPage, Compressor, Encoding, FileWriter,
    SchemaDescriptor, Version, WriteOptions,
};
use polars_utils::priority::Priority;

use super::partition::PartionableSinkNode;
use super::{SinkNode, SinkRecvPort};
use crate::async_executor::spawn;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::nodes::{JoinHandle, TaskPriority};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

type Linearized = Priority<Reverse<(usize, usize)>, Vec<Vec<CompressedPage>>>;
pub struct ParquetSinkNode {
    path: PathBuf,

    input_schema: SchemaRef,
    write_options: ParquetWriteOptions,

    parquet_schema: SchemaDescriptor,
    arrow_schema: ArrowSchema,
    encodings: Vec<Vec<Encoding>>,
}

impl ParquetSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        path: &Path,
        write_options: &ParquetWriteOptions,
    ) -> PolarsResult<Self> {
        let schema = schema_to_arrow_checked(&input_schema, CompatLevel::newest(), "parquet")?;
        let parquet_schema = to_parquet_schema(&schema)?;
        let encodings: Vec<Vec<Encoding>> = get_encodings(&schema);

        Ok(Self {
            path: path.to_path_buf(),

            input_schema,
            write_options: *write_options,

            parquet_schema,
            arrow_schema: schema,
            encodings,
        })
    }
}

// 512 ^ 2
const DEFAULT_ROW_GROUP_SIZE: usize = 1 << 18;

impl SinkNode for ParquetSinkNode {
    fn name(&self) -> &str {
        "parquet_sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        false
    }

    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
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

        let input_schema = self.input_schema.clone();
        let write_options = self.write_options;

        let options = WriteOptions {
            statistics: write_options.statistics,
            compression: write_options.compression.into(),
            version: Version::V1,
            data_page_size: write_options.data_page_size,
        };

        // Buffer task.
        //
        // This task linearizes and buffers morsels until a given a maximum chunk size is reached
        // and then sends the whole record batch to be encoded and written.
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut buffer = DataFrame::empty_with_schema(input_schema.as_ref());
            let row_group_size = write_options
                .row_group_size
                .unwrap_or(DEFAULT_ROW_GROUP_SIZE)
                .max(1);
            let mut stop_requested = false;
            let mut row_group_index = 0;

            while let Ok(input) = recv_ports_recv.recv().await {
                let mut receiver = input.port.serial();
                loop {
                    match receiver.recv().await {
                        Err(_) => stop_requested = true,
                        Ok(morsel) => {
                            let df = morsel.into_df();

                            // @NOTE: This also performs schema validation.
                            buffer.vstack_mut(&df)?;
                        },
                    }

                    while (stop_requested && buffer.height() > 0)
                        || buffer.height() >= row_group_size
                    {
                        let row_group;

                        (row_group, buffer) =
                            buffer.split_at(row_group_size.min(buffer.height()) as i64);

                        for (column_idx, column) in row_group.take_columns().into_iter().enumerate()
                        {
                            distribute
                                .send((row_group_index, column_idx, column))
                                .await
                                .unwrap();
                        }

                        row_group_index += 1;
                    }

                    if stop_requested {
                        break;
                    }
                }

                input.outcome.stop();
            }

            PolarsResult::Ok(())
        }));

        // Encode task.
        //
        // Task encodes the columns into their corresponding Parquet encoding.
        for (mut receiver, mut sender) in distribute_channels.into_iter().zip(senders.into_iter()) {
            let parquet_schema = self.parquet_schema.clone();
            let encodings = self.encodings.clone();

            join_handles.push(spawn(TaskPriority::High, async move {
                while let Ok((rg_idx, col_idx, column)) = receiver.recv().await {
                    let type_ = &parquet_schema.fields()[col_idx];
                    let encodings = &encodings[col_idx];

                    let array = column.as_materialized_series().rechunk();
                    let array = array.to_arrow(0, CompatLevel::newest());

                    // @TODO: This causes all structs fields to be handled on a single thread. It
                    // would be preferable to split the encoding among multiple threads.

                    // @NOTE: Since one Polars column might contain multiple Parquet columns (when
                    // it has a struct datatype), we return a Vec<Vec<CompressedPage>>.

                    // Array -> Parquet pages.
                    let encoded_columns =
                        array_to_columns(array, type_.clone(), options, encodings)?;

                    // Compress the pages.
                    let compressed_pages = encoded_columns
                        .into_iter()
                        .map(|encoded_pages| {
                            Compressor::new_from_vec(
                                encoded_pages.map(|result| {
                                    result.map_err(|e| {
                                        ParquetError::FeatureNotSupported(format!(
                                            "reraised in polars: {e}",
                                        ))
                                    })
                                }),
                                options.compression,
                                vec![],
                            )
                            .collect::<ParquetResult<Vec<_>>>()
                        })
                        .collect::<ParquetResult<Vec<_>>>()?;

                    sender
                        .insert(Priority(Reverse((rg_idx, col_idx)), compressed_pages))
                        .await
                        .unwrap();
                }

                PolarsResult::Ok(())
            }));
        }

        // IO task.
        //
        // Task that will actually do write to the target file. It is important that this is only
        // spawned once.
        let path = self.path.clone();
        let write_options = self.write_options;
        let input_schema = self.input_schema.clone();
        let arrow_schema = self.arrow_schema.clone();
        let parquet_schema = self.parquet_schema.clone();
        let encodings = self.encodings.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::fs::OpenOptions;

            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await
                .map_err(|err| polars_utils::_limit_path_len_io_err(path.as_path(), err))?;
            let file = file.into_std().await;
            let writer = BufWriter::new(file);
            let options = WriteOptions {
                statistics: write_options.statistics,
                compression: write_options.compression.into(),
                version: Version::V1,
                data_page_size: write_options.data_page_size,
            };
            let file_writer = Mutex::new(FileWriter::new_with_parquet_schema(
                writer,
                arrow_schema,
                parquet_schema,
                options,
            ));
            let mut writer = BatchedWriter::new(file_writer, encodings, options, false);

            let num_parquet_columns = writer.parquet_schema().leaves().len();
            let mut current_row_group = Vec::with_capacity(num_parquet_columns);

            // Linearize from all the Encoder tasks.
            while let Some(Priority(Reverse((_, col_idx)), compressed_pages)) =
                linearizer.get().await
            {
                assert!(col_idx < input_schema.len());
                current_row_group.extend(compressed_pages);

                // Only if it is the last column of the row group, write the row group to the file.
                if current_row_group.len() < num_parquet_columns {
                    continue;
                }

                // @TODO: At the moment this is a sync write, this is not ideal because we can only
                // have so many blocking threads in the tokio threadpool.
                assert_eq!(current_row_group.len(), num_parquet_columns);
                writer.write_row_group(&current_row_group)?;
                current_row_group.clear();
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

impl PartionableSinkNode for ParquetSinkNode {
    type SinkOptions = ParquetWriteOptions;

    fn new(
        path: &Path,
        input_schema: &SchemaRef,
        options: &Self::SinkOptions,
    ) -> impl Future<Output = PolarsResult<Self>> + Send + Sync {
        async move { Self::new(input_schema.clone(), path, options) }
    }

    fn key_to_path(
        keys: &[polars_core::prelude::AnyValue<'static>],
        options: &Self::SinkOptions,
    ) -> PolarsResult<PathBuf> {
        static CTR: AtomicUsize = AtomicUsize::new(0);

        // @Hack: Fix
        let mut pb = PathBuf::new();
        pb.push(&CTR.fetch_add(1, Ordering::Relaxed).to_string());
        pb.set_extension("parquet");
        Ok(pb)
    }
}
