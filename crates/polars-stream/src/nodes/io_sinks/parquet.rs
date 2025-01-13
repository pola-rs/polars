use std::cmp::Reverse;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
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

use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::nodes::{ComputeNode, JoinHandle, PortState, TaskPriority, TaskScope};
use crate::pipe::{RecvPort, SendPort};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub struct ParquetSinkNode {
    path: PathBuf,

    input_schema: SchemaRef,
    write_options: ParquetWriteOptions,

    parquet_schema: SchemaDescriptor,
    arrow_schema: ArrowSchema,
    encodings: Vec<Vec<Encoding>>,

    num_encoders: usize,
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

            num_encoders: 1,
        })
    }
}

// 512 ^ 2
const DEFAULT_ROW_GROUP_SIZE: usize = 1 << 18;

impl ComputeNode for ParquetSinkNode {
    fn name(&self) -> &str {
        "parquet_sink"
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
            Priority<Reverse<(usize, usize)>, Vec<Vec<CompressedPage>>>,
        >::new(
            self.num_encoders, DEFAULT_LINEARIZER_BUFFER_SIZE
        );

        let slf = &*self;

        let options = WriteOptions {
            statistics: slf.write_options.statistics,
            compression: slf.write_options.compression.into(),
            version: Version::V1,
            data_page_size: slf.write_options.data_page_size,
        };

        // Buffer task.
        //
        // This task linearizes and buffers morsels until a given a maximum chunk size is reached
        // and then sends the whole record batch to be encoded and written.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut buffer = DataFrame::empty_with_schema(slf.input_schema.as_ref());
            let row_group_size = slf
                .write_options
                .row_group_size
                .unwrap_or(DEFAULT_ROW_GROUP_SIZE)
                .max(1);
            let mut stop_requested = false;
            let mut row_group_index = 0;

            loop {
                match receiver.recv().await {
                    Err(_) => stop_requested = true,
                    Ok(morsel) => {
                        let df = morsel.into_df();

                        // @NOTE: This also performs schema validation.
                        buffer.vstack_mut(&df)?;
                    },
                }

                while (stop_requested && buffer.height() > 0) || buffer.height() >= row_group_size {
                    let row_group;

                    (row_group, buffer) =
                        buffer.split_at(row_group_size.min(buffer.height()) as i64);

                    for (column_idx, column) in row_group.take_columns().into_iter().enumerate() {
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

            PolarsResult::Ok(())
        }));

        // Encode task.
        //
        // Task encodes the columns into their corresponding Parquet encoding.
        for (mut receiver, mut sender) in distribute_channels.into_iter().zip(senders) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok((rg_idx, col_idx, column)) = receiver.recv().await {
                    let type_ = &slf.parquet_schema.fields()[col_idx];
                    let encodings = &slf.encodings[col_idx];

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
        // Task that will actually do write to the target file.
        let io_runtime = polars_io::pl_async::get_runtime();

        let path = slf.path.clone();
        let input_schema = slf.input_schema.clone();
        let arrow_schema = slf.arrow_schema.clone();
        let parquet_schema = slf.parquet_schema.clone();
        let encodings = slf.encodings.clone();

        let io_task = io_runtime.spawn(async move {
            use tokio::fs::OpenOptions;

            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await
                .map_err(|err| polars_utils::_limit_path_len_io_err(path.as_path(), err))?;
            let writer = BufWriter::new(file.into_std().await);
            let mut writer = BatchedWriter::new(
                Mutex::new(FileWriter::new_with_parquet_schema(
                    writer,
                    arrow_schema,
                    parquet_schema,
                    options,
                )),
                encodings,
                options,
                false,
            );

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

                assert_eq!(current_row_group.len(), num_parquet_columns);
                writer.write_row_group(&current_row_group)?;
                current_row_group.clear();
            }

            writer.finish()?;

            PolarsResult::Ok(())
        });
        join_handles
            .push(scope.spawn_task(TaskPriority::Low, async move { io_task.await.unwrap() }));
    }
}
