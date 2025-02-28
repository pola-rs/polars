use std::cmp::Reverse;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use polars_core::frame::DataFrame;
use polars_core::prelude::{ArrowSchema, Column, CompatLevel};
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

use super::{SinkNode, SinkRecvPort};
use crate::async_executor::spawn;
use crate::nodes::{JoinHandle, TaskPriority};

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
        _num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let (handle, mut buffer_rx, worker_rxs, mut io_rx) =
            recv_ports_recv.serial_into_distribute::<(usize, usize, Column), Linearized>();
        join_handles.push(handle);

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
            let mut row_group_index = 0;

            while let Ok((outcome, receiver, mut sender)) = buffer_rx.recv().await {
                match receiver {
                    None => {
                        // Flush the remaining rows.
                        while buffer.height() > 0 {
                            let row_group;

                            (row_group, buffer) =
                                buffer.split_at(row_group_size.min(buffer.height()) as i64);
                            for (column_idx, column) in
                                row_group.take_columns().into_iter().enumerate()
                            {
                                if sender
                                    .send((row_group_index, column_idx, column))
                                    .await
                                    .is_err()
                                {
                                    return Ok(());
                                }
                            }
                            row_group_index += 1;
                        }
                    },
                    Some(mut receiver) => {
                        while let Ok(morsel) = receiver.recv().await {
                            let (df, _, _, consume_token) = morsel.into_inner();
                            // @NOTE: This also performs schema validation.
                            buffer.vstack_mut(&df)?;

                            while buffer.height() >= row_group_size {
                                let row_group;

                                (row_group, buffer) =
                                    buffer.split_at(row_group_size.min(buffer.height()) as i64);

                                for (column_idx, column) in
                                    row_group.take_columns().into_iter().enumerate()
                                {
                                    if sender
                                        .send((row_group_index, column_idx, column))
                                        .await
                                        .is_err()
                                    {
                                        return Ok(());
                                    }
                                }

                                row_group_index += 1;
                            }
                            drop(consume_token); // Keep the consume_token until here to increase
                                                 // the backpressure.
                        }
                    },
                }

                outcome.stopped();
            }

            PolarsResult::Ok(())
        }));

        // Encode task.
        //
        // Task encodes the columns into their corresponding Parquet encoding.
        for mut worker_rx in worker_rxs {
            let parquet_schema = self.parquet_schema.clone();
            let encodings = self.encodings.clone();

            join_handles.push(spawn(TaskPriority::High, async move {
                while let Ok((outcome, mut dist_rx, mut lin_tx)) = worker_rx.recv().await {
                    while let Ok((rg_idx, col_idx, column)) = dist_rx.recv().await {
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

                        if lin_tx
                            .insert(Priority(Reverse((rg_idx, col_idx)), compressed_pages))
                            .await
                            .is_err()
                        {
                            return Ok(());
                        }
                    }

                    outcome.stopped();
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

            while let Ok((outcome, mut lin_rx)) = io_rx.recv().await {
                // Linearize from all the Encoder tasks.
                while let Some(Priority(Reverse((_, col_idx)), compressed_pages)) =
                    lin_rx.get().await
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

                outcome.stopped();
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
