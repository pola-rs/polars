use std::cmp::Reverse;
use std::io::BufWriter;
use std::sync::Mutex;

use polars_core::prelude::{ArrowSchema, CompatLevel};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::write::BatchedWriter;
use polars_io::prelude::{ParquetWriteOptions, get_column_write_options};
use polars_io::schema_to_arrow_checked;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    ColumnWriteOptions, CompressedPage, Compressor, FileWriter, SchemaDescriptor, Version,
    WriteOptions, array_to_columns, to_parquet_schema,
};
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

pub struct ParquetSinkNode {
    target: SinkTarget,

    input_schema: SchemaRef,
    sink_options: SinkOptions,
    write_options: ParquetWriteOptions,

    parquet_schema: SchemaDescriptor,
    arrow_schema: ArrowSchema,
    column_options: Vec<ColumnWriteOptions>,
    cloud_options: Option<CloudOptions>,
}

impl ParquetSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        target: SinkTarget,
        sink_options: SinkOptions,
        write_options: &ParquetWriteOptions,
        cloud_options: Option<CloudOptions>,
    ) -> PolarsResult<Self> {
        let schema = schema_to_arrow_checked(&input_schema, CompatLevel::newest(), "parquet")?;
        let column_options: Vec<ColumnWriteOptions> =
            get_column_write_options(&schema, &write_options.field_overwrites);
        let parquet_schema = to_parquet_schema(&schema, &column_options)?;

        Ok(Self {
            target,

            input_schema,
            sink_options,
            write_options: write_options.clone(),

            parquet_schema,
            arrow_schema: schema,
            column_options,
            cloud_options,
        })
    }
}

// 512 ^ 2
const DEFAULT_ROW_GROUP_SIZE: usize = 1 << 18;

impl SinkNode for ParquetSinkNode {
    fn name(&self) -> &str {
        "parquet-sink"
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
        let (mut io_tx, mut io_rx) = connector::<Vec<Vec<CompressedPage>>>();

        let write_options = &self.write_options;

        let options = WriteOptions {
            statistics: write_options.statistics,
            compression: write_options.compression.into(),
            version: Version::V1,
            data_page_size: write_options.data_page_size,
        };

        // Buffer task.
        join_handles.push(buffer_and_distribute_columns_task(
            recv_port_rx,
            dist_tx,
            write_options
                .row_group_size
                .unwrap_or(DEFAULT_ROW_GROUP_SIZE),
            self.input_schema.clone(),
        ));

        // Encode task.
        //
        // Task encodes the columns into their corresponding Parquet encoding.
        join_handles.extend(
            dist_rxs
                .into_iter()
                .zip(lin_txs)
                .map(|(mut dist_rx, mut lin_tx)| {
                    let parquet_schema = self.parquet_schema.clone();
                    let column_options = self.column_options.clone();

                    spawn(TaskPriority::High, async move {
                        while let Ok((rg_idx, col_idx, column)) = dist_rx.recv().await {
                            let type_ = &parquet_schema.fields()[col_idx];
                            let column_options = &column_options[col_idx];

                            let array = column.as_materialized_series().rechunk();
                            let array = array.to_arrow(0, CompatLevel::newest());

                            // @TODO: This causes all structs fields to be handled on a single thread. It
                            // would be preferable to split the encoding among multiple threads.

                            // @NOTE: Since one Polars column might contain multiple Parquet columns (when
                            // it has a struct datatype), we return a Vec<Vec<CompressedPage>>.

                            // Array -> Parquet pages.
                            let encoded_columns =
                                array_to_columns(array, type_.clone(), column_options, options)?;

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
                                .insert(Priority(Reverse(rg_idx), (col_idx, compressed_pages)))
                                .await
                                .is_err()
                            {
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
        let num_parquet_columns = self.parquet_schema.leaves().len();
        join_handles.push(spawn(TaskPriority::High, async move {
            struct Current {
                seq: usize,
                num_columns_seen: usize,
                columns: Vec<Option<Vec<Vec<CompressedPage>>>>,
            }

            let mut current = Current {
                seq: 0,
                num_columns_seen: 0,
                columns: (0..input_schema.len()).map(|_| None).collect(),
            };

            // Linearize from all the Encoder tasks.
            while let Some(Priority(Reverse(seq), (i, compressed_pages))) = lin_rx.get().await {
                if current.num_columns_seen == 0 {
                    current.seq = seq;
                }

                debug_assert_eq!(current.seq, seq);
                debug_assert!(current.columns[i].is_none());
                current.columns[i] = Some(compressed_pages);
                current.num_columns_seen += 1;

                if current.num_columns_seen == input_schema.len() {
                    // @Optimize: Keep track of these sizes so we can correctly preallocate
                    // them.
                    let mut current_row_group: Vec<Vec<CompressedPage>> =
                        Vec::with_capacity(num_parquet_columns);
                    for column in current.columns.iter_mut() {
                        current_row_group.extend(column.take().unwrap());
                    }

                    if io_tx.send(current_row_group).await.is_err() {
                        return Ok(());
                    }
                    current.num_columns_seen = 0;
                }
            }

            Ok(())
        }));

        // IO task.
        //
        // Task that will actually do write to the target file. It is important that this is only
        // spawned once.
        let target = self.target.clone();
        let sink_options = self.sink_options.clone();
        let cloud_options = self.cloud_options.clone();
        let write_options = self.write_options.clone();
        let arrow_schema = self.arrow_schema.clone();
        let parquet_schema = self.parquet_schema.clone();
        let column_options = self.column_options.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            let mut file = target
                .open_into_writeable_async(&sink_options, cloud_options.as_ref())
                .await?;

            let writer = BufWriter::new(&mut *file);
            let key_value_metadata = write_options.key_value_metadata;
            let write_options = WriteOptions {
                statistics: write_options.statistics,
                compression: write_options.compression.into(),
                version: Version::V1,
                data_page_size: write_options.data_page_size,
            };
            let file_writer = Mutex::new(FileWriter::new_with_parquet_schema(
                writer,
                arrow_schema,
                parquet_schema,
                write_options,
            ));
            let mut writer = BatchedWriter::new(
                file_writer,
                column_options,
                write_options,
                false,
                key_value_metadata,
            );

            let num_parquet_columns = writer.parquet_schema().leaves().len();
            while let Ok(current_row_group) = io_rx.recv().await {
                // @TODO: At the moment this is a sync write, this is not ideal because we can only
                // have so many blocking threads in the tokio threadpool.
                assert_eq!(current_row_group.len(), num_parquet_columns);
                writer.write_row_group(&current_row_group)?;
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
