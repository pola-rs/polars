use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::CompatLevel;
use polars_error::PolarsResult;
use polars_io::parquet::write::BatchedWriter;
use polars_io::pl_async;
use polars_io::prelude::{KeyValueMetadata, ParquetWriteOptions, get_column_write_options};
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    ColumnWriteOptions, CompressedPage, Compressor, FileWriter, SchemaDescriptor, Version,
    WriteOptions, array_to_columns, to_parquet_schema,
};
use polars_utils::{IdxSize, UnitVec};

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::opt_spawned_future::parallelize_first_to_local;
use crate::nodes::io_sinks2::components::sink_morsel::{SinkMorsel, SinkMorselPermit};
use crate::nodes::io_sinks2::components::size::RowCountAndSize;
use crate::nodes::io_sinks2::writers::interface::{
    FileWriterStarter, default_ideal_sink_morsel_size,
};
use crate::utils::tokio_handle_ext;

pub struct ParquetWriterStarter {
    pub options: ParquetWriteOptions,
    pub arrow_schema: ArrowSchemaRef,
    pub initialized_state: std::sync::Mutex<Option<InitializedState>>,
    pub pipeline_depth: usize,
    pub sync_on_close: SyncOnCloseType,
    pub row_group_size: Option<IdxSize>,
}

#[derive(Clone)]
pub struct InitializedState {
    column_options: Arc<Vec<ColumnWriteOptions>>,
    schema_descriptor: Arc<SchemaDescriptor>,
}

struct EncodedRowGroup {
    data: Vec<Vec<CompressedPage>>,
    morsel_permit: SinkMorselPermit,
}

impl FileWriterStarter for ParquetWriterStarter {
    fn writer_name(&self) -> &str {
        "parquet"
    }

    fn ideal_morsel_size(&self) -> RowCountAndSize {
        if let Some(row_group_size) = self.row_group_size {
            RowCountAndSize {
                num_rows: row_group_size,
                num_bytes: u64::MAX,
            }
        } else {
            default_ideal_sink_morsel_size()
        }
    }

    fn start_file_writer(
        &self,
        mut morsel_rx: connector::Receiver<SinkMorsel>,
        file: tokio_handle_ext::AbortOnDropHandle<
            PolarsResult<polars_io::prelude::file::Writeable>,
        >,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>> {
        let InitializedState {
            column_options,
            schema_descriptor,
        } = {
            let mut initialized_state = self.initialized_state.lock().unwrap();

            if initialized_state.is_none() {
                let column_options: Arc<Vec<ColumnWriteOptions>> = Arc::new(
                    get_column_write_options(&self.arrow_schema, &self.options.field_overwrites),
                );
                let schema_descriptor = Arc::new(to_parquet_schema(
                    &self.arrow_schema,
                    column_options.as_ref(),
                )?);

                *initialized_state = Some(InitializedState {
                    column_options,
                    schema_descriptor,
                })
            };

            initialized_state.clone().unwrap()
        };

        let (encoded_row_group_tx, encoded_row_group_rx) = tokio::sync::mpsc::channel::<
            async_executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>,
        >(self.pipeline_depth);

        let key_value_metadata = self.options.key_value_metadata.clone();
        let write_options = WriteOptions {
            statistics: self.options.statistics,
            compression: self.options.compression.into(),
            version: Version::V1,
            data_page_size: self.options.data_page_size,
        };

        let sync_on_close = self.sync_on_close;
        let arrow_schema = Arc::clone(&self.arrow_schema);
        let num_leaf_columns = schema_descriptor.leaves().len();

        let io_handle = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime().spawn(
                IOWriter {
                    file,
                    encoded_row_group_rx,
                    arrow_schema,
                    schema_descriptor: Arc::clone(&schema_descriptor),
                    write_options,
                    column_options: Arc::clone(&column_options),
                    key_value_metadata,
                    num_leaf_columns,
                    sync_on_close,
                }
                .run(),
            ),
        );

        let compute_handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::High,
            RowGroupEncoder {
                morsel_rx,
                encoded_row_group_tx,
                schema_descriptor,
                write_options,
                column_options,
                num_leaf_columns,
            }
            .run(),
        ));

        Ok(async_executor::spawn(TaskPriority::Low, async move {
            compute_handle.await?;
            io_handle.await.unwrap()?;
            Ok(())
        }))
    }
}

struct RowGroupEncoder {
    morsel_rx: connector::Receiver<SinkMorsel>,
    encoded_row_group_tx:
        tokio::sync::mpsc::Sender<async_executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>>,
    schema_descriptor: Arc<SchemaDescriptor>,
    write_options: WriteOptions,
    column_options: Arc<Vec<ColumnWriteOptions>>,
    num_leaf_columns: usize,
}

impl RowGroupEncoder {
    async fn run(self) -> PolarsResult<()> {
        let RowGroupEncoder {
            mut morsel_rx,
            encoded_row_group_tx,
            schema_descriptor,
            write_options,
            column_options,
            num_leaf_columns,
        } = self;

        while let Ok(morsel) = morsel_rx.recv().await {
            let schema_descriptor = Arc::clone(&schema_descriptor);
            let column_options = Arc::clone(&column_options);

            let row_group_encode_handle = async_executor::AbortOnDropHandle::new(
                async_executor::spawn(TaskPriority::High, async move {
                    let (df, morsel_permit) = morsel.into_inner();

                    let mut data: Vec<Vec<CompressedPage>> = Vec::with_capacity(num_leaf_columns);

                    for fut in parallelize_first_to_local(
                        TaskPriority::High,
                        df.into_columns().into_iter().enumerate().map(|(i, c)| {
                            let schema_descriptor = Arc::clone(&schema_descriptor);
                            let column_options = Arc::clone(&column_options);

                            async move {
                                let parquet_type = &schema_descriptor.fields()[i];
                                let column_options = &column_options[i];
                                let array = c
                                    .as_materialized_series()
                                    .rechunk()
                                    .to_arrow(0, CompatLevel::newest());

                                let mut data: UnitVec<Vec<CompressedPage>> =
                                    UnitVec::with_capacity(num_leaf_columns);

                                for encode_page_iter in array_to_columns(
                                    array,
                                    parquet_type.clone(),
                                    column_options,
                                    write_options,
                                )? {
                                    let compressed_pages: Vec<CompressedPage> =
                                        Compressor::new_from_vec(
                                            encode_page_iter.map(|result| {
                                                result.map_err(|e| {
                                                    ParquetError::FeatureNotSupported(format!(
                                                        "reraised in polars: {e}",
                                                    ))
                                                })
                                            }),
                                            write_options.compression,
                                            vec![],
                                        )
                                        .collect::<ParquetResult<_>>()?;

                                    data.push(compressed_pages)
                                }

                                PolarsResult::Ok(data)
                            }
                        }),
                    ) {
                        data.extend(fut.await?);
                    }

                    Ok(EncodedRowGroup {
                        data,
                        morsel_permit,
                    })
                }),
            );

            if encoded_row_group_tx
                .send(row_group_encode_handle)
                .await
                .is_err()
            {
                break;
            }
        }

        Ok(())
    }
}

struct IOWriter {
    file: tokio_handle_ext::AbortOnDropHandle<PolarsResult<polars_io::prelude::file::Writeable>>,
    encoded_row_group_rx: tokio::sync::mpsc::Receiver<
        async_executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>,
    >,
    arrow_schema: ArrowSchemaRef,
    schema_descriptor: Arc<SchemaDescriptor>,
    write_options: WriteOptions,
    column_options: Arc<Vec<ColumnWriteOptions>>,
    key_value_metadata: Option<KeyValueMetadata>,
    num_leaf_columns: usize,
    sync_on_close: SyncOnCloseType,
}

impl IOWriter {
    async fn run(self) -> PolarsResult<()> {
        let IOWriter {
            file,
            mut encoded_row_group_rx,
            arrow_schema,
            schema_descriptor,
            write_options,
            column_options,
            key_value_metadata,
            num_leaf_columns,
            sync_on_close,
        } = self;

        let mut file = file.await.unwrap()?;
        let mut buffered_file = file.as_buffered();

        let mut parquet_writer = BatchedWriter::new(
            std::sync::Mutex::new(FileWriter::new_with_parquet_schema(
                &mut *buffered_file,
                Arc::unwrap_or_clone(arrow_schema),
                Arc::unwrap_or_clone(schema_descriptor),
                write_options,
            )),
            Arc::unwrap_or_clone(column_options),
            write_options,
            false,
            key_value_metadata,
        );

        while let Some(handle) = encoded_row_group_rx.recv().await {
            let EncodedRowGroup {
                data,
                morsel_permit,
            } = handle.await?;
            assert_eq!(data.len(), num_leaf_columns);
            parquet_writer.write_row_group(&data)?;
            drop(data);
            drop(morsel_permit);
        }

        parquet_writer.finish()?;
        drop(parquet_writer);
        drop(buffered_file);

        file.close(sync_on_close)?;

        Ok(())
    }
}
