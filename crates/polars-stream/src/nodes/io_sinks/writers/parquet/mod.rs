use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_io::prelude::{ParquetWriteOptions, get_encodings};
use polars_parquet::write::{
    CompressedPage, Encoding, SchemaDescriptor, Version, WriteOptions, to_parquet_schema,
};
use polars_utils::IdxSize;
use polars_utils::index::NonZeroIdxSize;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::nodes::io_sinks::components::sink_morsel::{SinkMorsel, SinkMorselPermit};
use crate::nodes::io_sinks::components::size::{
    NonZeroRowCountAndSize, RowCountAndSize, TakeableRowsProvider,
};
use crate::nodes::io_sinks::writers::interface::{
    FileOpenTaskHandle, FileWriterStarter, ideal_sink_morsel_size_env,
};
use crate::utils::tokio_handle_ext;

mod io_writer;
mod row_group_encoder;

pub struct ParquetWriterStarter {
    pub options: Arc<ParquetWriteOptions>,
    pub arrow_schema: ArrowSchemaRef,
    pub initialized_state: std::sync::Mutex<Option<InitializedState>>,
    pub row_group_size: Option<IdxSize>,
}

#[derive(Clone)]
pub struct InitializedState {
    encodings: Buffer<Vec<Encoding>>,
    schema_descriptor: Arc<SchemaDescriptor>,
}

struct EncodedRowGroup {
    num_rows: usize,
    data: Vec<Vec<CompressedPage>>,
    morsel_permit: SinkMorselPermit,
}

impl FileWriterStarter for ParquetWriterStarter {
    fn writer_name(&self) -> &str {
        "parquet"
    }

    fn takeable_rows_provider(&self) -> TakeableRowsProvider {
        let max_size = if let Some(row_group_size) = self.row_group_size
            && row_group_size > 0
        {
            NonZeroRowCountAndSize::new(RowCountAndSize {
                num_rows: row_group_size,
                num_bytes: u64::MAX,
            })
            .unwrap()
        } else {
            let (num_rows, num_bytes) = ideal_sink_morsel_size_env();

            NonZeroRowCountAndSize::new(RowCountAndSize {
                num_rows: num_rows.unwrap_or(122_880),
                num_bytes: num_bytes.unwrap_or(u64::MAX),
            })
            .unwrap()
        };

        TakeableRowsProvider {
            max_size,
            byte_size_min_rows: NonZeroIdxSize::new(16384).unwrap(),
            allow_non_max_size: false,
        }
    }

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: FileOpenTaskHandle,
        num_pipelines: std::num::NonZeroUsize,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>> {
        let InitializedState {
            encodings,
            schema_descriptor,
        } = {
            let mut initialized_state = self.initialized_state.lock().unwrap();

            if initialized_state.is_none() {
                let schema_descriptor = Arc::new(to_parquet_schema(&self.arrow_schema)?);
                let encodings = get_encodings(&self.arrow_schema);

                *initialized_state = Some(InitializedState {
                    encodings,
                    schema_descriptor,
                })
            };

            initialized_state.clone().unwrap()
        };

        let (encoded_row_group_tx, encoded_row_group_rx) = tokio::sync::mpsc::channel::<
            async_executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>,
        >(num_pipelines.get());

        let key_value_metadata = self.options.key_value_metadata.clone();
        let write_options = WriteOptions {
            statistics: self.options.statistics,
            compression: self.options.compression.into(),
            version: Version::V1,
            data_page_size: self.options.data_page_size,
        };

        let arrow_schema = Arc::clone(&self.arrow_schema);
        let num_leaf_columns = schema_descriptor.leaves().len();

        let io_handle = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime().spawn(
                io_writer::IOWriter {
                    file,
                    encoded_row_group_rx,
                    arrow_schema,
                    schema_descriptor: Arc::clone(&schema_descriptor),
                    write_options,
                    encodings: Buffer::clone(&encodings),
                    key_value_metadata,
                    num_leaf_columns,
                }
                .run(),
            ),
        );

        let arrow_schema = Arc::clone(&self.arrow_schema);
        let compute_handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::High,
            row_group_encoder::RowGroupEncoder {
                morsel_rx,
                encoded_row_group_tx,
                arrow_schema,
                schema_descriptor,
                write_options,
                encodings,
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
