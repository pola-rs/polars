use std::sync::Arc;

use polars_core::config;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_io::prelude::{CsvSerializer, CsvWriterOptions};
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
mod morsel_serializer;

pub struct CsvWriterStarter {
    pub options: Arc<CsvWriterOptions>,
    /// `Mutex` is to handle `dyn ColumnSerializer` not being `Sync`.
    pub base_serializer: std::sync::Mutex<CsvSerializer>,
    pub schema: SchemaRef,
    pub initialized_state: std::sync::Mutex<Option<InitializedState>>,
}

#[derive(Clone)]
pub struct InitializedState {
    pub ideal_morsel_size: NonZeroRowCountAndSize,
    pub base_allocation_size: usize,
}

impl CsvWriterStarter {
    fn initialized_state(&self) -> InitializedState {
        let mut initialized_state = self.initialized_state.lock().unwrap();

        if initialized_state.is_none() {
            let (env_num_rows, env_num_bytes) = ideal_sink_morsel_size_env();

            let ideal_morsel_size = RowCountAndSize {
                num_rows: env_num_rows.unwrap_or(25 * 1024),
                num_bytes: env_num_bytes.unwrap_or(8 * 1024 * 1024),
            };

            let serialized_row_size_estimate = u64::saturating_mul(self.schema.len() as _, 25);

            let base_allocation_size: usize = u64::min(
                64 * 1024 * 1024,
                u64::min(
                    ideal_morsel_size.num_bytes.div_ceil(2).saturating_mul(5),
                    u64::saturating_mul(
                        serialized_row_size_estimate,
                        ideal_morsel_size.num_rows as _,
                    ),
                ),
            ) as _;

            if config::verbose() {
                eprintln!("[CsvWriterStarter]: base_allocation_size: {base_allocation_size}")
            }

            let ideal_morsel_size = NonZeroRowCountAndSize::new(ideal_morsel_size).unwrap();

            *initialized_state = Some(InitializedState {
                ideal_morsel_size,
                base_allocation_size,
            })
        }

        initialized_state.clone().unwrap()
    }
}

impl FileWriterStarter for CsvWriterStarter {
    fn writer_name(&self) -> &str {
        "csv"
    }

    fn takeable_rows_provider(&self) -> TakeableRowsProvider {
        TakeableRowsProvider {
            max_size: self.initialized_state().ideal_morsel_size,
            byte_size_min_rows: NonZeroIdxSize::new(256).unwrap(),
            allow_non_max_size: true,
        }
    }

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: FileOpenTaskHandle,
        num_pipelines: std::num::NonZeroUsize,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>> {
        let (filled_serializer_tx, filled_serializer_rx) = tokio::sync::mpsc::channel::<(
            async_executor::AbortOnDropHandle<PolarsResult<morsel_serializer::MorselSerializer>>,
            SinkMorselPermit,
        )>(num_pipelines.get());

        let max_serializers = num_pipelines.get();
        let (reuse_serializer_tx, reuse_serializer_rx) =
            tokio::sync::mpsc::channel::<morsel_serializer::MorselSerializer>(max_serializers);

        let io_handle = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime().spawn(
                io_writer::IOWriter {
                    file,
                    filled_serializer_rx,
                    reuse_serializer_tx,
                    schema: Arc::clone(&self.schema),
                    options: Arc::clone(&self.options),
                }
                .run(),
            ),
        );

        let base_csv_serializer = { self.base_serializer.lock().unwrap().clone() };
        let base_allocation_size = self.initialized_state().base_allocation_size;

        let serializer_handle = async_executor::spawn(
            TaskPriority::High,
            morsel_serializer::MorselSerializerPipeline {
                morsel_rx,
                filled_serializer_tx,
                reuse_serializer_rx,
                base_csv_serializer,
                base_allocation_size,
                max_serializers,
            }
            .run(),
        );

        Ok(async_executor::spawn(TaskPriority::Low, async move {
            io_handle.await.unwrap()?;
            serializer_handle.await;
            Ok(())
        }))
    }
}
