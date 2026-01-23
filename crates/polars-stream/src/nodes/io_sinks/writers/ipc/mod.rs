use std::sync::Arc;

use polars_core::config;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::io::ipc::write::{EncodedData, WriteOptions};
use polars_error::PolarsResult;
use polars_io::ipc::IpcWriterOptions;
use polars_io::pl_async;
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
use crate::nodes::io_sinks::writers::ipc::initialization::build_ipc_write_components;
use crate::utils::tokio_handle_ext;

mod initialization;
mod io_writer;
mod record_batch_encoder;

pub struct IpcWriterStarter {
    pub options: Arc<IpcWriterOptions>,
    pub schema: SchemaRef,
    pub record_batch_size: Option<IdxSize>,
}

enum IpcBatch {
    Record(
        async_executor::AbortOnDropHandle<EncodedData>,
        SinkMorselPermit,
    ),
    Dictionary(EncodedData),
}

impl FileWriterStarter for IpcWriterStarter {
    fn writer_name(&self) -> &str {
        "ipc"
    }

    fn takeable_rows_provider(&self) -> TakeableRowsProvider {
        let max_size = if let Some(record_batch_size) = self.record_batch_size
            && record_batch_size > 0
        {
            NonZeroRowCountAndSize::new(RowCountAndSize {
                num_rows: record_batch_size,
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
        let file_schema = Arc::clone(&self.schema);
        let options = Arc::clone(&self.options);
        let compression = self.options.compression.map(|x| x.into());

        // Note. Environment variable is unstable.
        let write_statistics_flags = self.options.record_batch_statistics;

        if write_statistics_flags && config::verbose() {
            eprintln!(
                "[IpcWriterStarter]: write_record_batch_statistics_flags: {write_statistics_flags}"
            )
        }

        let handle = async_executor::spawn(TaskPriority::High, async move {
            let (ipc_batch_tx, ipc_batch_rx) =
                tokio::sync::mpsc::channel::<IpcBatch>(num_pipelines.get());

            let (arrow_converters, ipc_fields, dictionary_id_offsets) =
                build_ipc_write_components(file_schema.as_ref(), options.compat_level);

            let io_handle = tokio_handle_ext::AbortOnDropHandle(
                pl_async::get_runtime().spawn(
                    io_writer::IOWriter {
                        file,
                        ipc_batch_rx,
                        options,
                        schema: file_schema,
                        ipc_fields,
                    }
                    .run(),
                ),
            );

            let record_batch_encoder_handle =
                async_executor::AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::High,
                    record_batch_encoder::RecordBatchEncoder {
                        morsel_rx,
                        ipc_batch_tx,
                        arrow_converters,
                        dictionary_id_offsets,
                        write_options: WriteOptions { compression },
                        write_statistics_flags,
                    }
                    .run(),
                ));

            record_batch_encoder_handle.await?;
            io_handle.await.unwrap()?;
            Ok(())
        });

        Ok(handle)
    }
}
