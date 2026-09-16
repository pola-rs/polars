use std::num::NonZeroU64;
use std::sync::Arc;

use polars_async::executor::{self, TaskPriority};
use polars_async::primitives::connector;
use polars_async::primitives::wait_group::{WaitGroup, WaitToken};
use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::io::ipc::write::WriteOptions;
use polars_error::PolarsResult;
use polars_io::ipc::IpcWriterOptions;
use polars_io::utils::bytes_bufferer::BytesBuffererConfig;
use polars_utils::IdxSize;
use polars_utils::index::NonZeroIdxSize;

use crate::nodes::io_sinks::components::sink_morsel::{SinkMorsel, SinkMorselPermit};
use crate::nodes::io_sinks::components::size::{SplitMode, TargetSinkMorselSize};
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
    pub bytes_bufferer_config: BytesBuffererConfig,
}

pub enum IpcBatchType {
    Dictionary,
    Record,
}

pub struct IpcBatch {
    pub batch_type: IpcBatchType,
    pub num_rows: IdxSize,
    pub continuation_bytes: Vec<u8>,
    pub message_bytes: <Vec<Buffer<u8>> as IntoIterator>::IntoIter,
    pub message_num_bytes: usize,
    pub arrow_data_bytes: <Vec<Buffer<u8>> as IntoIterator>::IntoIter,
    pub arrow_data_num_bytes: usize,
    pub morsel_permit: Option<SinkMorselPermit>,
}

impl FileWriterStarter for IpcWriterStarter {
    fn writer_name(&self) -> &str {
        "ipc"
    }

    fn target_sink_morsel_size(&self) -> TargetSinkMorselSize {
        let target_num_bytes_min_rows = const { NonZeroIdxSize::new(16384).unwrap() };

        if let Some(record_batch_size) = self.record_batch_size
            && record_batch_size > 0
        {
            TargetSinkMorselSize {
                target_num_rows: record_batch_size.try_into().unwrap(),
                target_num_bytes: NonZeroU64::MAX,
                target_num_bytes_min_rows,
                target_num_rows_mode: SplitMode::Exact,
            }
        } else {
            let (env_num_rows, env_num_bytes) = ideal_sink_morsel_size_env();

            TargetSinkMorselSize {
                target_num_rows: env_num_rows
                    .unwrap_or(const { NonZeroIdxSize::new(122_880).unwrap() }),
                target_num_bytes: env_num_bytes.unwrap_or(NonZeroU64::MAX),
                target_num_bytes_min_rows,
                target_num_rows_mode: SplitMode::Approximate,
            }
        }
    }

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: FileOpenTaskHandle,
        num_pipelines: std::num::NonZeroUsize,
    ) -> PolarsResult<executor::JoinHandle<PolarsResult<()>>> {
        let file_schema = Arc::clone(&self.schema);
        let options = Arc::clone(&self.options);
        let compression = self.options.compression.map(|x| x.into());

        // Note. Environment variable is unstable.
        let write_statistics_flags = self.options.record_batch_statistics;
        let bytes_bufferer_config = self.bytes_bufferer_config.clone();

        let finish_record_batch_write_wg = WaitGroup::default();

        let handle = executor::spawn(TaskPriority::High, async move {
            let (ipc_batch_tx, ipc_batch_rx) = tokio::sync::mpsc::channel::<(
                executor::AbortOnDropHandle<PolarsResult<IpcBatch>>,
                Option<WaitToken>,
            )>(num_pipelines.get());

            let (arrow_converters, ipc_fields, dictionary_id_offsets) =
                build_ipc_write_components(file_schema.as_ref(), options.compat_level);

            let compat_level = options.compat_level;

            let io_handle = tokio_handle_ext::AbortOnDropHandle(
                ASYNC.spawn(
                    io_writer::IOWriter {
                        file,
                        ipc_batch_rx,
                        options,
                        schema: file_schema,
                        ipc_fields,
                        write_custom_pl_metadata: write_statistics_flags,
                    }
                    .run(),
                ),
            );

            let record_batch_encoder_handle = executor::AbortOnDropHandle::new(executor::spawn(
                TaskPriority::High,
                record_batch_encoder::RecordBatchEncoder {
                    morsel_rx,
                    ipc_batch_tx,
                    arrow_converters,
                    compat_level,
                    dictionary_id_offsets,
                    write_options: WriteOptions { compression },
                    write_statistics_flags,
                    bytes_bufferer_config,
                    finish_record_batch_write_wg,
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
