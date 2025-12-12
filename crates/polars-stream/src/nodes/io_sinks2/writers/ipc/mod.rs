use std::sync::Arc;

use arrow::io::ipc::IpcField;
use polars_core::schema::SchemaRef;
use polars_core::series::ToArrowConverter;
use polars_core::utils::arrow;
use polars_core::utils::arrow::io::ipc::write::{EncodedData, WriteOptions};
use polars_error::PolarsResult;
use polars_io::ipc::IpcWriterOptions;
use polars_io::pl_async;
use polars_io::utils::sync_on_close::SyncOnCloseType;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::nodes::io_sinks2::components::sink_morsel::{SinkMorsel, SinkMorselPermit};
use crate::nodes::io_sinks2::components::size::RowCountAndSize;
use crate::nodes::io_sinks2::writers::interface::{
    FileWriterStarter, default_ideal_sink_morsel_size,
};
use crate::utils::tokio_handle_ext;

pub mod initialization;
mod io_writer;
mod record_batch_encoder;

pub struct IpcWriterStarter {
    pub options: IpcWriterOptions,
    pub schema: SchemaRef,
    pub arrow_converters: Vec<ToArrowConverter>,
    pub ipc_fields: Vec<IpcField>,
    pub dictionary_id_offsets: Arc<[usize]>,
    pub pipeline_depth: usize,
    pub sync_on_close: SyncOnCloseType,
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

    fn ideal_morsel_size(&self) -> RowCountAndSize {
        default_ideal_sink_morsel_size()
    }

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: tokio_handle_ext::AbortOnDropHandle<
            PolarsResult<polars_io::prelude::file::Writeable>,
        >,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>> {
        let (ipc_batch_tx, ipc_batch_rx) =
            tokio::sync::mpsc::channel::<IpcBatch>(self.pipeline_depth);

        let io_handle = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime().spawn(
                io_writer::IOWriter {
                    file,
                    ipc_batch_rx,
                    options: self.options,
                    schema: Arc::clone(&self.schema),
                    ipc_fields: self.ipc_fields.clone(),
                    sync_on_close: self.sync_on_close,
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
                    arrow_converters: Vec::clone(&self.arrow_converters),
                    dictionary_id_offsets: Arc::clone(&self.dictionary_id_offsets),
                    write_options: WriteOptions {
                        compression: self.options.compression.map(|x| x.into()),
                    },
                }
                .run(),
            ));

        Ok(async_executor::spawn(TaskPriority::Low, async move {
            record_batch_encoder_handle.await?;
            io_handle.await.unwrap()?;
            Ok(())
        }))
    }
}
