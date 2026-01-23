use std::sync::Arc;

use arrow::io::ipc::IpcField;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow;
use polars_error::PolarsResult;
use polars_io::SerWriter;
use polars_io::ipc::{IpcWriter, IpcWriterOptions};

use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;
use crate::nodes::io_sinks::writers::ipc::IpcBatch;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub ipc_batch_rx: tokio::sync::mpsc::Receiver<IpcBatch>,
    pub options: Arc<IpcWriterOptions>,
    pub schema: SchemaRef,
    pub ipc_fields: Vec<IpcField>,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        let IOWriter {
            file,
            mut ipc_batch_rx,
            options,
            schema,
            ipc_fields,
        } = self;

        let (mut file, sync_on_close) = file.await?;
        let mut buffered_file = file.as_buffered();

        let mut ipc_writer = IpcWriter::new(&mut *buffered_file)
            .with_compression(options.compression)
            .with_compat_level(options.compat_level)
            .with_parallel(false)
            .batched(&schema, ipc_fields)?;

        while let Some(batch) = ipc_batch_rx.recv().await {
            match batch {
                IpcBatch::Record(handle, sink_morsel_permit) => {
                    let encoded_data = handle.await;
                    ipc_writer.write_encoded(&[], &encoded_data)?;
                    drop(encoded_data);
                    drop(sink_morsel_permit);
                },
                IpcBatch::Dictionary(dictionary_data) => {
                    ipc_writer.write_encoded_dictionaries(&[dictionary_data])?
                },
            }
        }

        ipc_writer.finish()?;
        drop(ipc_writer);
        drop(buffered_file);

        file.close(sync_on_close)?;

        Ok(())
    }
}
