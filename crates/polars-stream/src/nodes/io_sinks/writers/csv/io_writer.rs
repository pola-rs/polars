use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::prelude::{CsvWriterOptions, write_bom, write_csv_header};
use polars_io::utils::file::AsyncWriteable;
use tokio::io::AsyncWriteExt as _;

use crate::async_executor;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorselPermit;
use crate::nodes::io_sinks::writers::csv::morsel_serializer::MorselSerializer;
use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub filled_serializer_rx: tokio::sync::mpsc::Receiver<(
        async_executor::AbortOnDropHandle<PolarsResult<MorselSerializer>>,
        SinkMorselPermit,
    )>,
    pub reuse_serializer_tx: tokio::sync::mpsc::Sender<MorselSerializer>,
    pub schema: SchemaRef,
    pub options: Arc<CsvWriterOptions>,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        let IOWriter {
            file,
            mut filled_serializer_rx,
            reuse_serializer_tx,
            schema,
            options,
        } = self;

        let (mut file, sync_on_close) = file.await?;

        if options.include_bom {
            write_bom(&mut *file)?
        }

        if options.include_header {
            let names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
            write_csv_header(&mut *file, names.as_slice(), &options.serialize_options)?;
        }

        let mut file: AsyncWriteable = file.try_into_async_writeable()?;

        while let Some((handle, permit)) = filled_serializer_rx.recv().await {
            let serializer = handle.await?;

            file.write_all(&serializer.serialized_data).await?;

            drop(permit);

            let _ = reuse_serializer_tx.send(serializer).await;
        }

        file.close(sync_on_close).await?;

        Ok(())
    }
}
