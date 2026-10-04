use std::sync::Arc;

use polars_async::executor;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::prelude::{CsvWriterOptions, ExternalCompression, UTF8_BOM, csv_header};

use crate::nodes::io_sinks::components::sink_morsel::SinkMorselPermit;
use crate::nodes::io_sinks::writers::csv::morsel_serializer::MorselSerializer;
use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub filled_serializer_rx: tokio::sync::mpsc::Receiver<(
        executor::AbortOnDropHandle<PolarsResult<MorselSerializer>>,
        SinkMorselPermit,
    )>,
    pub reuse_serializer_tx: tokio::sync::mpsc::Sender<MorselSerializer>,
    pub schema: SchemaRef,
    pub options: Arc<CsvWriterOptions>,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        let Self {
            file,
            mut filled_serializer_rx,
            reuse_serializer_tx,
            schema,
            options,
        } = self;

        let (mut target, sync_on_close) = file.await?;

        // TODO: Refactor to avoid doing compression work on a tokio async thread here.
        let (mut writer, finish_writer) =
            if options.compression == ExternalCompression::Uncompressed {
                (target.as_writable(), None)
            } else {
                target.as_compressed_writable(options.compression)?
            };

        if options.include_bom {
            writer.write_all(&UTF8_BOM).await?;
        }

        if options.include_header {
            let names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
            writer
                .write_all(&csv_header(names.as_slice(), &options.serialize_options)?)
                .await?;
        }

        while let Some((handle, permit)) = filled_serializer_rx.recv().await {
            let mut serializer = handle.await?;

            writer
                .write_all_owned(&mut serializer.serialized_data)
                .await?;

            drop(permit);

            let _ = reuse_serializer_tx.send(serializer).await;
        }

        if let Some(finish_writer) = &finish_writer {
            finish_writer(writer)?.flush().await?;
        } else {
            writer.flush().await?;
            drop(writer);
        }

        target.close(sync_on_close).await?;

        Ok(())
    }
}
