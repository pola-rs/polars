use polars_async::executor;
use polars_error::PolarsResult;
use polars_io::ExternalCompression;
use polars_io::ndjson::NDJsonWriterOptions;

use crate::nodes::io_sinks::components::sink_morsel::SinkMorselPermit;
use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;
use crate::nodes::io_sinks::writers::ndjson::morsel_serializer::MorselSerializer;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub filled_serializer_rx: tokio::sync::mpsc::Receiver<(
        executor::AbortOnDropHandle<PolarsResult<MorselSerializer>>,
        SinkMorselPermit,
    )>,
    pub reuse_serializer_tx: tokio::sync::mpsc::Sender<MorselSerializer>,
    pub options: NDJsonWriterOptions,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        let IOWriter {
            file,
            mut filled_serializer_rx,
            reuse_serializer_tx,
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

        while let Some((handle, permit)) = filled_serializer_rx.recv().await {
            let serializer = handle.await?;

            writer.write_all(&serializer.serialized_data).await?;

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
