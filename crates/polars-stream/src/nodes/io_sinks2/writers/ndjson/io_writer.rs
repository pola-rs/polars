use polars_error::PolarsResult;
use polars_io::utils::file::AsyncWriteable;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use tokio::io::AsyncWriteExt as _;

use crate::async_executor;
use crate::nodes::io_sinks2::components::sink_morsel::SinkMorselPermit;
use crate::nodes::io_sinks2::writers::ndjson::morsel_serializer::MorselSerializer;
use crate::utils::tokio_handle_ext;

pub struct IOWriter {
    pub file:
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<polars_io::prelude::file::Writeable>>,
    pub filled_serializer_rx: tokio::sync::mpsc::Receiver<(
        async_executor::AbortOnDropHandle<PolarsResult<MorselSerializer>>,
        SinkMorselPermit,
    )>,
    pub reuse_serializer_tx: tokio::sync::mpsc::Sender<MorselSerializer>,
    pub sync_on_close: SyncOnCloseType,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        let IOWriter {
            file,
            mut filled_serializer_rx,
            reuse_serializer_tx,
            sync_on_close,
        } = self;

        let file = file.await.unwrap()?;
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
