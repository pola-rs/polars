use std::sync::Arc;

use polars_error::PolarsResult;
use polars_io::prelude::FileMetadata;
use polars_io::utils::byte_source::DynByteSource;

use super::ParquetSourceNode;
use crate::async_primitives::connector::connector;
use crate::utils::task_handles_ext;

impl ParquetSourceNode {
    /// Constructs the task that fetches file metadata.
    /// Note: This must be called AFTER `self.projected_arrow_schema` has been initialized.
    #[allow(clippy::type_complexity)]
    pub(super) fn init_metadata_fetcher(
        &mut self,
    ) -> (
        crate::async_primitives::connector::Receiver<(
            usize,
            usize,
            Arc<DynByteSource>,
            Arc<FileMetadata>,
        )>,
        task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        let (mut metadata_tx, metadata_rx) = connector();

        let byte_source_builder = self.byte_source_builder.clone();

        if self.verbose {
            eprintln!(
                "[ParquetSource]: Byte source builder: {:?}",
                &byte_source_builder
            );
        }

        let (start_tx, start_rx) = tokio::sync::oneshot::channel();
        self.morsel_stream_starter = Some(start_tx);

        let metadata = self.metadata.clone();
        let scan_sources = self.scan_sources.clone();
        let cloud_options = self.cloud_options.clone();
        let byte_source_builder = byte_source_builder.clone();
        let metadata_task_handle = io_runtime.spawn(async move {
            let current_row_offset_ref = &mut 0usize;
            let current_path_index_ref = &mut 0usize;

            if start_rx.await.is_err() {
                return Ok(());
            }

            if verbose {
                eprintln!("[ParquetSource]: Starting data fetch")
            }

            *current_path_index_ref += 1;

            let byte_source = Arc::new(
                scan_sources
                    .get(0)
                    .unwrap()
                    .to_dyn_byte_source(&byte_source_builder, cloud_options.as_ref())
                    .await?,
            );

            let current_row_offset = *current_row_offset_ref;
            *current_row_offset_ref = current_row_offset.saturating_add(metadata.num_rows);

            if metadata_tx
                .send((0, current_row_offset, byte_source, metadata))
                .await
                .is_err()
            {
                return Ok(());
            }

            Ok(())
        });

        let metadata_task_handle = task_handles_ext::AbortOnDropHandle(metadata_task_handle);

        (metadata_rx, metadata_task_handle)
    }
}
