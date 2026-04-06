use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_plan::dsl::file_provider::FileProviderArgs;
use polars_plan::dsl::sink::SinkedFileInfo;

use crate::async_executor;
use crate::async_primitives::connector;
use crate::nodes::TaskPriority;
use crate::nodes::io_sinks::components::file_provider::FileProvider;
use crate::nodes::io_sinks::components::file_sink::{FileSinkPermit, FileSinkTaskData};
use crate::nodes::io_sinks::components::size::RowCountAndSize;
use crate::nodes::io_sinks::writers::interface::{FileOpenTaskHandle, FileWriterStarter};
use crate::utils::tokio_handle_ext;

#[derive(Clone)]
pub struct PartitionSinkStarter {
    pub file_provider: Arc<FileProvider>,
    pub writer_starter: Arc<dyn FileWriterStarter>,
    pub sync_on_close: SyncOnCloseType,
    pub num_pipelines_per_sink: NonZeroUsize,
    pub compute_file_stats: bool,
}

impl PartitionSinkStarter {
    pub fn start_sink(
        &self,
        file_provider_args: FileProviderArgs,
        start_position: RowCountAndSize,
        file_permit: FileSinkPermit,
    ) -> PolarsResult<FileSinkTaskData> {
        let file_provider = Arc::clone(&self.file_provider);
        let (path_tx, path_rx) = opt_connector(self.file_provider.sinked_file_info_list.is_some());
        let partition_keys = if self.file_provider.sinked_file_info_list.is_some() {
            Some(file_provider_args.partition_keys.clone())
        } else {
            None
        };
        let file_open_task = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime()
                .spawn(async move { file_provider.open_file(file_provider_args, path_tx).await }),
        );

        let (morsel_tx, morsel_rx) = connector::connector();
        let (file_stats_tx, file_stats_rx) = opt_connector(self.compute_file_stats);

        let writer_handle = self.writer_starter.start_file_writer(
            morsel_rx,
            FileOpenTaskHandle::new(file_open_task, self.sync_on_close),
            self.num_pipelines_per_sink,
            file_stats_tx,
        )?;

        let sinked_file_info_list = self.file_provider.sinked_file_info_list.clone();
        let task_handle = async_executor::spawn(TaskPriority::High, async move {
            writer_handle.await?;

            if let Some(sinked_file_info_list) = sinked_file_info_list {
                let path = path_rx.unwrap().try_recv().unwrap();
                let stats = file_stats_rx.and_then(|mut rx| rx.try_recv().ok());
                sinked_file_info_list
                    .file_info_list
                    .lock()
                    .push(SinkedFileInfo {
                        path,
                        partition_keys: partition_keys.unwrap(),
                        stats,
                    });
            }

            Ok(file_permit)
        });

        Ok(FileSinkTaskData {
            morsel_tx,
            start_position,
            task_handle,
        })
    }
}

fn opt_connector<T>(
    condition: bool,
) -> (Option<connector::Sender<T>>, Option<connector::Receiver<T>>) {
    if condition {
        let (tx, rx) = connector::connector();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    }
}
