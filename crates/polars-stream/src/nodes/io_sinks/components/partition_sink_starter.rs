use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_async::executor;
use polars_async::primitives::connector;
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_plan::dsl::file_provider::FileProviderArgs;

use crate::nodes::TaskPriority;
use crate::nodes::io_sinks::components::file_provider::FileProvider;
use crate::nodes::io_sinks::components::file_sink::{FileSinkPermit, FileSinkTaskData};
use crate::nodes::io_sinks::components::sinked_path_info_list::SinkedPathInfoList;
use crate::nodes::io_sinks::components::size::RowCountAndSize;
use crate::nodes::io_sinks::writers::interface::{FileOpenTaskHandle, FileWriterStarter};
use crate::utils::tokio_handle_ext;

#[derive(Clone)]
pub struct PartitionSinkStarter {
    pub file_provider: Arc<FileProvider>,
    pub writer_starter: Arc<dyn FileWriterStarter>,
    pub sync_on_close: SyncOnCloseType,
    pub num_pipelines_per_sink: NonZeroUsize,
    pub sinked_path_info_list: Option<SinkedPathInfoList>,
}

impl PartitionSinkStarter {
    pub fn start_sink(
        &self,
        file_provider_args: FileProviderArgs,
        start_position: RowCountAndSize,
        file_permit: FileSinkPermit,
    ) -> PolarsResult<FileSinkTaskData> {
        let file_provider = Arc::clone(&self.file_provider);
        let path_info_entry = self.sinked_path_info_list.as_ref().map(|x| x.new_entry());
        let file_open_task = tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn({
            let path_info_entry = path_info_entry.clone();
            async move {
                file_provider
                    .open_file(file_provider_args, path_info_entry)
                    .await
            }
        }));

        let (morsel_tx, morsel_rx) = connector::connector();

        let writer_handle = self.writer_starter.start_file_writer(
            morsel_rx,
            FileOpenTaskHandle::new(file_open_task, self.sync_on_close),
            self.num_pipelines_per_sink,
        )?;

        let task_handle = executor::spawn(TaskPriority::High, async move {
            writer_handle.await?;
            Ok(file_permit)
        });

        Ok(FileSinkTaskData::new(
            morsel_tx,
            start_position,
            task_handle,
            path_info_entry,
        ))
    }
}
