use polars_async::executor;
use polars_async::primitives::connector;
use polars_error::PolarsResult;

use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::components::sinked_path_info_list::SinkedPathInfoEntry;
use crate::nodes::io_sinks::components::size::RowCountAndSize;

pub type FileSinkPermit = tokio::sync::OwnedSemaphorePermit;

pub struct FileSinkTaskData {
    morsel_tx: connector::Sender<SinkMorsel>,
    start_position: RowCountAndSize,
    task_handle: executor::JoinHandle<PolarsResult<FileSinkPermit>>,
    path_info_entry: Option<SinkedPathInfoEntry>,
    sent_rows: u64,
}

impl FileSinkTaskData {
    pub fn new(
        morsel_tx: connector::Sender<SinkMorsel>,
        start_position: RowCountAndSize,
        task_handle: executor::JoinHandle<PolarsResult<FileSinkPermit>>,
        path_info_entry: Option<SinkedPathInfoEntry>,
    ) -> Self {
        Self {
            morsel_tx,
            start_position,
            task_handle,
            path_info_entry,
            sent_rows: 0,
        }
    }

    pub fn start_position(&self) -> RowCountAndSize {
        self.start_position
    }

    pub async fn send_morsel(&mut self, morsel: SinkMorsel) -> Result<(), SinkMorsel> {
        let h = morsel.height();
        self.morsel_tx.send(morsel).await?;
        self.sent_rows = u64::saturating_add(self.sent_rows, h as u64);
        Ok(())
    }

    /// Signals to the writer to close, and returns its task handle.
    pub fn close(self) -> executor::JoinHandle<PolarsResult<FileSinkPermit>> {
        if let Some(path_info_entry) = self.path_info_entry {
            path_info_entry.set_num_rows(self.sent_rows);
        }

        self.task_handle
    }
}
