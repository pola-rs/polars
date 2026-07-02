use polars_async::executor;
use polars_async::primitives::connector;
use polars_error::PolarsResult;

use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::components::size::RowCountAndSize;

pub type FileSinkPermit = tokio::sync::OwnedSemaphorePermit;

pub struct FileSinkTaskData {
    pub morsel_tx: connector::Sender<SinkMorsel>,
    pub start_position: RowCountAndSize,
    pub task_handle: executor::JoinHandle<PolarsResult<FileSinkPermit>>,
}

impl FileSinkTaskData {
    /// Signals to the writer to close, and returns its task handle.
    pub fn close(self) -> executor::JoinHandle<PolarsResult<FileSinkPermit>> {
        self.task_handle
    }
}
