use std::num::{NonZeroU64, NonZeroUsize};

use futures::FutureExt;
use polars_error::PolarsResult;
use polars_io::utils::file::Writeable;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::index::NonZeroIdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor;
use crate::async_primitives::connector;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::components::size::TakeableRowsProvider;
use crate::utils::tokio_handle_ext;

pub const IPC_RW_RECORD_BATCH_FLAGS_KEY: PlSmallStr =
    PlSmallStr::from_static("polars:statistics:v1");

pub trait FileWriterStarter: Send + Sync + 'static {
    fn writer_name(&self) -> &str;

    /// Hints to the sender how morsels should be sized.
    fn takeable_rows_provider(&self) -> TakeableRowsProvider;

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: FileOpenTaskHandle,
        num_pipelines: NonZeroUsize,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>>;
}

pub struct FileOpenTaskHandle {
    handle: tokio_handle_ext::AbortOnDropHandle<PolarsResult<Writeable>>,
    sync_on_close: SyncOnCloseType,
}

impl FileOpenTaskHandle {
    pub fn new(
        handle: tokio_handle_ext::AbortOnDropHandle<PolarsResult<Writeable>>,
        sync_on_close: SyncOnCloseType,
    ) -> Self {
        Self {
            handle,
            sync_on_close,
        }
    }
}

impl std::future::Future for FileOpenTaskHandle {
    type Output = PolarsResult<(Writeable, SyncOnCloseType)>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::task::Poll;

        let file: Result<_, tokio::task::JoinError> = futures::ready!(self.handle.poll_unpin(cx));
        let file: PolarsResult<Writeable> = file.unwrap();

        Poll::Ready(file.map(|file| (file, self.sync_on_close)))
    }
}

/// Load ideal morsel size configuration from environment variables.
pub(super) fn ideal_sink_morsel_size_env() -> (Option<IdxSize>, Option<u64>) {
    let num_rows = std::env::var("POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS")
        .map(|x| {
            x.parse::<NonZeroIdxSize>()
                .ok()
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS: {x}")
                })
                .get()
        })
        .ok();

    let num_bytes = std::env::var("POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES")
        .map(|x| {
            x.parse::<NonZeroU64>()
                .ok()
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES: {x}")
                })
                .get()
        })
        .ok();

    (
        num_rows,
        num_bytes.or(num_rows.is_some().then_some(u64::MAX)),
    )
}
