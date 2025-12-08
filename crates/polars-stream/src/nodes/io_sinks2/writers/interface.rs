use polars_error::PolarsResult;
use polars_io::utils::file::Writeable;

use crate::async_executor;
use crate::async_primitives::connector;
use crate::nodes::io_sinks2::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks2::components::size::RowCountAndSize;
use crate::utils::task_handles_ext;

pub trait FileWriterStarter: Send + Sync + 'static {
    fn writer_name(&self) -> &str;

    /// Hints to the sender how morsels should be sized.
    fn ideal_morsel_size(&self) -> RowCountAndSize;

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: task_handles_ext::AbortOnDropHandle<PolarsResult<Writeable>>,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>>;
}

pub(super) fn default_ideal_sink_morsel_size() -> RowCountAndSize {
    RowCountAndSize {
        num_rows: 122_880,
        num_bytes: {
            std::env::var("POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES")
                .map(|x| {
                    x.parse::<u64>().ok().filter(|x| *x > 0).unwrap_or_else(|| {
                        panic!("invalid value for POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES: {x}")
                    })
                })
                .unwrap_or(64 * 1024 * 1024)
        },
    }
}
