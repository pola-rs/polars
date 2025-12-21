use polars_error::PolarsResult;
use polars_io::utils::file::Writeable;
use polars_utils::IdxSize;

use crate::async_executor;
use crate::async_primitives::connector;
use crate::nodes::io_sinks2::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks2::components::size::RowCountAndSize;
use crate::utils::tokio_handle_ext;

pub trait FileWriterStarter: Send + Sync + 'static {
    fn writer_name(&self) -> &str;

    /// Hints to the sender how morsels should be sized.
    ///
    /// `num_rows` will be respected exactly.
    fn ideal_morsel_size(&self) -> RowCountAndSize;

    fn start_file_writer(
        &self,
        morsel_rx: connector::Receiver<SinkMorsel>,
        file: tokio_handle_ext::AbortOnDropHandle<PolarsResult<Writeable>>,
    ) -> PolarsResult<async_executor::JoinHandle<PolarsResult<()>>>;
}

/// Load environment configuration for ideal morsel size.
pub(super) fn ideal_sink_morsel_size_env() -> (Option<IdxSize>, Option<u64>) {
    let num_rows = std::env::var("POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS")
        .map(|x| {
            x.parse::<IdxSize>()
                .ok()
                .filter(|x| *x > 0)
                .unwrap_or_else(|| {
                    panic!("invalid value for POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS: {x}")
                })
        })
        .ok();

    let mut num_bytes = std::env::var("POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES")
        .map(|x| {
            x.parse::<u64>().ok().filter(|x| *x > 0).unwrap_or_else(|| {
                panic!("invalid value for POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES: {x}")
            })
        })
        .ok();

    (
        num_rows,
        num_bytes.or(num_rows.is_some().then_some(u64::MAX)),
    )
}
