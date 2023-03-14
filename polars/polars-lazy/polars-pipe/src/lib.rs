#[cfg(feature = "compile")]
mod executors;
#[cfg(feature = "compile")]
pub mod expressions;
#[cfg(feature = "compile")]
pub mod operators;
#[cfg(feature = "compile")]
pub mod pipeline;

#[cfg(feature = "compile")]
pub use operators::SExecutionContext;
#[cfg(feature = "compile")]
use polars_core::prelude::*;
/// ideal chunk size we strive to have
/// scale the chunk size depending on the number of
/// columns. With 10 columns we use a chunk size of 40_000
#[cfg(feature = "compile")]
pub(crate) fn determine_chunk_size(n_cols: usize, n_threads: usize) -> PolarsResult<usize> {
    if let Ok(val) = std::env::var("POLARS_STREAMING_CHUNK_SIZE") {
        val.parse().map_err(
            |_| polars_err!(ComputeError: "could not parse 'POLARS_STREAMING_CHUNK_SIZE' env var"),
        )
    } else {
        let thread_factor = std::cmp::max(12 / n_threads, 1);
        Ok(std::cmp::max(400_000 / n_cols * thread_factor, 1000))
    }
}
