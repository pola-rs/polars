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

/// ideal chunk size we strive to have
/// scale the chunk size depending on the number of
/// columns. With 10 columns we use a chunk size of 40_000
#[cfg(feature = "compile")]
pub(crate) fn determine_chunk_size(n_cols: usize, n_threads: usize) -> usize {
    let thread_factor = std::cmp::max(12 / n_threads, 1);
    std::cmp::max(400_000 / n_cols * thread_factor, 1000)
}
