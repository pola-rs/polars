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
pub(crate) fn chunk_size(n_cols: usize) -> usize {
    std::cmp::max(400_000 / n_cols, 1000)
}
