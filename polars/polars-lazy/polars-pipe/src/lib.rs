mod executors;
pub mod expressions;
pub mod operators;
pub mod pipeline;

// ideal chunk size we strive to
#[cfg(any(feature = "cross_join", feature = "csv-file"))]
pub(crate) const CHUNK_SIZE: usize = 50_000;
