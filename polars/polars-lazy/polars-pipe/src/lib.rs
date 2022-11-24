#[cfg(feature = "compile")]
mod executors;
#[cfg(feature = "compile")]
pub mod expressions;
#[cfg(feature = "compile")]
pub mod operators;
#[cfg(feature = "compile")]
pub mod pipeline;

// ideal chunk size we strive to
#[cfg(feature = "compile")]
pub(crate) const CHUNK_SIZE: usize = 50_000;
#[cfg(feature = "compile")]
pub use operators::SExecutionContext;
