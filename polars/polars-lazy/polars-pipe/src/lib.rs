extern crate core;

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
