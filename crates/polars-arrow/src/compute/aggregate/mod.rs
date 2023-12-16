//! Contains different aggregation functions
#[cfg(feature = "compute_aggregate")]
mod sum;
#[cfg(feature = "compute_aggregate")]
pub use sum::*;

mod memory;
pub use memory::*;
#[cfg(feature = "compute_aggregate")]
mod simd;
