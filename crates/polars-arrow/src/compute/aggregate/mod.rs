//! Contains different aggregation functions
#[cfg(feature = "compute_aggregate")]
mod sum;
#[cfg(feature = "compute_aggregate")]
pub use sum::*;

#[cfg(feature = "compute_aggregate")]
mod min_max;
#[cfg(feature = "compute_aggregate")]
pub use min_max::*;

mod memory;
pub use memory::*;
#[cfg(feature = "compute_aggregate")]
mod simd;
