pub mod arena;
pub mod contention_pool;
mod error;
pub mod mem;
pub mod sort;

#[cfg(not(feature = "bigint"))]
pub type IdxSize = u32;
#[cfg(feature = "bigint")]
pub type IdxSize = u64;
