#[cfg(any(feature = "parquet", feature = "ipc"))]
mod file_sink;
pub(crate) mod groupby;
mod io;
mod joins;
mod memory;
mod ordered;
mod reproject;
mod slice;
mod sort;
mod utils;

#[cfg(any(feature = "parquet", feature = "ipc"))]
pub(crate) use file_sink::*;
pub(crate) use joins::*;
pub(crate) use ordered::*;
pub(crate) use reproject::*;
pub(crate) use slice::*;
pub(crate) use sort::*;

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
const HASHMAP_INIT_SIZE: usize = 64;
