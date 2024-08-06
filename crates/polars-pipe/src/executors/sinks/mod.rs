pub(crate) mod group_by;
mod io;
mod joins;
mod memory;
mod ordered;
mod output;
mod reproject;
mod slice;
mod sort;
mod utils;

pub(crate) use joins::*;
pub(crate) use ordered::*;
#[cfg(any(
    feature = "parquet",
    feature = "ipc",
    feature = "csv",
    feature = "json"
))]
pub(crate) use output::*;
pub(crate) use reproject::*;
pub(crate) use slice::*;
pub(crate) use sort::*;

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
const HASHMAP_INIT_SIZE: usize = 64;
