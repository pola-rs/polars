pub(crate) mod groupby;
mod joins;
mod ordered;
#[cfg(feature = "parquet")]
mod parquet_sink;
mod slice;
mod utils;

pub(crate) use joins::*;
pub(crate) use ordered::*;
#[cfg(feature = "parquet")]
pub(crate) use parquet_sink::ParquetSink;
pub(crate) use slice::*;

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
const HASHMAP_INIT_SIZE: usize = 64;
