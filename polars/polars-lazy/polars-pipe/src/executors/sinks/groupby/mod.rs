pub(crate) mod aggregates;
mod generic;
mod primitive;
mod utils;

pub(crate) use generic::*;
pub(crate) use primitive::*;

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
const HASHMAP_INIT_SIZE: usize = 64;
