mod identity;
pub(crate) mod vector_hasher;

use std::hash::{BuildHasherDefault, Hash, Hasher};

pub use identity::*;
pub use vector_hasher::*;

// We must strike a balance between cache
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub const _HASHMAP_INIT_SIZE: usize = 512;
