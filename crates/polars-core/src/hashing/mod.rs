mod identity;
pub(crate) mod vector_hasher;

use std::hash::{BuildHasherDefault, Hash, Hasher};

pub use identity::*;
pub use vector_hasher::*;

// hash combine from c++' boost lib
#[inline]
pub fn _boost_hash_combine(l: u64, r: u64) -> u64 {
    l ^ r.wrapping_add(0x9e3779b9u64.wrapping_add(l << 6).wrapping_add(r >> 2))
}

// We must strike a balance between cache
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub const _HASHMAP_INIT_SIZE: usize = 512;
