mod fx;
mod hasher;
mod identity;
mod partition;
mod vector_hasher;

use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

pub use fx::*;
pub use hasher::*;
pub use identity::*;
pub(crate) use partition::*;
pub use vector_hasher::*;

use crate::prelude::*;

// hash combine from c++' boost lib
#[inline]
pub fn _boost_hash_combine(l: u64, r: u64) -> u64 {
    l ^ r.wrapping_add(0x9e3779b9u64.wrapping_add(l << 6).wrapping_add(r >> 2))
}
