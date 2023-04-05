use std::hash::{Hash, Hasher};

use super::*;

#[derive(Copy, Clone)]
pub(super) struct Key {
    pub(super) hash: u64,
    pub(super) idx: IdxSize,
}

impl Key {
    #[inline]
    pub(super) fn new(hash: u64, idx: IdxSize) -> Self {
        Self { hash, idx }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}
