//! Special list utility methods
mod iterator;
#[cfg(feature = "list")]
pub mod namespace;

use crate::prelude::*;

impl ListChunked {
    pub(crate) fn set_fast_explode(&mut self) {
        self.bit_settings |= 1 << 2;
    }

    pub(crate) fn can_fast_explode(&self) -> bool {
        self.bit_settings & 1 << 2 != 0
    }
}
