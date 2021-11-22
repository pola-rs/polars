//! Special list utility methods
mod iterator;
#[cfg(feature = "list")]
pub mod namespace;

use crate::prelude::*;

impl ListChunked {
    #[cfg(feature = "private")]
    pub fn set_fast_explode(&mut self) {
        self.bit_settings |= 1 << 2;
    }

    pub(crate) fn can_fast_explode(&self) -> bool {
        self.bit_settings & 1 << 2 != 0
    }

    pub(crate) fn is_nested(&self) -> bool {
        match self.dtype() {
            DataType::List(inner) => matches!(&**inner, DataType::List(_)),
            _ => unreachable!(),
        }
    }
}
