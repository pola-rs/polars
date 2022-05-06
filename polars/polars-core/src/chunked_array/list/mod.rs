//! Special list utility methods
mod iterator;
#[cfg(feature = "list")]
#[cfg_attr(docsrs, doc(cfg(feature = "list")))]
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

    pub fn to_logical(&mut self, inner_dtype: DataType) {
        assert_eq!(inner_dtype.to_physical(), self.inner_dtype());
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::List(Box::new(inner_dtype)))
    }
}
