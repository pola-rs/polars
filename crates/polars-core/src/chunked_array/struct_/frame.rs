use polars_utils::pl_str::PlSmallStr;

use crate::frame::DataFrame;
use crate::prelude::StructChunked;

impl DataFrame {
    pub fn into_struct(self, name: PlSmallStr) -> StructChunked {
        StructChunked::from_columns(name, self.height(), &self.columns).expect("same invariants")
    }
}
