use polars_utils::pl_str::PlSmallStr;

use crate::frame::DataFrame;
use crate::prelude::StructChunked;

impl DataFrame {
    pub fn into_struct(self, name: PlSmallStr) -> StructChunked {
        StructChunked::from_series(name, &self.columns).expect("same invariants")
    }
}
