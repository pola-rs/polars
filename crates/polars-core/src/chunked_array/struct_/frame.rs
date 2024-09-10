use polars_utils::pl_str::PlSmallStr;

use crate::frame::DataFrame;
use crate::prelude::StructChunked;

impl DataFrame {
    pub fn into_struct(self, name: PlSmallStr) -> StructChunked {
        // @scalar-opt
        let series = self.materialized_column_iter().cloned().collect::<Vec<_>>();
        StructChunked::from_series(name, &series).expect("same invariants")
    }
}
