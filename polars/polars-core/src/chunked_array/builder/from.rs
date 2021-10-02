use crate::prelude::{ChunkedArray, PolarsNumericType};
use arrow::array::PrimitiveArray;
use std::sync::Arc;

impl<T: PolarsNumericType> From<(&str, PrimitiveArray<T::Native>)> for ChunkedArray<T> {
    fn from(tpl: (&str, PrimitiveArray<T::Native>)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        ChunkedArray::new_from_chunks(name, vec![Arc::new(arr)])
    }
}
