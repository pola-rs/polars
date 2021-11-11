use crate::prelude::{BooleanChunked, ChunkedArray, PolarsNumericType};
use arrow::array::{BooleanArray, PrimitiveArray, Utf8Array};
use std::sync::Arc;

impl<T: PolarsNumericType> From<(&str, PrimitiveArray<T::Native>)> for ChunkedArray<T> {
    fn from(tpl: (&str, PrimitiveArray<T::Native>)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        ChunkedArray::new_from_chunks(name, vec![Arc::new(arr)])
    }
}

impl From<(&str, BooleanArray)> for BooleanChunked {
    fn from(tpl: (&str, BooleanArray)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        ChunkedArray::new_from_chunks(name, vec![Arc::new(arr)])
    }
}

impl From<BooleanArray> for BooleanChunked {
    fn from(arr: BooleanArray) -> Self {
        ChunkedArray::new_from_chunks("", vec![Arc::new(arr)])
    }
}

impl From<(&str, Utf8Array<i64>)> for BooleanChunked {
    fn from(tpl: (&str, Utf8Array<i64>)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        ChunkedArray::new_from_chunks(name, vec![Arc::new(arr)])
    }
}
