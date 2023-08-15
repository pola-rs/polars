use arrow::array::{BooleanArray, PrimitiveArray, Utf8Array};

use crate::prelude::*;

impl<T: PolarsNumericType> From<(&str, PrimitiveArray<T::Native>)> for ChunkedArray<T> {
    fn from(tpl: (&str, PrimitiveArray<T::Native>)) -> Self {
        ChunkedArray::from_chunk_iter(tpl.0, [tpl.1])
    }
}

impl<T: PolarsNumericType> From<&[T::Native]> for ChunkedArray<T> {
    fn from(slice: &[T::Native]) -> Self {
        ChunkedArray::from_slice("", slice)
    }
}

impl From<(&str, BooleanArray)> for BooleanChunked {
    fn from(tpl: (&str, BooleanArray)) -> Self {
        ChunkedArray::from_chunk_iter(tpl.0, [tpl.1])
    }
}

impl From<BooleanArray> for BooleanChunked {
    fn from(arr: BooleanArray) -> Self {
        ChunkedArray::from_chunk_iter("", [arr])
    }
}

impl From<(&str, Utf8Array<i64>)> for Utf8Chunked {
    fn from(tpl: (&str, Utf8Array<i64>)) -> Self {
        ChunkedArray::from_chunk_iter(tpl.0, [tpl.1])
    }
}

impl From<Utf8Array<i64>> for Utf8Chunked {
    fn from(arr: Utf8Array<i64>) -> Self {
        ChunkedArray::from_chunk_iter("", [arr])
    }
}

impl From<(&str, BinaryArray<i64>)> for BinaryChunked {
    fn from(tpl: (&str, BinaryArray<i64>)) -> Self {
        ChunkedArray::from_chunk_iter(tpl.0, [tpl.1])
    }
}

impl From<BinaryArray<i64>> for BinaryChunked {
    fn from(arr: BinaryArray<i64>) -> Self {
        ChunkedArray::from_chunk_iter("", [arr])
    }
}

impl From<(&str, ListArray<i64>)> for ListChunked {
    fn from(tpl: (&str, ListArray<i64>)) -> Self {
        ChunkedArray::from_chunk_iter(tpl.0, [tpl.1])
    }
}

impl From<ListArray<i64>> for ListChunked {
    fn from(arr: ListArray<i64>) -> Self {
        ChunkedArray::from_chunk_iter("", [arr])
    }
}
