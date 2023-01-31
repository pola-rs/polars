use arrow::array::{BooleanArray, PrimitiveArray, Utf8Array};

use crate::prelude::*;

impl<T: PolarsNumericType> From<(&str, PrimitiveArray<T::Native>)> for ChunkedArray<T> {
    fn from(tpl: (&str, PrimitiveArray<T::Native>)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        // safety: same type
        unsafe { ChunkedArray::from_chunks(name, vec![Box::new(arr)]) }
    }
}

impl<T: PolarsNumericType> From<&[T::Native]> for ChunkedArray<T> {
    fn from(slice: &[T::Native]) -> Self {
        ChunkedArray::from_slice("", slice)
    }
}

impl From<(&str, BooleanArray)> for BooleanChunked {
    fn from(tpl: (&str, BooleanArray)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        // safety: same type
        unsafe { ChunkedArray::from_chunks(name, vec![Box::new(arr)]) }
    }
}

impl From<BooleanArray> for BooleanChunked {
    fn from(arr: BooleanArray) -> Self {
        // safety: same type
        unsafe { ChunkedArray::from_chunks("", vec![Box::new(arr)]) }
    }
}

impl From<(&str, Utf8Array<i64>)> for Utf8Chunked {
    fn from(tpl: (&str, Utf8Array<i64>)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        // safety: same type
        unsafe { ChunkedArray::from_chunks(name, vec![Box::new(arr)]) }
    }
}

#[cfg(feature = "dtype-binary")]
impl From<(&str, BinaryArray<i64>)> for BinaryChunked {
    fn from(tpl: (&str, BinaryArray<i64>)) -> Self {
        let name = tpl.0;
        let arr = tpl.1;

        // safety: same type
        unsafe { ChunkedArray::from_chunks(name, vec![Box::new(arr)]) }
    }
}
