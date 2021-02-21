use crate::prelude::Arc;
use arrow::array::{ArrayRef, LargeStringArray, UInt32Array};

pub(crate) fn string_lengths(array: &LargeStringArray) -> ArrayRef {
    let array: UInt32Array = array.iter().map(|v| v.map(|v| v.len() as u32)).collect();
    Arc::new(array)
}
