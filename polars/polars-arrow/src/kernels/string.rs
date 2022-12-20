use arrow::array::{UInt32Array, Utf8Array};
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;

use crate::prelude::*;
use crate::trusted_len::PushUnchecked;

pub fn string_lengths(array: &Utf8Array<i64>) -> ArrayRef {
    let values = array
        .offsets()
        .as_slice()
        .windows(2)
        .map(|x| (x[1] - x[0]) as u32);
    let values: Buffer<_> = Vec::from_trusted_len_iter(values).into();
    let array = UInt32Array::new(DataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}

pub fn string_nchars(array: &Utf8Array<i64>) -> ArrayRef {
    let values = array.values_iter().map(|x| x.chars().count() as u32);
    let values: Buffer<_> = Vec::from_trusted_len_iter(values).into();
    let array = UInt32Array::new(DataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
