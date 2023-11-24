use crate::array::{ArrayRef, UInt32Array, Utf8Array};
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::legacy::trusted_len::TrustedLenPush;

pub fn string_len_bytes(array: &Utf8Array<i64>) -> ArrayRef {
    let values = array
        .offsets()
        .as_slice()
        .windows(2)
        .map(|x| (x[1] - x[0]) as u32);
    let values: Buffer<_> = Vec::from_trusted_len_iter(values).into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}

pub fn string_len_chars(array: &Utf8Array<i64>) -> ArrayRef {
    let values = array.values_iter().map(|x| x.chars().count() as u32);
    let values: Buffer<_> = Vec::from_trusted_len_iter(values).into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
