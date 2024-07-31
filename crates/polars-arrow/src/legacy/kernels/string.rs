use crate::array::{Array, ArrayRef, UInt32Array, Utf8ViewArray};
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

pub fn utf8view_len_bytes(array: &Utf8ViewArray) -> ArrayRef {
    let values: Buffer<_> = array.len_iter().collect();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}

pub fn string_len_chars(array: &Utf8ViewArray) -> ArrayRef {
    let values: Buffer<_> = array
        .values_iter()
        .map(|x| x.chars().count() as u32)
        .collect();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
