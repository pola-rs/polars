use crate::array::{Array, ArrayRef, UInt32Array, Utf8ViewArray};
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::legacy::trusted_len::TrustedLenPush;

pub fn utf8view_len_bytes(array: &Utf8ViewArray) -> ArrayRef {
    let values = array.len_iter().collect::<Vec<_>>();
    let values: Buffer<_> = values.into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}

pub fn string_len_chars(array: &Utf8ViewArray) -> ArrayRef {
    let values = array.values_iter().map(|x| x.chars().count() as u32);
    let values: Buffer<_> = Vec::from_trusted_len_iter(values).into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
