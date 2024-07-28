use crate::array::{Array, ArrayRef, BinaryViewArray, UInt32Array};
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

pub fn binary_size_bytes(array: &BinaryViewArray) -> ArrayRef {
    let values = array.len_iter().collect::<Vec<_>>();
    let values: Buffer<_> = values.into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
