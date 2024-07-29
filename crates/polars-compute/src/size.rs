use arrow::array::{Array, ArrayRef, BinaryViewArray, UInt32Array};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;

pub fn binary_size_bytes(array: &BinaryViewArray) -> ArrayRef {
    let values: Buffer<_> = array.len_iter().collect();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
