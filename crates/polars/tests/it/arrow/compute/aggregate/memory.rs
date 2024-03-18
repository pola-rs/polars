use arrow::array::*;
use arrow::compute::aggregate::estimated_bytes_size;
use arrow::datatypes::{ArrowDataType, Field};

#[test]
fn primitive() {
    let a = Int32Array::from_slice([1, 2, 3, 4, 5]);
    assert_eq!(5 * std::mem::size_of::<i32>(), estimated_bytes_size(&a));
}

#[test]
fn boolean() {
    let a = BooleanArray::from_slice([true]);
    assert_eq!(1, estimated_bytes_size(&a));
}

#[test]
fn utf8() {
    let a = Utf8Array::<i32>::from_slice(["aaa"]);
    assert_eq!(3 + 2 * std::mem::size_of::<i32>(), estimated_bytes_size(&a));
}

#[test]
fn fixed_size_list() {
    let data_type = ArrowDataType::FixedSizeList(
        Box::new(Field::new("elem", ArrowDataType::Float32, false)),
        3,
    );
    let values = Box::new(Float32Array::from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let a = FixedSizeListArray::new(data_type, values, None);
    assert_eq!(6 * std::mem::size_of::<f32>(), estimated_bytes_size(&a));
}
