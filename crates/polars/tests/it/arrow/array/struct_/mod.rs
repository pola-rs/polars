mod iterator;
mod mutable;

use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::datatypes::*;

#[test]
fn debug() {
    let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
    let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

    let fields = vec![
        Field::new("b", ArrowDataType::Boolean, false),
        Field::new("c", ArrowDataType::Int32, false),
    ];

    let array = StructArray::new(
        ArrowDataType::Struct(fields),
        vec![boolean.clone(), int.clone()],
        Some(Bitmap::from([true, true, false, true])),
    );
    assert_eq!(
        format!("{array:?}"),
        "StructArray[{b: false, c: 42}, {b: false, c: 28}, None, {b: true, c: 31}]"
    );
}
