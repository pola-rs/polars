use arrow::array::growable::{Growable, GrowableNull};
use arrow::array::NullArray;
use arrow::datatypes::ArrowDataType;

#[test]
fn null() {
    let mut mutable = GrowableNull::default();

    unsafe {
        mutable.extend(0, 1, 2);
    }
    unsafe {
        mutable.extend(1, 0, 1);
    }
    assert_eq!(mutable.len(), 3);

    let result: NullArray = mutable.into();

    let expected = NullArray::new(ArrowDataType::Null, 3);
    assert_eq!(result, expected);
}
