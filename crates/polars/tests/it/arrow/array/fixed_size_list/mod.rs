mod mutable;

use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field};

fn data() -> FixedSizeListArray {
    let values = Int32Array::from_slice([10, 20, 0, 0]);

    FixedSizeListArray::try_new(
        ArrowDataType::FixedSizeList(
            Box::new(Field::new("a", values.data_type().clone(), true)),
            2,
        ),
        values.boxed(),
        Some([true, false].into()),
    )
    .unwrap()
}

#[test]
fn basics() {
    let array = data();
    assert_eq!(array.size(), 2);
    assert_eq!(array.len(), 2);
    assert_eq!(array.validity(), Some(&Bitmap::from([true, false])));

    assert_eq!(array.value(0).as_ref(), Int32Array::from_slice([10, 20]));
    assert_eq!(array.value(1).as_ref(), Int32Array::from_slice([0, 0]));

    let array = array.sliced(1, 1);

    assert_eq!(array.value(0).as_ref(), Int32Array::from_slice([0, 0]));
}

#[test]
fn split_at() {
    let (lhs, rhs) = data().split_at(1);

    assert_eq!(lhs.value(0).as_ref(), Int32Array::from_slice([10, 20]));
    assert_eq!(rhs.value(0).as_ref(), Int32Array::from_slice([0, 0]));
}

#[test]
fn with_validity() {
    let array = data();

    let a = array.with_validity(None);
    assert!(a.validity().is_none());
}

#[test]
fn debug() {
    let array = data();

    assert_eq!(format!("{array:?}"), "FixedSizeListArray[[10, 20], None]");
}

#[test]
fn empty() {
    let array = FixedSizeListArray::new_empty(ArrowDataType::FixedSizeList(
        Box::new(Field::new("a", ArrowDataType::Int32, true)),
        2,
    ));
    assert_eq!(array.values().len(), 0);
    assert_eq!(array.validity(), None);
}

#[test]
fn null() {
    let array = FixedSizeListArray::new_null(
        ArrowDataType::FixedSizeList(Box::new(Field::new("a", ArrowDataType::Int32, true)), 2),
        2,
    );
    assert_eq!(array.values().len(), 4);
    assert_eq!(array.validity().cloned(), Some([false, false].into()));
}

#[test]
fn wrong_size() {
    let values = Int32Array::from_slice([10, 20, 0]);
    assert!(FixedSizeListArray::try_new(
        ArrowDataType::FixedSizeList(Box::new(Field::new("a", ArrowDataType::Int32, true)), 2),
        values.boxed(),
        None
    )
    .is_err());
}

#[test]
fn wrong_len() {
    let values = Int32Array::from_slice([10, 20, 0]);
    assert!(FixedSizeListArray::try_new(
        ArrowDataType::FixedSizeList(Box::new(Field::new("a", ArrowDataType::Int32, true)), 2),
        values.boxed(),
        Some([true, false, false].into()), // it should be 2
    )
    .is_err());
}

#[test]
fn wrong_data_type() {
    let values = Int32Array::from_slice([10, 20, 0]);
    assert!(FixedSizeListArray::try_new(
        ArrowDataType::Binary,
        values.boxed(),
        Some([true, false, false].into()), // it should be 2
    )
    .is_err());
}
