use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;

mod mutable;

#[test]
fn basics() {
    let array = FixedSizeBinaryArray::new(
        ArrowDataType::FixedSizeBinary(2),
        Buffer::from(vec![1, 2, 3, 4, 5, 6]),
        Some(Bitmap::from([true, false, true])),
    );
    assert_eq!(array.size(), 2);
    assert_eq!(array.len(), 3);
    assert_eq!(array.validity(), Some(&Bitmap::from([true, false, true])));

    assert_eq!(array.value(0), [1, 2]);
    assert_eq!(array.value(2), [5, 6]);

    let array = array.sliced(1, 2);

    assert_eq!(array.value(1), [5, 6]);
}

#[test]
fn with_validity() {
    let a = FixedSizeBinaryArray::new(
        ArrowDataType::FixedSizeBinary(2),
        vec![1, 2, 3, 4, 5, 6].into(),
        None,
    );
    let a = a.with_validity(Some(Bitmap::from([true, false, true])));
    assert!(a.validity().is_some());
}

#[test]
fn debug() {
    let a = FixedSizeBinaryArray::new(
        ArrowDataType::FixedSizeBinary(2),
        vec![1, 2, 3, 4, 5, 6].into(),
        Some(Bitmap::from([true, false, true])),
    );
    assert_eq!(format!("{a:?}"), "FixedSizeBinary(2)[[1, 2], None, [5, 6]]");
}

#[test]
fn empty() {
    let array = FixedSizeBinaryArray::new_empty(ArrowDataType::FixedSizeBinary(2));
    assert_eq!(array.values().len(), 0);
    assert_eq!(array.validity(), None);
}

#[test]
fn null() {
    let array = FixedSizeBinaryArray::new_null(ArrowDataType::FixedSizeBinary(2), 2);
    assert_eq!(array.values().len(), 4);
    assert_eq!(array.validity().cloned(), Some([false, false].into()));
}

#[test]
fn from_iter() {
    let iter = std::iter::repeat(vec![1u8, 2]).take(2).map(Some);
    let a = FixedSizeBinaryArray::from_iter(iter, 2);
    assert_eq!(a.len(), 2);
}

#[test]
fn wrong_size() {
    let values = Buffer::from(b"abb".to_vec());
    assert!(
        FixedSizeBinaryArray::try_new(ArrowDataType::FixedSizeBinary(2), values, None).is_err()
    );
}

#[test]
fn wrong_len() {
    let values = Buffer::from(b"abba".to_vec());
    let validity = Some([true, false, false].into()); // it should be 2
    assert!(
        FixedSizeBinaryArray::try_new(ArrowDataType::FixedSizeBinary(2), values, validity).is_err()
    );
}

#[test]
fn wrong_data_type() {
    let values = Buffer::from(b"abba".to_vec());
    assert!(FixedSizeBinaryArray::try_new(ArrowDataType::Binary, values, None).is_err());
}

#[test]
fn to() {
    let values = Buffer::from(b"abba".to_vec());
    let a = FixedSizeBinaryArray::new(ArrowDataType::FixedSizeBinary(2), values, None);

    let extension = ArrowDataType::Extension(
        "a".to_string(),
        Box::new(ArrowDataType::FixedSizeBinary(2)),
        None,
    );
    let _ = a.to(extension);
}
