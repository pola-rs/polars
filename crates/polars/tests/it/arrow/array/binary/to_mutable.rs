use arrow::array::BinaryArray;
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;

#[test]
fn not_shared() {
    let array = BinaryArray::<i32>::from([Some("hello"), Some(" "), None]);
    assert!(array.into_mut().is_right());
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_validity() {
    let validity = Bitmap::from([true]);
    let array = BinaryArray::<i32>::new(
        ArrowDataType::Binary,
        vec![0, 1].try_into().unwrap(),
        b"a".to_vec().into(),
        Some(validity.clone()),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_values() {
    let values: Buffer<u8> = b"a".to_vec().into();
    let array = BinaryArray::<i32>::new(
        ArrowDataType::Binary,
        vec![0, 1].try_into().unwrap(),
        values.clone(),
        Some(Bitmap::from([true])),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_offsets_values() {
    let offsets: Buffer<i32> = vec![0, 1].into();
    let values: Buffer<u8> = b"a".to_vec().into();
    let array = BinaryArray::<i32>::new(
        ArrowDataType::Binary,
        offsets.clone().try_into().unwrap(),
        values.clone(),
        Some(Bitmap::from([true])),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_offsets() {
    let offsets: Buffer<i32> = vec![0, 1].into();
    let array = BinaryArray::<i32>::new(
        ArrowDataType::Binary,
        offsets.clone().try_into().unwrap(),
        b"a".to_vec().into(),
        Some(Bitmap::from([true])),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_all() {
    let array = BinaryArray::<i32>::from([Some("hello"), Some(" "), None]);
    assert!(array.clone().into_mut().is_left())
}
