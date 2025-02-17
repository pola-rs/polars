use arrow::array::Utf8Array;
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;

#[test]
fn not_shared() {
    let array = Utf8Array::<i32>::from([Some("hello"), Some(" "), None]);
    assert!(array.into_mut().is_right());
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_validity() {
    let validity = Bitmap::from([true]);
    let array = Utf8Array::<i32>::new(
        ArrowDataType::Utf8,
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
    let array = Utf8Array::<i32>::new(
        ArrowDataType::Utf8,
        vec![0, 1].try_into().unwrap(),
        values.clone(),
        Some(Bitmap::from([true])),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_offsets_values() {
    let offsets: OffsetsBuffer<i32> = vec![0, 1].try_into().unwrap();
    let values: Buffer<u8> = b"a".to_vec().into();
    let array = Utf8Array::<i32>::new(
        ArrowDataType::Utf8,
        offsets.clone(),
        values.clone(),
        Some(Bitmap::from([true])),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_offsets() {
    let offsets: OffsetsBuffer<i32> = vec![0, 1].try_into().unwrap();
    let array = Utf8Array::<i32>::new(
        ArrowDataType::Utf8,
        offsets.clone(),
        b"a".to_vec().into(),
        Some(Bitmap::from([true])),
    );
    assert!(array.into_mut().is_left())
}

#[test]
#[allow(clippy::redundant_clone)]
fn shared_all() {
    let array = Utf8Array::<i32>::from([Some("hello"), Some(" "), None]);
    assert!(array.clone().into_mut().is_left())
}
