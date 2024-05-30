use arrow::array::{Array, BinaryArray, Splitable};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;
use polars_error::PolarsResult;

mod mutable;
mod mutable_values;
mod to_mutable;

fn array() -> BinaryArray<i32> {
    vec![Some(b"hello".to_vec()), None, Some(b"hello2".to_vec())]
        .into_iter()
        .collect()
}

#[test]
fn basics() {
    let array = array();

    assert_eq!(array.value(0), b"hello");
    assert_eq!(array.value(1), b"");
    assert_eq!(array.value(2), b"hello2");
    assert_eq!(unsafe { array.value_unchecked(2) }, b"hello2");
    assert_eq!(array.values().as_slice(), b"hellohello2");
    assert_eq!(array.offsets().as_slice(), &[0, 5, 5, 11]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00000101], 3))
    );
    assert!(array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(array.is_valid(2));

    let array2 = BinaryArray::<i32>::new(
        ArrowDataType::Binary,
        array.offsets().clone(),
        array.values().clone(),
        array.validity().cloned(),
    );
    assert_eq!(array, array2);

    let array = array.sliced(1, 2);
    assert_eq!(array.value(0), b"");
    assert_eq!(array.value(1), b"hello2");
    // note how this keeps everything: the offsets were sliced
    assert_eq!(array.values().as_slice(), b"hellohello2");
    assert_eq!(array.offsets().as_slice(), &[5, 5, 11]);
}

#[test]
fn split_at() {
    let (lhs, rhs) = array().split_at(1);

    assert_eq!(lhs.value(0), b"hello");
    assert_eq!(rhs.value(0), b"");
    assert_eq!(rhs.value(1), b"hello2");

    // note how this keeps everything: the offsets were sliced
    assert_eq!(lhs.values().as_slice(), b"hellohello2");
    assert_eq!(rhs.values().as_slice(), b"hellohello2");
    assert_eq!(lhs.offsets().as_slice(), &[0, 5]);
    assert_eq!(rhs.offsets().as_slice(), &[5, 5, 11]);
    assert_eq!(lhs.validity().map_or(0, |v| v.set_bits()), 0);
    assert_eq!(rhs.validity().map_or(0, |v| v.set_bits()), 1);
}

#[test]
fn empty() {
    let array = BinaryArray::<i32>::new_empty(ArrowDataType::Binary);
    assert_eq!(array.values().as_slice(), b"");
    assert_eq!(array.offsets().as_slice(), &[0]);
    assert_eq!(array.validity(), None);
}

#[test]
fn from() {
    let array = BinaryArray::<i32>::from([Some(b"hello".as_ref()), Some(b" ".as_ref()), None]);

    let a = array.validity().unwrap();
    assert_eq!(a, &Bitmap::from([true, true, false]));
}

#[test]
fn from_trusted_len_iter() {
    let iter = std::iter::repeat(b"hello").take(2).map(Some);
    let a = BinaryArray::<i32>::from_trusted_len_iter(iter);
    assert_eq!(a.len(), 2);
}

#[test]
fn try_from_trusted_len_iter() {
    let iter = std::iter::repeat(b"hello".as_ref())
        .take(2)
        .map(Some)
        .map(PolarsResult::Ok);
    let a = BinaryArray::<i32>::try_from_trusted_len_iter(iter).unwrap();
    assert_eq!(a.len(), 2);
}

#[test]
fn from_iter() {
    let iter = std::iter::repeat(b"hello").take(2).map(Some);
    let a: BinaryArray<i32> = iter.collect();
    assert_eq!(a.len(), 2);
}

#[test]
fn with_validity() {
    let array = BinaryArray::<i32>::from([Some(b"hello".as_ref()), Some(b" ".as_ref()), None]);

    let array = array.with_validity(None);

    let a = array.validity();
    assert_eq!(a, None);
}

#[test]
#[should_panic]
fn wrong_offsets() {
    let offsets = vec![0, 5, 4].try_into().unwrap(); // invalid offsets
    let values = Buffer::from(b"abbbbb".to_vec());
    BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, None);
}

#[test]
#[should_panic]
fn wrong_data_type() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = Buffer::from(b"abbb".to_vec());
    BinaryArray::<i32>::new(ArrowDataType::Int8, offsets, values, None);
}

#[test]
#[should_panic]
fn value_with_wrong_offsets_panics() {
    let offsets = vec![0, 10, 11, 4].try_into().unwrap();
    let values = Buffer::from(b"abbb".to_vec());
    // the 10-11 is not checked
    let array = BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, None);

    // but access is still checked (and panics)
    // without checks, this would result in reading beyond bounds
    array.value(0);
}

#[test]
#[should_panic]
fn index_out_of_bounds_panics() {
    let offsets = vec![0, 1, 2, 4].try_into().unwrap();
    let values = Buffer::from(b"abbb".to_vec());
    let array = BinaryArray::<i32>::new(ArrowDataType::Utf8, offsets, values, None);

    array.value(3);
}

#[test]
#[should_panic]
fn value_unchecked_with_wrong_offsets_panics() {
    let offsets = vec![0, 10, 11, 4].try_into().unwrap();
    let values = Buffer::from(b"abbb".to_vec());
    // the 10-11 is not checked
    let array = BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, None);

    // but access is still checked (and panics)
    // without checks, this would result in reading beyond bounds,
    // even if `0` is in bounds
    unsafe { array.value_unchecked(0) };
}

#[test]
fn debug() {
    let array = BinaryArray::<i32>::from([Some([1, 2].as_ref()), Some(&[]), None]);

    assert_eq!(format!("{array:?}"), "BinaryArray[[1, 2], [], None]");
}

#[test]
fn into_mut_1() {
    let offsets = vec![0, 1].try_into().unwrap();
    let values = Buffer::from(b"a".to_vec());
    let a = values.clone(); // cloned values
    assert_eq!(a, values);
    let array = BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, None);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_2() {
    let offsets: OffsetsBuffer<i32> = vec![0, 1].try_into().unwrap();
    let values = Buffer::from(b"a".to_vec());
    let a = offsets.clone(); // cloned offsets
    assert_eq!(a, offsets);
    let array = BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, None);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_3() {
    let offsets = vec![0, 1].try_into().unwrap();
    let values = Buffer::from(b"a".to_vec());
    let validity = Some([true].into());
    let a = validity.clone(); // cloned validity
    assert_eq!(a, validity);
    let array = BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, validity);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_4() {
    let offsets = vec![0, 1].try_into().unwrap();
    let values = Buffer::from(b"a".to_vec());
    let validity = Some([true].into());
    let array = BinaryArray::<i32>::new(ArrowDataType::Binary, offsets, values, validity);
    assert!(array.into_mut().is_right());
}

#[test]
fn rev_iter() {
    let array = BinaryArray::<i32>::from([Some("hello".as_bytes()), Some(" ".as_bytes()), None]);

    assert_eq!(
        array.into_iter().rev().collect::<Vec<_>>(),
        vec![None, Some(" ".as_bytes()), Some("hello".as_bytes())]
    );
}

#[test]
fn iter_nth() {
    let array = BinaryArray::<i32>::from([Some("hello"), Some(" "), None]);

    assert_eq!(array.iter().nth(1), Some(Some(" ".as_bytes())));
    assert_eq!(array.iter().nth(10), None);
}
