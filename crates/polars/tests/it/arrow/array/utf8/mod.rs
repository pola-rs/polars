use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;
use polars_error::PolarsResult;

mod mutable;
mod mutable_values;
mod to_mutable;

fn array() -> Utf8Array<i32> {
    vec![Some("hello"), None, Some("hello2")]
        .into_iter()
        .collect()
}

#[test]
fn basics() {
    let array = array();

    assert_eq!(array.value(0), "hello");
    assert_eq!(array.value(1), "");
    assert_eq!(array.value(2), "hello2");
    assert_eq!(unsafe { array.value_unchecked(2) }, "hello2");
    assert_eq!(array.values().as_slice(), b"hellohello2");
    assert_eq!(array.offsets().as_slice(), &[0, 5, 5, 11]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00000101], 3))
    );
    assert!(array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(array.is_valid(2));

    let array2 = Utf8Array::<i32>::new(
        ArrowDataType::Utf8,
        array.offsets().clone(),
        array.values().clone(),
        array.validity().cloned(),
    );
    assert_eq!(array, array2);

    let array = array.sliced(1, 2);
    assert_eq!(array.value(0), "");
    assert_eq!(array.value(1), "hello2");
    // note how this keeps everything: the offsets were sliced
    assert_eq!(array.values().as_slice(), b"hellohello2");
    assert_eq!(array.offsets().as_slice(), &[5, 5, 11]);
}

#[test]
fn split_at() {
    let (lhs, rhs) = array().split_at(1);

    assert_eq!(lhs.value(0), "hello");
    assert_eq!(rhs.value(0), "");
    assert_eq!(rhs.value(1), "hello2");
    // note how this keeps everything: the offsets were sliced
    assert_eq!(lhs.values().as_slice(), b"hellohello2");
    assert_eq!(rhs.values().as_slice(), b"hellohello2");
    assert_eq!(lhs.offsets().as_slice(), &[0, 5]);
    assert_eq!(rhs.offsets().as_slice(), &[5, 5, 11]);
}

#[test]
fn empty() {
    let array = Utf8Array::<i32>::new_empty(ArrowDataType::Utf8);
    assert_eq!(array.values().as_slice(), b"");
    assert_eq!(array.offsets().as_slice(), &[0]);
    assert_eq!(array.validity(), None);
}

#[test]
fn from() {
    let array = Utf8Array::<i32>::from([Some("hello"), Some(" "), None]);

    let a = array.validity().unwrap();
    assert_eq!(a, &Bitmap::from([true, true, false]));
}

#[test]
fn from_slice() {
    let b = Utf8Array::<i32>::from_slice(["a", "b", "cc"]);

    let offsets = vec![0, 1, 2, 4].try_into().unwrap();
    let values = b"abcc".to_vec().into();
    assert_eq!(
        b,
        Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None)
    );
}

#[test]
fn from_iter_values() {
    let b = Utf8Array::<i32>::from_iter_values(["a", "b", "cc"].iter());

    let offsets = vec![0, 1, 2, 4].try_into().unwrap();
    let values = b"abcc".to_vec().into();
    assert_eq!(
        b,
        Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None)
    );
}

#[test]
fn from_trusted_len_iter() {
    let b =
        Utf8Array::<i32>::from_trusted_len_iter(vec![Some("a"), Some("b"), Some("cc")].into_iter());

    let offsets = vec![0, 1, 2, 4].try_into().unwrap();
    let values = b"abcc".to_vec().into();
    assert_eq!(
        b,
        Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None)
    );
}

#[test]
fn try_from_trusted_len_iter() {
    let b = Utf8Array::<i32>::try_from_trusted_len_iter(
        vec![Some("a"), Some("b"), Some("cc")]
            .into_iter()
            .map(PolarsResult::Ok),
    )
    .unwrap();

    let offsets = vec![0, 1, 2, 4].try_into().unwrap();
    let values = b"abcc".to_vec().into();
    assert_eq!(
        b,
        Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None)
    );
}

#[test]
fn not_utf8() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = vec![0, 159, 146, 150].into(); // invalid utf8
    assert!(Utf8Array::<i32>::try_new(ArrowDataType::Utf8, offsets, values, None).is_err());
}

#[test]
fn not_utf8_individually() {
    let offsets = vec![0, 1, 2].try_into().unwrap();
    let values = vec![207, 128].into(); // each is invalid utf8, but together is valid
    assert!(Utf8Array::<i32>::try_new(ArrowDataType::Utf8, offsets, values, None).is_err());
}

#[test]
fn wrong_data_type() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = b"abbb".to_vec().into();
    assert!(Utf8Array::<i32>::try_new(ArrowDataType::Int32, offsets, values, None).is_err());
}

#[test]
fn out_of_bounds_offsets_panics() {
    // the 10 is out of bounds
    let offsets = vec![0, 10, 11].try_into().unwrap();
    let values = b"abbb".to_vec().into();
    assert!(Utf8Array::<i32>::try_new(ArrowDataType::Utf8, offsets, values, None).is_err());
}

#[test]
#[should_panic]
fn index_out_of_bounds_panics() {
    let offsets = vec![0, 1, 2, 4].try_into().unwrap();
    let values = b"abbb".to_vec().into();
    let array = Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None);

    array.value(3);
}

#[test]
fn debug() {
    let array = Utf8Array::<i32>::from([Some("aa"), Some(""), None]);

    assert_eq!(format!("{array:?}"), "Utf8Array[aa, , None]");
}

#[test]
fn into_mut_1() {
    let offsets = vec![0, 1].try_into().unwrap();
    let values = Buffer::from(b"a".to_vec());
    let a = values.clone(); // cloned values
    assert_eq!(a, values);
    let array = Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_2() {
    let offsets: OffsetsBuffer<i32> = vec![0, 1].try_into().unwrap();
    let values = b"a".to_vec().into();
    let a = offsets.clone(); // cloned offsets
    assert_eq!(a, offsets);
    let array = Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, None);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_3() {
    let offsets = vec![0, 1].try_into().unwrap();
    let values = b"a".to_vec().into();
    let validity = Some([true].into());
    let a = validity.clone(); // cloned validity
    assert_eq!(a, validity);
    let array = Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, validity);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_4() {
    let offsets = vec![0, 1].try_into().unwrap();
    let values = b"a".to_vec().into();
    let validity = Some([true].into());
    let array = Utf8Array::<i32>::new(ArrowDataType::Utf8, offsets, values, validity);
    assert!(array.into_mut().is_right());
}

#[test]
fn rev_iter() {
    let array = Utf8Array::<i32>::from([Some("hello"), Some(" "), None]);

    assert_eq!(
        array.into_iter().rev().collect::<Vec<_>>(),
        vec![None, Some(" "), Some("hello")]
    );
}

#[test]
fn iter_nth() {
    let array = Utf8Array::<i32>::from([Some("hello"), Some(" "), None]);

    assert_eq!(array.iter().nth(1), Some(Some(" ")));
    assert_eq!(array.iter().nth(10), None);
}

#[test]
fn test_apply_validity() {
    let mut array = Utf8Array::<i32>::from([Some("Red"), Some("Green"), Some("Blue")]);
    array.set_validity(Some([true, true, true].into()));

    array.apply_validity(|bitmap| {
        let mut mut_bitmap = bitmap.into_mut().right().unwrap();
        mut_bitmap.set(1, false);
        mut_bitmap.set(2, false);
        mut_bitmap.into()
    });

    assert!(array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(!array.is_valid(2));
}
