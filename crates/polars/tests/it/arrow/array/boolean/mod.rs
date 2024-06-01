use arrow::array::{Array, BooleanArray, Splitable};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

mod mutable;

fn array() -> BooleanArray {
    vec![Some(true), None, Some(false)].into_iter().collect()
}

#[test]
fn basics() {
    let array = array();

    assert_eq!(array.data_type(), &ArrowDataType::Boolean);

    assert!(array.value(0));
    assert!(!array.value(1));
    assert!(!array.value(2));
    assert!(!unsafe { array.value_unchecked(2) });
    assert_eq!(array.values(), &Bitmap::from_u8_slice([0b00000001], 3));
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00000101], 3))
    );
    assert!(array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(array.is_valid(2));

    let array2 = BooleanArray::new(
        ArrowDataType::Boolean,
        array.values().clone(),
        array.validity().cloned(),
    );
    assert_eq!(array, array2);

    let array = array.sliced(1, 2);
    assert!(!array.value(0));
    assert!(!array.value(1));
}

#[test]
fn split_at() {
    let (lhs, rhs) = array().split_at(1);

    assert!(lhs.is_valid(0));
    assert!(!rhs.is_valid(0));
    assert!(rhs.is_valid(1));

    assert!(lhs.value(0));
    assert!(!rhs.value(0));
    assert!(!rhs.value(1));
}

#[test]
fn try_new_invalid() {
    assert!(BooleanArray::try_new(ArrowDataType::Int32, [true].into(), None).is_err());
    assert!(BooleanArray::try_new(
        ArrowDataType::Boolean,
        [true].into(),
        Some([false, true].into())
    )
    .is_err());
}

#[test]
fn with_validity() {
    let bitmap = Bitmap::from([true, false, true]);
    let a = BooleanArray::new(ArrowDataType::Boolean, bitmap, None);
    let a = a.with_validity(Some(Bitmap::from([true, false, true])));
    assert!(a.validity().is_some());
}

#[test]
fn debug() {
    let array = BooleanArray::from([Some(true), None, Some(false)]);
    assert_eq!(format!("{array:?}"), "BooleanArray[true, None, false]");
}

#[test]
fn into_mut_valid() {
    let bitmap = Bitmap::from([true, false, true]);
    let a = BooleanArray::new(ArrowDataType::Boolean, bitmap, None);
    let _ = a.into_mut().right().unwrap();

    let bitmap = Bitmap::from([true, false, true]);
    let validity = Bitmap::from([true, false, true]);
    let a = BooleanArray::new(ArrowDataType::Boolean, bitmap, Some(validity));
    let _ = a.into_mut().right().unwrap();
}

#[test]
fn into_mut_invalid() {
    let bitmap = Bitmap::from([true, false, true]);
    let _other = bitmap.clone(); // values is shared
    let a = BooleanArray::new(ArrowDataType::Boolean, bitmap, None);
    let _ = a.into_mut().left().unwrap();

    let bitmap = Bitmap::from([true, false, true]);
    let validity = Bitmap::from([true, false, true]);
    let _other = validity.clone(); // validity is shared
    let a = BooleanArray::new(ArrowDataType::Boolean, bitmap, Some(validity));
    let _ = a.into_mut().left().unwrap();
}

#[test]
fn empty() {
    let array = BooleanArray::new_empty(ArrowDataType::Boolean);
    assert_eq!(array.values().len(), 0);
    assert_eq!(array.validity(), None);
}

#[test]
fn from_trusted_len_iter() {
    let iter = std::iter::repeat(true).take(2).map(Some);
    let a = BooleanArray::from_trusted_len_iter(iter.clone());
    assert_eq!(a.len(), 2);
    let a = unsafe { BooleanArray::from_trusted_len_iter_unchecked(iter) };
    assert_eq!(a.len(), 2);
}

#[test]
fn try_from_trusted_len_iter() {
    let iter = std::iter::repeat(true)
        .take(2)
        .map(Some)
        .map(PolarsResult::Ok);
    let a = BooleanArray::try_from_trusted_len_iter(iter.clone()).unwrap();
    assert_eq!(a.len(), 2);
    let a = unsafe { BooleanArray::try_from_trusted_len_iter_unchecked(iter).unwrap() };
    assert_eq!(a.len(), 2);
}

#[test]
fn from_trusted_len_values_iter() {
    let iter = std::iter::repeat(true).take(2);
    let a = BooleanArray::from_trusted_len_values_iter(iter.clone());
    assert_eq!(a.len(), 2);
    let a = unsafe { BooleanArray::from_trusted_len_values_iter_unchecked(iter) };
    assert_eq!(a.len(), 2);
}

#[test]
fn from_iter() {
    let iter = std::iter::repeat(true).take(2).map(Some);
    let a: BooleanArray = iter.collect();
    assert_eq!(a.len(), 2);
}

#[test]
fn into_iter() {
    let data = vec![Some(true), None, Some(false)];
    let rev = data.clone().into_iter().rev();

    let array: BooleanArray = data.clone().into_iter().collect();

    assert_eq!(array.clone().into_iter().collect::<Vec<_>>(), data);

    assert!(array.into_iter().rev().eq(rev))
}
