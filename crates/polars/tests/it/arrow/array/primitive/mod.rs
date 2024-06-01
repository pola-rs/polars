use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::*;
use arrow::types::months_days_ns;

mod fmt;
mod mutable;
mod to_mutable;

fn array() -> Int32Array {
    vec![Some(1), None, Some(10)].into_iter().collect()
}

#[test]
fn basics() {
    let array = array();

    assert_eq!(array.value(0), 1);
    assert_eq!(array.value(1), 0);
    assert_eq!(array.value(2), 10);
    assert_eq!(array.values().as_slice(), &[1, 0, 10]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00000101], 3))
    );
    assert!(array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(array.is_valid(2));

    let array2 = Int32Array::new(
        ArrowDataType::Int32,
        array.values().clone(),
        array.validity().cloned(),
    );
    assert_eq!(array, array2);

    let array = array.sliced(1, 2);
    assert_eq!(array.value(0), 0);
    assert_eq!(array.value(1), 10);
    assert_eq!(array.values().as_slice(), &[0, 10]);

    unsafe {
        assert_eq!(array.value_unchecked(0), 0);
        assert_eq!(array.value_unchecked(1), 10);
    }
}

#[test]
fn split_at() {
    let (lhs, rhs) = array().split_at(1);

    assert!(lhs.is_valid(0));
    assert!(!rhs.is_valid(0));
    assert!(rhs.is_valid(1));

    assert_eq!(lhs.value(0), 1);
    assert_eq!(rhs.value(0), 0);
    assert_eq!(rhs.value(1), 10);
}

#[test]
fn empty() {
    let array = Int32Array::new_empty(ArrowDataType::Int32);
    assert_eq!(array.values().len(), 0);
    assert_eq!(array.validity(), None);
}

#[test]
fn from() {
    let data = vec![Some(1), None, Some(10)];

    let array = PrimitiveArray::from(data.clone());
    assert_eq!(array.len(), 3);

    let array = PrimitiveArray::from_iter(data.clone());
    assert_eq!(array.len(), 3);

    let array = PrimitiveArray::from_trusted_len_iter(data.into_iter());
    assert_eq!(array.len(), 3);

    let data = vec![1i32, 2, 3];

    let array = PrimitiveArray::from_values(data.clone());
    assert_eq!(array.len(), 3);

    let array = PrimitiveArray::from_trusted_len_values_iter(data.into_iter());
    assert_eq!(array.len(), 3);
}

#[test]
fn months_days_ns_from_slice() {
    let data = &[
        months_days_ns::new(1, 1, 2),
        months_days_ns::new(1, 1, 3),
        months_days_ns::new(2, 3, 3),
    ];

    let array = MonthsDaysNsArray::from_slice(data);

    let a = array.values().as_slice();
    assert_eq!(a, data.as_ref());
}

#[test]
fn wrong_data_type() {
    let values = Buffer::from(b"abbb".to_vec());
    assert!(PrimitiveArray::try_new(ArrowDataType::Utf8, values, None).is_err());
}

#[test]
fn wrong_len() {
    let values = Buffer::from(b"abbb".to_vec());
    let validity = Some([true, false].into());
    assert!(PrimitiveArray::try_new(ArrowDataType::Utf8, values, validity).is_err());
}

#[test]
fn into_mut_1() {
    let values = Buffer::<i32>::from(vec![0, 1]);
    let a = values.clone(); // cloned values
    assert_eq!(a, values);
    let array = PrimitiveArray::new(ArrowDataType::Int32, values, None);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_2() {
    let values = Buffer::<i32>::from(vec![0, 1]);
    let validity = Some([true, false].into());
    let a = validity.clone(); // cloned values
    assert_eq!(a, validity);
    let array = PrimitiveArray::new(ArrowDataType::Int32, values, validity);
    assert!(array.into_mut().is_left());
}

#[test]
fn into_mut_3() {
    let values = Buffer::<i32>::from(vec![0, 1]);
    let validity = Some([true, false].into());
    let array = PrimitiveArray::new(ArrowDataType::Int32, values, validity);
    assert!(array.into_mut().is_right());
}

#[test]
fn into_iter() {
    let data = vec![Some(1), None, Some(10)];
    let rev = data.clone().into_iter().rev();

    let array: Int32Array = data.clone().into_iter().collect();

    assert_eq!(array.clone().into_iter().collect::<Vec<_>>(), data);

    assert!(array.into_iter().rev().eq(rev))
}
