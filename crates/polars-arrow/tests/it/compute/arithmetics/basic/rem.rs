use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::basic::*;
use polars_arrow::compute::arithmetics::{ArrayCheckedRem, ArrayRem};

#[test]
#[should_panic]
fn test_rem_mismatched_length() {
    let a = Int32Array::from_slice([5, 6]);
    let b = Int32Array::from_slice([5]);
    rem(&a, &b);
}

#[test]
fn test_rem() {
    let a = Int32Array::from(&[Some(5), Some(6)]);
    let b = Int32Array::from(&[Some(4), Some(4)]);
    let result = rem(&a, &b);
    let expected = Int32Array::from(&[Some(1), Some(2)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.rem(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_rem_panic() {
    let a = Int8Array::from(&[Some(10i8)]);
    let b = Int8Array::from(&[Some(0i8)]);
    let _ = rem(&a, &b);
}

#[test]
fn test_rem_checked() {
    let a = Int32Array::from(&[Some(5), None, Some(3), Some(6)]);
    let b = Int32Array::from(&[Some(5), Some(3), None, Some(5)]);
    let result = checked_rem(&a, &b);
    let expected = Int32Array::from(&[Some(0), None, None, Some(1)]);
    assert_eq!(result, expected);

    let a = Int32Array::from(&[Some(5), None, Some(3), Some(6)]);
    let b = Int32Array::from(&[Some(5), Some(0), Some(0), Some(5)]);
    let result = checked_rem(&a, &b);
    let expected = Int32Array::from(&[Some(0), None, None, Some(1)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_rem(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_rem_scalar() {
    let a = Int32Array::from(&[None, Some(6), None, Some(5)]);
    let result = rem_scalar(&a, &2i32);
    let expected = Int32Array::from(&[None, Some(0), None, Some(1)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.rem(&2i32);
    assert_eq!(result, expected);

    // check the strength reduced branches
    let a = UInt64Array::from(&[None, Some(6), None, Some(5)]);
    let result = rem_scalar(&a, &2u64);
    let expected = UInt64Array::from(&[None, Some(0), None, Some(1)]);
    assert_eq!(result, expected);

    let a = UInt32Array::from(&[None, Some(6), None, Some(5)]);
    let result = rem_scalar(&a, &2u32);
    let expected = UInt32Array::from(&[None, Some(0), None, Some(1)]);
    assert_eq!(result, expected);

    let a = UInt16Array::from(&[None, Some(6), None, Some(5)]);
    let result = rem_scalar(&a, &2u16);
    let expected = UInt16Array::from(&[None, Some(0), None, Some(1)]);
    assert_eq!(result, expected);

    let a = UInt8Array::from(&[None, Some(6), None, Some(5)]);
    let result = rem_scalar(&a, &2u8);
    let expected = UInt8Array::from(&[None, Some(0), None, Some(1)]);
    assert_eq!(result, expected);
}

#[test]
fn test_rem_scalar_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(7)]);
    let result = checked_rem_scalar(&a, &2i32);
    let expected = Int32Array::from(&[None, Some(0), None, Some(1)]);
    assert_eq!(result, expected);

    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = checked_rem_scalar(&a, &0);
    let expected = Int32Array::from(&[None, None, None, None]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_rem(&0);
    assert_eq!(result, expected);
}
