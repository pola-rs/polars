use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::basic::*;
use polars_arrow::compute::arithmetics::{ArrayCheckedDiv, ArrayDiv};

#[test]
#[should_panic]
fn test_div_mismatched_length() {
    let a = Int32Array::from_slice([5, 6]);
    let b = Int32Array::from_slice([5]);
    div(&a, &b);
}

#[test]
fn test_div() {
    let a = Int32Array::from(&[Some(5), Some(6)]);
    let b = Int32Array::from(&[Some(5), Some(6)]);
    let result = div(&a, &b);
    let expected = Int32Array::from(&[Some(1), Some(1)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.div(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_div_panic() {
    let a = Int8Array::from(&[Some(10i8)]);
    let b = Int8Array::from(&[Some(0i8)]);
    let _ = div(&a, &b);
}

#[test]
fn test_div_checked() {
    let a = Int32Array::from(&[Some(5), None, Some(3), Some(6)]);
    let b = Int32Array::from(&[Some(5), Some(3), None, Some(6)]);
    let result = checked_div(&a, &b);
    let expected = Int32Array::from(&[Some(1), None, None, Some(1)]);
    assert_eq!(result, expected);

    let a = Int32Array::from(&[Some(5), None, Some(3), Some(6)]);
    let b = Int32Array::from(&[Some(5), Some(0), Some(0), Some(6)]);
    let result = checked_div(&a, &b);
    let expected = Int32Array::from(&[Some(1), None, None, Some(1)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_div(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_div_scalar() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = div_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.div(&1i32);
    assert_eq!(result, expected);

    // check the strength reduced branches
    let a = UInt64Array::from(&[None, Some(6), None, Some(6)]);
    let result = div_scalar(&a, &1u64);
    let expected = UInt64Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    let a = UInt32Array::from(&[None, Some(6), None, Some(6)]);
    let result = div_scalar(&a, &1u32);
    let expected = UInt32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    let a = UInt16Array::from(&[None, Some(6), None, Some(6)]);
    let result = div_scalar(&a, &1u16);
    let expected = UInt16Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    let a = UInt8Array::from(&[None, Some(6), None, Some(6)]);
    let result = div_scalar(&a, &1u8);
    let expected = UInt8Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);
}

#[test]
fn test_div_scalar_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = checked_div_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = checked_div_scalar(&a, &0);
    let expected = Int32Array::from(&[None, None, None, None]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_div(&0);
    assert_eq!(result, expected);
}
