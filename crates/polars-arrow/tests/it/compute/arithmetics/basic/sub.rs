use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::arithmetics::basic::*;
use polars_arrow::compute::arithmetics::{
    ArrayCheckedSub, ArrayOverflowingSub, ArraySaturatingSub, ArraySub,
};

#[test]
#[should_panic]
fn test_sub_mismatched_length() {
    let a = Int32Array::from_slice([5, 6]);
    let b = Int32Array::from_slice([5]);
    sub(&a, &b);
}

#[test]
fn test_sub() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = sub(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(0)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.sub(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_sub_panic() {
    let a = Int8Array::from(&[Some(-100i8)]);
    let b = Int8Array::from(&[Some(100i8)]);
    let _ = sub(&a, &b);
}

#[test]
fn test_sub_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = checked_sub(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(0)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(100i8), Some(-100i8), Some(100i8)]);
    let b = Int8Array::from(&[Some(1i8), Some(100i8), Some(0i8)]);
    let result = checked_sub(&a, &b);
    let expected = Int8Array::from(&[Some(99i8), None, Some(100i8)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_sub(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_sub_saturating() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = saturating_sub(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(0)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(-100i8)]);
    let b = Int8Array::from(&[Some(100i8)]);
    let result = saturating_sub(&a, &b);
    let expected = Int8Array::from(&[Some(-128)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.saturating_sub(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_sub_overflowing() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let (result, overflow) = overflowing_sub(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(0)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, false, false, false]));

    let a = Int8Array::from(&[Some(1i8), Some(-100i8)]);
    let b = Int8Array::from(&[Some(1i8), Some(100i8)]);
    let (result, overflow) = overflowing_sub(&a, &b);
    let expected = Int8Array::from(&[Some(0i8), Some(56i8)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));

    // Trait testing
    let (result, overflow) = a.overflowing_sub(&b);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));
}

#[test]
fn test_sub_scalar() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = sub_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(5), None, Some(5)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.sub(&1i32);
    assert_eq!(result, expected);
}

#[test]
fn test_sub_scalar_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = checked_sub_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(5), None, Some(5)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[None, Some(-100), None, Some(-100)]);
    let result = checked_sub_scalar(&a, &100i8);
    let expected = Int8Array::from(&[None, None, None, None]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_sub(&100i8);
    assert_eq!(result, expected);
}

#[test]
fn test_sub_scalar_saturating() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = saturating_sub_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(5), None, Some(5)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(-100i8)]);
    let result = saturating_sub_scalar(&a, &100i8);
    let expected = Int8Array::from(&[Some(-128)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.saturating_sub(&100i8);
    assert_eq!(result, expected);
}

#[test]
fn test_sub_scalar_overflowing() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let (result, overflow) = overflowing_sub_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(5), None, Some(5)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, false, false, false]));

    let a = Int8Array::from(&[Some(1i8), Some(-100i8)]);
    let (result, overflow) = overflowing_sub_scalar(&a, &100i8);
    let expected = Int8Array::from(&[Some(-99i8), Some(56i8)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));

    // Trait testing
    let (result, overflow) = a.overflowing_sub(&100i8);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));
}
