use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::arithmetics::basic::*;
use polars_arrow::compute::arithmetics::{
    ArrayCheckedMul, ArrayMul, ArrayOverflowingMul, ArraySaturatingMul,
};

#[test]
#[should_panic]
fn test_mul_mismatched_length() {
    let a = Int32Array::from_slice([5, 6]);
    let b = Int32Array::from_slice([5]);
    mul(&a, &b);
}

#[test]
fn test_mul() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = mul(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(36)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.mul(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_mul_panic() {
    let a = Int8Array::from(&[Some(-100i8)]);
    let b = Int8Array::from(&[Some(100i8)]);
    let _ = mul(&a, &b);
}

#[test]
fn test_mul_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = checked_mul(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(36)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(100i8), Some(100i8), Some(100i8)]);
    let b = Int8Array::from(&[Some(1i8), Some(100i8), Some(1i8)]);
    let result = checked_mul(&a, &b);
    let expected = Int8Array::from(&[Some(100i8), None, Some(100i8)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_mul(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_mul_saturating() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = saturating_mul(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(36)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(-100i8)]);
    let b = Int8Array::from(&[Some(100i8)]);
    let result = saturating_mul(&a, &b);
    let expected = Int8Array::from(&[Some(-128)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.saturating_mul(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_mul_overflowing() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let (result, overflow) = overflowing_mul(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(36)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, false, false, false]));

    let a = Int8Array::from(&[Some(1i8), Some(-100i8)]);
    let b = Int8Array::from(&[Some(1i8), Some(100i8)]);
    let (result, overflow) = overflowing_mul(&a, &b);
    let expected = Int8Array::from(&[Some(1i8), Some(-16i8)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));

    // Trait testing
    let (result, overflow) = a.overflowing_mul(&b);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));
}

#[test]
fn test_mul_scalar() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = mul_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.mul(&1i32);
    assert_eq!(result, expected);
}

#[test]
fn test_mul_scalar_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = checked_mul_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[None, Some(100), None, Some(100)]);
    let result = checked_mul_scalar(&a, &100i8);
    let expected = Int8Array::from(&[None, None, None, None]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_mul(&100i8);
    assert_eq!(result, expected);
}

#[test]
fn test_mul_scalar_saturating() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = saturating_mul_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(-100i8)]);
    let result = saturating_mul_scalar(&a, &100i8);
    let expected = Int8Array::from(&[Some(-128)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.saturating_mul(&100i8);
    assert_eq!(result, expected);
}

#[test]
fn test_mul_scalar_overflowing() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let (result, overflow) = overflowing_mul_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(6), None, Some(6)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, false, false, false]));

    let a = Int8Array::from(&[Some(1i8), Some(-100i8)]);
    let (result, overflow) = overflowing_mul_scalar(&a, &100i8);
    let expected = Int8Array::from(&[Some(100i8), Some(-16i8)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));

    // Trait testing
    let (result, overflow) = a.overflowing_mul(&100i8);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));
}
