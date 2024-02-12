use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::arithmetics::basic::*;
use polars_arrow::compute::arithmetics::{
    ArrayAdd, ArrayCheckedAdd, ArrayOverflowingAdd, ArraySaturatingAdd,
};

#[test]
#[should_panic]
fn test_add_mismatched_length() {
    let a = Int32Array::from_slice([5, 6]);
    let b = Int32Array::from_slice([5]);
    add(&a, &b);
}

#[test]
fn test_add() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = add(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(12)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.add(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_add_panic() {
    let a = Int8Array::from(&[Some(100i8)]);
    let b = Int8Array::from(&[Some(100i8)]);
    add(&a, &b);
}

#[test]
fn test_add_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = checked_add(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(12)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(100i8), Some(100i8), Some(100i8)]);
    let b = Int8Array::from(&[Some(0i8), Some(100i8), Some(0i8)]);
    let result = checked_add(&a, &b);
    let expected = Int8Array::from(&[Some(100i8), None, Some(100i8)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_add(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_add_saturating() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = saturating_add(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(12)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(100i8)]);
    let b = Int8Array::from(&[Some(100i8)]);
    let result = saturating_add(&a, &b);
    let expected = Int8Array::from(&[Some(127)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.saturating_add(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_add_overflowing() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let (result, overflow) = overflowing_add(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(12)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, false, false, false]));

    let a = Int8Array::from(&[Some(1i8), Some(100i8)]);
    let b = Int8Array::from(&[Some(1i8), Some(100i8)]);
    let (result, overflow) = overflowing_add(&a, &b);
    let expected = Int8Array::from(&[Some(2i8), Some(-56i8)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));

    // Trait testing
    let (result, overflow) = a.overflowing_add(&b);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));
}

#[test]
fn test_add_scalar() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = add_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(7), None, Some(7)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.add(&1i32);
    assert_eq!(result, expected);
}

#[test]
fn test_add_scalar_checked() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = checked_add_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(7), None, Some(7)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[None, Some(100), None, Some(100)]);
    let result = checked_add_scalar(&a, &100i8);
    let expected = Int8Array::from(&[None, None, None, None]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.checked_add(&100i8);
    assert_eq!(result, expected);
}

#[test]
fn test_add_scalar_saturating() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = saturating_add_scalar(&a, &1i32);
    let expected = Int32Array::from(&[None, Some(7), None, Some(7)]);
    assert_eq!(result, expected);

    let a = Int8Array::from(&[Some(100i8)]);
    let result = saturating_add_scalar(&a, &100i8);
    let expected = Int8Array::from(&[Some(127)]);
    assert_eq!(result, expected);

    // Trait testing
    let result = a.saturating_add(&100i8);
    assert_eq!(result, expected);
}

#[test]
fn test_add_scalar_overflowing() {
    let a = Int32Array::from(&vec![None, Some(6), None, Some(6)]);
    let (result, overflow) = overflowing_add_scalar(&a, &1i32);
    let expected = Int32Array::from(&vec![None, Some(7), None, Some(7)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, false, false, false]));

    let a = Int8Array::from(&vec![Some(1i8), Some(100i8)]);
    let (result, overflow) = overflowing_add_scalar(&a, &100i8);
    let expected = Int8Array::from(&vec![Some(101i8), Some(-56i8)]);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));

    // Trait testing
    let (result, overflow) = a.overflowing_add(&100i8);
    assert_eq!(result, expected);
    assert_eq!(overflow, Bitmap::from([false, true]));
}
