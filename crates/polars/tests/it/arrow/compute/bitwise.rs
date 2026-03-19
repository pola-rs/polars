use arrow::array::*;
use arrow::compute::bitwise::*;

#[test]
fn test_xor() {
    let a = Int32Array::from(&[Some(2), Some(4), Some(6), Some(7)]);
    let b = Int32Array::from(&[None, Some(6), Some(9), Some(7)]);
    let result = xor(&a, &b);
    let expected = Int32Array::from(&[None, Some(2), Some(15), Some(0)]);

    assert_eq!(result, expected);
}

#[test]
fn test_and() {
    let a = Int32Array::from(&[Some(1), Some(2), Some(15)]);
    let b = Int32Array::from(&[None, Some(2), Some(6)]);
    let result = and(&a, &b);
    let expected = Int32Array::from(&[None, Some(2), Some(6)]);

    assert_eq!(result, expected);
}

#[test]
fn test_or() {
    let a = Int32Array::from(&[Some(1), Some(2), Some(0)]);
    let b = Int32Array::from(&[None, Some(2), Some(0)]);
    let result = or(&a, &b);
    let expected = Int32Array::from(&[None, Some(2), Some(0)]);

    assert_eq!(result, expected);
}

#[test]
fn test_not() {
    let a = Int8Array::from(&[None, Some(1i8), Some(-100i8)]);
    let result = not(&a);
    let expected = Int8Array::from(&[None, Some(-2), Some(99)]);

    assert_eq!(result, expected);
}
