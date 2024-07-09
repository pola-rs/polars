use std::iter::FromIterator;

use arrow::array::*;
use arrow::compute::boolean::*;
use arrow::scalar::BooleanScalar;

#[test]
fn array_and() {
    let a = BooleanArray::from_slice(vec![false, false, true, true]);
    let b = BooleanArray::from_slice(vec![false, true, false, true]);
    let c = and(&a, &b);

    let expected = BooleanArray::from_slice(vec![false, false, false, true]);

    assert_eq!(c, expected);
}

#[test]
fn array_or() {
    let a = BooleanArray::from_slice(vec![false, false, true, true]);
    let b = BooleanArray::from_slice(vec![false, true, false, true]);
    let c = or(&a, &b);

    let expected = BooleanArray::from_slice(vec![false, true, true, true]);

    assert_eq!(c, expected);
}

#[test]
fn array_or_validity() {
    let a = BooleanArray::from(vec![
        None,
        None,
        None,
        Some(false),
        Some(false),
        Some(false),
        Some(true),
        Some(true),
        Some(true),
    ]);
    let b = BooleanArray::from(vec![
        None,
        Some(false),
        Some(true),
        None,
        Some(false),
        Some(true),
        None,
        Some(false),
        Some(true),
    ]);
    let c = or(&a, &b);

    let expected = BooleanArray::from(vec![
        None,
        None,
        None,
        None,
        Some(false),
        Some(true),
        None,
        Some(true),
        Some(true),
    ]);

    assert_eq!(c, expected);
}

#[test]
fn array_not() {
    let a = BooleanArray::from_slice(vec![false, true]);
    let c = not(&a);

    let expected = BooleanArray::from_slice(vec![true, false]);

    assert_eq!(c, expected);
}

#[test]
fn array_and_validity() {
    let a = BooleanArray::from(vec![
        None,
        None,
        None,
        Some(false),
        Some(false),
        Some(false),
        Some(true),
        Some(true),
        Some(true),
    ]);
    let b = BooleanArray::from(vec![
        None,
        Some(false),
        Some(true),
        None,
        Some(false),
        Some(true),
        None,
        Some(false),
        Some(true),
    ]);
    let c = and(&a, &b);

    let expected = BooleanArray::from(vec![
        None,
        None,
        None,
        None,
        Some(false),
        Some(false),
        None,
        Some(false),
        Some(true),
    ]);

    assert_eq!(c, expected);
}

#[test]
fn array_and_sliced_same_offset() {
    let a = BooleanArray::from_slice(vec![
        false, false, false, false, false, false, false, false, false, false, true, true,
    ]);
    let b = BooleanArray::from_slice(vec![
        false, false, false, false, false, false, false, false, false, true, false, true,
    ]);

    let a = a.sliced(8, 4);
    let b = b.sliced(8, 4);
    let c = and(&a, &b);

    let expected = BooleanArray::from_slice(vec![false, false, false, true]);

    assert_eq!(expected, c);
}

#[test]
fn array_and_sliced_same_offset_mod8() {
    let a = BooleanArray::from_slice(vec![
        false, false, true, true, false, false, false, false, false, false, false, false,
    ]);
    let b = BooleanArray::from_slice(vec![
        false, false, false, false, false, false, false, false, false, true, false, true,
    ]);

    let a = a.sliced(0, 4);
    let b = b.sliced(8, 4);

    let c = and(&a, &b);

    let expected = BooleanArray::from_slice(vec![false, false, false, true]);

    assert_eq!(expected, c);
}

#[test]
fn array_and_sliced_offset1() {
    let a = BooleanArray::from_slice(vec![
        false, false, false, false, false, false, false, false, false, false, true, true,
    ]);
    let b = BooleanArray::from_slice(vec![false, true, false, true]);

    let a = a.sliced(8, 4);

    let c = and(&a, &b);

    let expected = BooleanArray::from_slice(vec![false, false, false, true]);

    assert_eq!(expected, c);
}

#[test]
fn array_and_sliced_offset2() {
    let a = BooleanArray::from_slice(vec![false, false, true, true]);
    let b = BooleanArray::from_slice(vec![
        false, false, false, false, false, false, false, false, false, true, false, true,
    ]);

    let b = b.sliced(8, 4);

    let c = and(&a, &b);

    let expected = BooleanArray::from_slice(vec![false, false, false, true]);

    assert_eq!(expected, c);
}

#[test]
fn array_and_validity_offset() {
    let a = BooleanArray::from(vec![None, Some(false), Some(true), None, Some(true)]);
    let a = a.sliced(1, 4);
    let a = a.as_any().downcast_ref::<BooleanArray>().unwrap();

    let b = BooleanArray::from(vec![
        None,
        None,
        Some(true),
        Some(false),
        Some(true),
        Some(true),
    ]);

    let b = b.sliced(2, 4);
    let b = b.as_any().downcast_ref::<BooleanArray>().unwrap();

    let c = and(a, b);

    let expected = BooleanArray::from(vec![Some(false), Some(false), None, Some(true)]);

    assert_eq!(expected, c);
}

#[test]
fn test_nonnull_array_is_null() {
    let a = Int32Array::from_slice([1, 2, 3, 4]);

    let res = is_null(&a);

    let expected = BooleanArray::from_slice(vec![false, false, false, false]);

    assert_eq!(expected, res);
}

#[test]
fn test_nonnull_array_with_offset_is_null() {
    let a = Int32Array::from_slice(vec![1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]);
    let a = a.sliced(8, 4);

    let res = is_null(&a);

    let expected = BooleanArray::from_slice(vec![false, false, false, false]);

    assert_eq!(expected, res);
}

#[test]
fn test_nonnull_array_is_not_null() {
    let a = Int32Array::from_slice([1, 2, 3, 4]);

    let res = is_not_null(&a);

    let expected = BooleanArray::from_slice(vec![true, true, true, true]);

    assert_eq!(expected, res);
}

#[test]
fn test_nonnull_array_with_offset_is_not_null() {
    let a = Int32Array::from_slice([1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]);
    let a = a.sliced(8, 4);

    let res = is_not_null(&a);

    let expected = BooleanArray::from_slice([true, true, true, true]);

    assert_eq!(expected, res);
}

#[test]
fn test_nullable_array_is_null() {
    let a = Int32Array::from(vec![Some(1), None, Some(3), None]);

    let res = is_null(&a);

    let expected = BooleanArray::from_slice(vec![false, true, false, true]);

    assert_eq!(expected, res);
}

#[test]
fn test_nullable_array_with_offset_is_null() {
    let a = Int32Array::from(vec![
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        // offset 8, previous None values are skipped by the slice
        Some(1),
        None,
        Some(2),
        None,
        Some(3),
        Some(4),
        None,
        None,
    ]);
    let a = a.sliced(8, 4);

    let res = is_null(&a);

    let expected = BooleanArray::from_slice(vec![false, true, false, true]);

    assert_eq!(expected, res);
}

#[test]
fn test_nullable_array_is_not_null() {
    let a = Int32Array::from(vec![Some(1), None, Some(3), None]);

    let res = is_not_null(&a);

    let expected = BooleanArray::from_slice(vec![true, false, true, false]);

    assert_eq!(expected, res);
}

#[test]
fn test_nullable_array_with_offset_is_not_null() {
    let a = Int32Array::from(vec![
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        // offset 8, previous None values are skipped by the slice
        Some(1),
        None,
        Some(2),
        None,
        Some(3),
        Some(4),
        None,
        None,
    ]);
    let a = a.sliced(8, 4);

    let res = is_not_null(&a);

    let expected = BooleanArray::from_slice(vec![true, false, true, false]);

    assert_eq!(expected, res);
}

#[test]
fn array_and_scalar() {
    let array = BooleanArray::from_slice([false, false, true, true]);

    let scalar = BooleanScalar::new(Some(true));
    let real = and_scalar(&array, &scalar);

    let expected = BooleanArray::from_slice([false, false, true, true]);
    assert_eq!(real, expected);

    let scalar = BooleanScalar::new(Some(false));
    let real = and_scalar(&array, &scalar);

    let expected = BooleanArray::from_slice([false, false, false, false]);

    assert_eq!(real, expected);
}

#[test]
fn array_and_scalar_validity() {
    let array = BooleanArray::from(&[None, Some(false), Some(true)]);

    let scalar = BooleanScalar::new(Some(true));
    let real = and_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None, Some(false), Some(true)]);
    assert_eq!(real, expected);

    let scalar = BooleanScalar::new(None);
    let real = and_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None; 3]);
    assert_eq!(real, expected);

    let array = BooleanArray::from_slice([true, false, true]);
    let real = and_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None; 3]);
    assert_eq!(real, expected);
}

#[test]
fn array_or_scalar() {
    let array = BooleanArray::from_slice([false, false, true, true]);

    let scalar = BooleanScalar::new(Some(true));
    let real = or_scalar(&array, &scalar);

    let expected = BooleanArray::from_slice([true, true, true, true]);
    assert_eq!(real, expected);

    let scalar = BooleanScalar::new(Some(false));
    let real = or_scalar(&array, &scalar);

    let expected = BooleanArray::from_slice([false, false, true, true]);
    assert_eq!(real, expected);
}

#[test]
fn array_or_scalar_validity() {
    let array = BooleanArray::from(&[None, Some(false), Some(true)]);

    let scalar = BooleanScalar::new(Some(true));
    let real = or_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None, Some(true), Some(true)]);
    assert_eq!(real, expected);

    let scalar = BooleanScalar::new(None);
    let real = or_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None; 3]);
    assert_eq!(real, expected);

    let array = BooleanArray::from_slice([true, false, true]);
    let real = and_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None; 3]);
    assert_eq!(real, expected);
}

#[test]
fn test_any_all() {
    let array = BooleanArray::from(&[None, Some(false), Some(true)]);
    assert!(any(&array));
    assert!(!all(&array));
    let array = BooleanArray::from(&[None, Some(false), Some(false)]);
    assert!(!any(&array));
    assert!(!all(&array));
    let array = BooleanArray::from(&[None, Some(true), Some(true)]);
    assert!(any(&array));
    assert!(all(&array));
    let array = BooleanArray::from_iter(std::iter::repeat(false).take(10).map(Some));
    assert!(!any(&array));
    assert!(!all(&array));
    let array = BooleanArray::from_iter(std::iter::repeat(true).take(10).map(Some));
    assert!(any(&array));
    assert!(all(&array));
    let array = BooleanArray::from_iter([true, false, true, true].map(Some));
    assert!(any(&array));
    assert!(!all(&array));
    let array = BooleanArray::from(&[Some(true)]);
    assert!(any(&array));
    assert!(all(&array));
    let array = BooleanArray::from(&[Some(false)]);
    assert!(!any(&array));
    assert!(!all(&array));
    let array = BooleanArray::from(&[]);
    assert!(!any(&array));
    assert!(all(&array));
}
