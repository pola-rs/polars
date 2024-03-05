use arrow::array::BooleanArray;
use arrow::compute::boolean_kleene::*;
use arrow::scalar::BooleanScalar;

#[test]
fn and_generic() {
    let lhs = BooleanArray::from(&[
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
    let rhs = BooleanArray::from(&[
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
    let c = and(&lhs, &rhs);

    let expected = BooleanArray::from(&[
        None,
        Some(false),
        None,
        Some(false),
        Some(false),
        Some(false),
        None,
        Some(false),
        Some(true),
    ]);

    assert_eq!(c, expected);
}

#[test]
fn or_generic() {
    let a = BooleanArray::from(&[
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
    let b = BooleanArray::from(&[
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

    let expected = BooleanArray::from(&[
        None,
        None,
        Some(true),
        None,
        Some(false),
        Some(true),
        Some(true),
        Some(true),
        Some(true),
    ]);

    assert_eq!(c, expected);
}

#[test]
fn or_right_nulls() {
    let a = BooleanArray::from_slice([false, false, false, true, true, true]);

    let b = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let c = or(&a, &b);

    let expected = BooleanArray::from(&[
        Some(true),
        Some(false),
        None,
        Some(true),
        Some(true),
        Some(true),
    ]);

    assert_eq!(c, expected);
}

#[test]
fn or_left_nulls() {
    let a = BooleanArray::from(vec![
        Some(true),
        Some(false),
        None,
        Some(true),
        Some(false),
        None,
    ]);

    let b = BooleanArray::from_slice([false, false, false, true, true, true]);

    let c = or(&a, &b);

    let expected = BooleanArray::from(vec![
        Some(true),
        Some(false),
        None,
        Some(true),
        Some(true),
        Some(true),
    ]);

    assert_eq!(c, expected);
}

#[test]
fn array_and_true() {
    let array = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let scalar = BooleanScalar::new(Some(true));
    let result = and_scalar(&array, &scalar);

    // Should be same as argument array if scalar is true.
    assert_eq!(result, array);
}

#[test]
fn array_and_false() {
    let array = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let scalar = BooleanScalar::new(Some(false));
    let result = and_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[
        Some(false),
        Some(false),
        Some(false),
        Some(false),
        Some(false),
        Some(false),
    ]);

    assert_eq!(result, expected);
}

#[test]
fn array_and_none() {
    let array = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let scalar = BooleanScalar::new(None);
    let result = and_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[None, Some(false), None, None, Some(false), None]);

    assert_eq!(result, expected);
}

#[test]
fn array_or_true() {
    let array = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let scalar = BooleanScalar::new(Some(true));
    let result = or_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[
        Some(true),
        Some(true),
        Some(true),
        Some(true),
        Some(true),
        Some(true),
    ]);

    assert_eq!(result, expected);
}

#[test]
fn array_or_false() {
    let array = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let scalar = BooleanScalar::new(Some(false));
    let result = or_scalar(&array, &scalar);

    // Should be same as argument array if scalar is false.
    assert_eq!(result, array);
}

#[test]
fn array_or_none() {
    let array = BooleanArray::from(&[Some(true), Some(false), None, Some(true), Some(false), None]);

    let scalar = BooleanScalar::new(None);
    let result = or_scalar(&array, &scalar);

    let expected = BooleanArray::from(&[Some(true), None, None, Some(true), None, None]);

    assert_eq!(result, expected);
}

#[test]
fn array_empty() {
    let array = BooleanArray::from(&[]);
    assert_eq!(any(&array), Some(false));
    assert_eq!(all(&array), Some(true));
}
