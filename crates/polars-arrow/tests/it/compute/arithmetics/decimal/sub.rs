#![allow(clippy::zero_prefixed_literal, clippy::inconsistent_digit_grouping)]

use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::decimal::{adaptive_sub, checked_sub, saturating_sub, sub};
use polars_arrow::compute::arithmetics::{ArrayCheckedSub, ArraySaturatingSub, ArraySub};
use polars_arrow::datatypes::DataType;

#[test]
fn test_subtract_normal() {
    let a = PrimitiveArray::from([Some(11111i128), Some(22200i128), None, Some(40000i128)])
        .to(DataType::Decimal(5, 2));

    let b = PrimitiveArray::from([Some(22222i128), Some(11100i128), None, Some(11100i128)])
        .to(DataType::Decimal(5, 2));

    let result = sub(&a, &b);
    let expected = PrimitiveArray::from([Some(-11111i128), Some(11100i128), None, Some(28900i128)])
        .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.sub(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_subtract_decimal_wrong_precision() {
    let a = PrimitiveArray::from([None]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([None]).to(DataType::Decimal(6, 2));
    sub(&a, &b);
}

#[test]
#[should_panic(expected = "Overflow in subtract presented for precision 5")]
fn test_subtract_panic() {
    let a = PrimitiveArray::from([Some(-99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(1i128)]).to(DataType::Decimal(5, 2));
    let _ = sub(&a, &b);
}

#[test]
fn test_subtract_saturating() {
    let a = PrimitiveArray::from([Some(11111i128), Some(22200i128), None, Some(40000i128)])
        .to(DataType::Decimal(5, 2));

    let b = PrimitiveArray::from([Some(22222i128), Some(11100i128), None, Some(11100i128)])
        .to(DataType::Decimal(5, 2));

    let result = saturating_sub(&a, &b);
    let expected = PrimitiveArray::from([Some(-11111i128), Some(11100i128), None, Some(28900i128)])
        .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.saturating_sub(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_saturating_overflow() {
    let a = PrimitiveArray::from([
        Some(-99999i128),
        Some(-99999i128),
        Some(-99999i128),
        Some(99999i128),
    ])
    .to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([
        Some(00001i128),
        Some(00100i128),
        Some(10000i128),
        Some(-99999i128),
    ])
    .to(DataType::Decimal(5, 2));

    let result = saturating_sub(&a, &b);

    let expected = PrimitiveArray::from([
        Some(-99999i128),
        Some(-99999i128),
        Some(-99999i128),
        Some(99999i128),
    ])
    .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.saturating_sub(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_checked() {
    let a = PrimitiveArray::from([Some(11111i128), Some(22200i128), None, Some(40000i128)])
        .to(DataType::Decimal(5, 2));

    let b = PrimitiveArray::from([Some(22222i128), Some(11100i128), None, Some(11100i128)])
        .to(DataType::Decimal(5, 2));

    let result = checked_sub(&a, &b);
    let expected = PrimitiveArray::from([Some(-11111i128), Some(11100i128), None, Some(28900i128)])
        .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.checked_sub(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_checked_overflow() {
    let a = PrimitiveArray::from([Some(4i128), Some(-99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(2i128), Some(1i128)]).to(DataType::Decimal(5, 2));
    let result = checked_sub(&a, &b);
    let expected = PrimitiveArray::from([Some(2i128), None]).to(DataType::Decimal(5, 2));
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_adaptive() {
    //     11.1111 -> 6, 4
    //  11111.11   -> 7, 2
    // ------------------
    // -11099.9989 -> 9, 4
    let a = PrimitiveArray::from([Some(11_1111i128)]).to(DataType::Decimal(6, 4));
    let b = PrimitiveArray::from([Some(11111_11i128)]).to(DataType::Decimal(7, 2));
    let result = adaptive_sub(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(-11099_9989i128)]).to(DataType::Decimal(9, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 4));

    // 11111.0    -> 6, 1
    //     0.1111 -> 5, 4
    // -----------------
    // 11110.8889 -> 9, 4
    let a = PrimitiveArray::from([Some(11111_0i128)]).to(DataType::Decimal(6, 1));
    let b = PrimitiveArray::from([Some(1111i128)]).to(DataType::Decimal(5, 4));
    let result = adaptive_sub(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(11110_8889i128)]).to(DataType::Decimal(9, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 4));

    //  11111.11   -> 7, 2
    //  11111.111  -> 8, 3
    // -----------------
    // -00000.001  -> 8, 3
    let a = PrimitiveArray::from([Some(11111_11i128)]).to(DataType::Decimal(7, 2));
    let b = PrimitiveArray::from([Some(11111_111i128)]).to(DataType::Decimal(8, 3));
    let result = adaptive_sub(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(-00000_001i128)]).to(DataType::Decimal(8, 3));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(8, 3));

    //  99.9999 -> 6, 4
    // -00.0001 -> 6, 4
    // -----------------
    // 100.0000 -> 7, 4
    let a = PrimitiveArray::from([Some(99_9999i128)]).to(DataType::Decimal(6, 4));
    let b = PrimitiveArray::from([Some(-00_0001i128)]).to(DataType::Decimal(6, 4));
    let result = adaptive_sub(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(100_0000i128)]).to(DataType::Decimal(7, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(7, 4));
}
