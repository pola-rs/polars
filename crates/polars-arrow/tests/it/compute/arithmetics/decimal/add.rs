#![allow(clippy::zero_prefixed_literal, clippy::inconsistent_digit_grouping)]

use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::decimal::{adaptive_add, add, checked_add, saturating_add};
use polars_arrow::compute::arithmetics::{ArrayAdd, ArrayCheckedAdd, ArraySaturatingAdd};
use polars_arrow::datatypes::DataType;

#[test]
fn test_add_normal() {
    let a = PrimitiveArray::from([Some(11111i128), Some(11100i128), None, Some(22200i128)])
        .to(DataType::Decimal(5, 2));

    let b = PrimitiveArray::from([Some(22222i128), Some(22200i128), None, Some(11100i128)])
        .to(DataType::Decimal(5, 2));

    let result = add(&a, &b);
    let expected = PrimitiveArray::from([Some(33333i128), Some(33300i128), None, Some(33300i128)])
        .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.add(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_add_decimal_wrong_precision() {
    let a = PrimitiveArray::from([None]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([None]).to(DataType::Decimal(6, 2));
    add(&a, &b);
}

#[test]
#[should_panic(expected = "Overflow in addition presented for precision 5")]
fn test_add_panic() {
    let a = PrimitiveArray::from([Some(99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(1i128)]).to(DataType::Decimal(5, 2));
    let _ = add(&a, &b);
}

#[test]
fn test_add_saturating() {
    let a = PrimitiveArray::from([Some(11111i128), Some(11100i128), None, Some(22200i128)])
        .to(DataType::Decimal(5, 2));

    let b = PrimitiveArray::from([Some(22222i128), Some(22200i128), None, Some(11100i128)])
        .to(DataType::Decimal(5, 2));

    let result = saturating_add(&a, &b);
    let expected = PrimitiveArray::from([Some(33333i128), Some(33300i128), None, Some(33300i128)])
        .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.saturating_add(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_add_saturating_overflow() {
    let a = PrimitiveArray::from([
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(-99999i128),
    ])
    .to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([
        Some(00001i128),
        Some(00100i128),
        Some(10000i128),
        Some(-99999i128),
    ])
    .to(DataType::Decimal(5, 2));

    let result = saturating_add(&a, &b);

    let expected = PrimitiveArray::from([
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(-99999i128),
    ])
    .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.saturating_add(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_add_checked() {
    let a = PrimitiveArray::from([Some(11111i128), Some(11100i128), None, Some(22200i128)])
        .to(DataType::Decimal(5, 2));

    let b = PrimitiveArray::from([Some(22222i128), Some(22200i128), None, Some(11100i128)])
        .to(DataType::Decimal(5, 2));

    let result = checked_add(&a, &b);
    let expected = PrimitiveArray::from([Some(33333i128), Some(33300i128), None, Some(33300i128)])
        .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.checked_add(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_add_checked_overflow() {
    let a = PrimitiveArray::from([Some(1i128), Some(99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(1i128), Some(1i128)]).to(DataType::Decimal(5, 2));
    let result = checked_add(&a, &b);
    let expected = PrimitiveArray::from([Some(2i128), None]).to(DataType::Decimal(5, 2));
    assert_eq!(result, expected);

    // Testing trait
    let result = a.checked_add(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_add_adaptive() {
    //    11.1111 -> 6, 4
    // 11111.11   -> 7, 2
    // -----------------
    // 11122.2211 -> 9, 4
    let a = PrimitiveArray::from([Some(11_1111i128)]).to(DataType::Decimal(6, 4));
    let b = PrimitiveArray::from([Some(11111_11i128)]).to(DataType::Decimal(7, 2));
    let result = adaptive_add(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(11122_2211i128)]).to(DataType::Decimal(9, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 4));

    //     0.1111 -> 5, 4
    // 11111.0    -> 6, 1
    // -----------------
    // 11111.1111 -> 9, 4
    let a = PrimitiveArray::from([Some(1111i128)]).to(DataType::Decimal(5, 4));
    let b = PrimitiveArray::from([Some(11111_0i128)]).to(DataType::Decimal(6, 1));
    let result = adaptive_add(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(11111_1111i128)]).to(DataType::Decimal(9, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 4));

    // 11111.11   -> 7, 2
    // 11111.111  -> 8, 3
    // -----------------
    // 22222.221  -> 8, 3
    let a = PrimitiveArray::from([Some(11111_11i128)]).to(DataType::Decimal(7, 2));
    let b = PrimitiveArray::from([Some(11111_111i128)]).to(DataType::Decimal(8, 3));
    let result = adaptive_add(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(22222_221i128)]).to(DataType::Decimal(8, 3));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(8, 3));

    //  99.9999 -> 6, 4
    //  00.0001 -> 6, 4
    // -----------------
    // 100.0000 -> 7, 4
    let a = PrimitiveArray::from([Some(99_9999i128)]).to(DataType::Decimal(6, 4));
    let b = PrimitiveArray::from([Some(00_0001i128)]).to(DataType::Decimal(6, 4));
    let result = adaptive_add(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(100_0000i128)]).to(DataType::Decimal(7, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(7, 4));
}
