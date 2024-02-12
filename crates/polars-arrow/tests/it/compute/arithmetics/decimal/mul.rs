#![allow(clippy::zero_prefixed_literal, clippy::inconsistent_digit_grouping)]

use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::decimal::{adaptive_mul, checked_mul, mul, saturating_mul};
use polars_arrow::compute::arithmetics::{ArrayCheckedMul, ArrayMul, ArraySaturatingMul};
use polars_arrow::datatypes::DataType;

#[test]
fn test_multiply_normal() {
    //   111.11 -->     11111
    //   222.22 -->     22222
    // --------       -------
    // 24690.86 <-- 246908642
    let a = PrimitiveArray::from([
        Some(111_11i128),
        Some(10_00i128),
        Some(20_00i128),
        None,
        Some(30_00i128),
        Some(123_45i128),
    ])
    .to(DataType::Decimal(7, 2));

    let b = PrimitiveArray::from([
        Some(222_22i128),
        Some(2_00i128),
        Some(3_00i128),
        None,
        Some(4_00i128),
        Some(543_21i128),
    ])
    .to(DataType::Decimal(7, 2));

    let result = mul(&a, &b);
    let expected = PrimitiveArray::from([
        Some(24690_86i128),
        Some(20_00i128),
        Some(60_00i128),
        None,
        Some(120_00i128),
        Some(67059_27i128),
    ])
    .to(DataType::Decimal(7, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.mul(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_multiply_decimal_wrong_precision() {
    let a = PrimitiveArray::from([None]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([None]).to(DataType::Decimal(6, 2));
    mul(&a, &b);
}

#[test]
#[should_panic(expected = "Overflow in multiplication presented for precision 5")]
fn test_multiply_panic() {
    let a = PrimitiveArray::from([Some(99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(100_00i128)]).to(DataType::Decimal(5, 2));
    let _ = mul(&a, &b);
}

#[test]
fn test_multiply_saturating() {
    let a = PrimitiveArray::from([
        Some(111_11i128),
        Some(10_00i128),
        Some(20_00i128),
        None,
        Some(30_00i128),
        Some(123_45i128),
    ])
    .to(DataType::Decimal(7, 2));

    let b = PrimitiveArray::from([
        Some(222_22i128),
        Some(2_00i128),
        Some(3_00i128),
        None,
        Some(4_00i128),
        Some(543_21i128),
    ])
    .to(DataType::Decimal(7, 2));

    let result = saturating_mul(&a, &b);
    let expected = PrimitiveArray::from([
        Some(24690_86i128),
        Some(20_00i128),
        Some(60_00i128),
        None,
        Some(120_00i128),
        Some(67059_27i128),
    ])
    .to(DataType::Decimal(7, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.saturating_mul(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_multiply_saturating_overflow() {
    let a = PrimitiveArray::from([
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
    ])
    .to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([
        Some(-00100i128),
        Some(01000i128),
        Some(10000i128),
        Some(-99999i128),
    ])
    .to(DataType::Decimal(5, 2));

    let result = saturating_mul(&a, &b);

    let expected = PrimitiveArray::from([
        Some(-99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(-99999i128),
    ])
    .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.saturating_mul(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_multiply_checked() {
    let a = PrimitiveArray::from([
        Some(111_11i128),
        Some(10_00i128),
        Some(20_00i128),
        None,
        Some(30_00i128),
        Some(123_45i128),
    ])
    .to(DataType::Decimal(7, 2));

    let b = PrimitiveArray::from([
        Some(222_22i128),
        Some(2_00i128),
        Some(3_00i128),
        None,
        Some(4_00i128),
        Some(543_21i128),
    ])
    .to(DataType::Decimal(7, 2));

    let result = checked_mul(&a, &b);
    let expected = PrimitiveArray::from([
        Some(24690_86i128),
        Some(20_00i128),
        Some(60_00i128),
        None,
        Some(120_00i128),
        Some(67059_27i128),
    ])
    .to(DataType::Decimal(7, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.checked_mul(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_multiply_checked_overflow() {
    let a = PrimitiveArray::from([Some(99999i128), Some(1_00i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(10000i128), Some(2_00i128)]).to(DataType::Decimal(5, 2));
    let result = checked_mul(&a, &b);
    let expected = PrimitiveArray::from([None, Some(2_00i128)]).to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);
}

#[test]
fn test_multiply_adaptive() {
    //  1000.00   -> 7, 2
    //    10.0000 -> 6, 4
    // -----------------
    // 10000.0000 -> 9, 4
    let a = PrimitiveArray::from([Some(1000_00i128)]).to(DataType::Decimal(7, 2));
    let b = PrimitiveArray::from([Some(10_0000i128)]).to(DataType::Decimal(6, 4));
    let result = adaptive_mul(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(10000_0000i128)]).to(DataType::Decimal(9, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 4));

    //   11111.0    -> 6, 1
    //      10.002  -> 5, 3
    // -----------------
    //  111132.222  -> 9, 3
    let a = PrimitiveArray::from([Some(11111_0i128)]).to(DataType::Decimal(6, 1));
    let b = PrimitiveArray::from([Some(10_002i128)]).to(DataType::Decimal(5, 3));
    let result = adaptive_mul(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(111132_222i128)]).to(DataType::Decimal(9, 3));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 3));

    //     12345.67   ->  7, 2
    //     12345.678  ->  8, 3
    // -----------------
    // 152415666.514  -> 11, 3
    let a = PrimitiveArray::from([Some(12345_67i128)]).to(DataType::Decimal(7, 2));
    let b = PrimitiveArray::from([Some(12345_678i128)]).to(DataType::Decimal(8, 3));
    let result = adaptive_mul(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(152415666_514i128)]).to(DataType::Decimal(12, 3));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(12, 3));
}
