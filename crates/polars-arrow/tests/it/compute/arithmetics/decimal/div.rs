#![allow(clippy::zero_prefixed_literal, clippy::inconsistent_digit_grouping)]

use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::decimal::{
    adaptive_div, checked_div, div, div_scalar, saturating_div,
};
use polars_arrow::compute::arithmetics::{ArrayCheckedDiv, ArrayDiv};
use polars_arrow::datatypes::DataType;
use polars_arrow::scalar::PrimitiveScalar;

#[test]
fn test_divide_normal() {
    //   222.222 -->  222222000
    //   123.456 -->     123456
    // --------       ---------
    //     1.800 <--       1800
    let a = PrimitiveArray::from([
        Some(222_222i128),
        Some(10_000i128),
        Some(20_000i128),
        None,
        Some(30_000i128),
        Some(123_456i128),
    ])
    .to(DataType::Decimal(7, 3));

    let b = PrimitiveArray::from([
        Some(123_456i128),
        Some(2_000i128),
        Some(3_000i128),
        Some(4_000i128),
        Some(4_000i128),
        Some(654_321i128),
    ])
    .to(DataType::Decimal(7, 3));

    let result = div(&a, &b);
    let expected = PrimitiveArray::from([
        Some(1_800i128),
        Some(5_000i128),
        Some(6_666i128),
        None,
        Some(7_500i128),
        Some(0_188i128),
    ])
    .to(DataType::Decimal(7, 3));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.div(&b);
    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_divide_decimal_wrong_precision() {
    let a = PrimitiveArray::from([None]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([None]).to(DataType::Decimal(6, 2));
    div(&a, &b);
}

#[test]
#[should_panic(expected = "Overflow in multiplication presented for precision 5")]
fn test_divide_panic() {
    let a = PrimitiveArray::from([Some(99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([Some(000_01i128)]).to(DataType::Decimal(5, 2));
    div(&a, &b);
}

#[test]
fn test_div_scalar() {
    //   222.222 -->  222222000
    //   123.456 -->     123456
    // --------       ---------
    //     1.800 <--       1800
    let a = PrimitiveArray::from([Some(222_222i128), None]).to(DataType::Decimal(7, 3));
    let b = PrimitiveScalar::from(Some(123_456i128)).to(DataType::Decimal(7, 3));
    let result = div_scalar(&a, &b);

    let expected = PrimitiveArray::from([Some(1_800i128), None]).to(DataType::Decimal(7, 3));
    assert_eq!(result, expected);
}

#[test]
#[should_panic(expected = "Overflow in multiplication presented for precision 5")]
fn test_divide_scalar_panic() {
    let a = PrimitiveArray::from([Some(99999i128)]).to(DataType::Decimal(5, 2));
    let b = PrimitiveScalar::from(Some(000_01i128)).to(DataType::Decimal(5, 2));
    div_scalar(&a, &b);
}

#[test]
fn test_divide_saturating() {
    let a = PrimitiveArray::from([
        Some(222_222i128),
        Some(10_000i128),
        Some(20_000i128),
        None,
        Some(30_000i128),
        Some(123_456i128),
    ])
    .to(DataType::Decimal(7, 3));

    let b = PrimitiveArray::from([
        Some(123_456i128),
        Some(2_000i128),
        Some(3_000i128),
        Some(4_000i128),
        Some(4_000i128),
        Some(654_321i128),
    ])
    .to(DataType::Decimal(7, 3));

    let result = saturating_div(&a, &b);
    let expected = PrimitiveArray::from([
        Some(1_800i128),
        Some(5_000i128),
        Some(6_666i128),
        None,
        Some(7_500i128),
        Some(0_188i128),
    ])
    .to(DataType::Decimal(7, 3));

    assert_eq!(result, expected);
}

#[test]
fn test_divide_saturating_overflow() {
    let a = PrimitiveArray::from([
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(99999i128),
    ])
    .to(DataType::Decimal(5, 2));
    let b = PrimitiveArray::from([
        Some(-00001i128),
        Some(00001i128),
        Some(00010i128),
        Some(-00020i128),
        Some(00000i128),
    ])
    .to(DataType::Decimal(5, 2));

    let result = saturating_div(&a, &b);

    let expected = PrimitiveArray::from([
        Some(-99999i128),
        Some(99999i128),
        Some(99999i128),
        Some(-99999i128),
        Some(00000i128),
    ])
    .to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);
}

#[test]
fn test_divide_checked() {
    let a = PrimitiveArray::from([
        Some(222_222i128),
        Some(10_000i128),
        Some(20_000i128),
        None,
        Some(30_000i128),
        Some(123_456i128),
    ])
    .to(DataType::Decimal(7, 3));

    let b = PrimitiveArray::from([
        Some(123_456i128),
        Some(2_000i128),
        Some(3_000i128),
        Some(4_000i128),
        Some(4_000i128),
        Some(654_321i128),
    ])
    .to(DataType::Decimal(7, 3));

    let result = div(&a, &b);
    let expected = PrimitiveArray::from([
        Some(1_800i128),
        Some(5_000i128),
        Some(6_666i128),
        None,
        Some(7_500i128),
        Some(0_188i128),
    ])
    .to(DataType::Decimal(7, 3));

    assert_eq!(result, expected);
}

#[test]
fn test_divide_checked_overflow() {
    let a = PrimitiveArray::from([Some(1_00i128), Some(4_00i128), Some(6_00i128)])
        .to(DataType::Decimal(5, 2));
    let b =
        PrimitiveArray::from([Some(000_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));

    let result = checked_div(&a, &b);
    let expected = PrimitiveArray::from([None, None, Some(3_00i128)]).to(DataType::Decimal(5, 2));

    assert_eq!(result, expected);

    // Testing trait
    let result = a.checked_div(&b);
    assert_eq!(result, expected);
}

#[test]
fn test_divide_adaptive() {
    //  1000.00   -> 7, 2
    //    10.0000 -> 6, 4
    // -----------------
    //   100.0000 -> 9, 4
    let a = PrimitiveArray::from([Some(1000_00i128)]).to(DataType::Decimal(7, 2));
    let b = PrimitiveArray::from([Some(10_0000i128)]).to(DataType::Decimal(6, 4));
    let result = adaptive_div(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(100_0000i128)]).to(DataType::Decimal(9, 4));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(9, 4));

    //   11111.0    -> 6, 1
    //      10.002  -> 5, 3
    // -----------------
    //    1110.877  -> 8, 3
    let a = PrimitiveArray::from([Some(11111_0i128)]).to(DataType::Decimal(6, 1));
    let b = PrimitiveArray::from([Some(10_002i128)]).to(DataType::Decimal(5, 3));
    let result = adaptive_div(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(1110_877i128)]).to(DataType::Decimal(8, 3));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(8, 3));

    //     12345.67   ->  7, 2
    //     12345.678  ->  8, 3
    // -----------------
    //         0.999  ->  8, 3
    let a = PrimitiveArray::from([Some(12345_67i128)]).to(DataType::Decimal(7, 2));
    let b = PrimitiveArray::from([Some(12345_678i128)]).to(DataType::Decimal(8, 3));
    let result = adaptive_div(&a, &b).unwrap();

    let expected = PrimitiveArray::from([Some(0_999i128)]).to(DataType::Decimal(8, 3));

    assert_eq!(result, expected);
    assert_eq!(result.data_type(), &DataType::Decimal(8, 3));
}
