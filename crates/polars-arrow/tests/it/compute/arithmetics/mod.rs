mod basic;
mod decimal;
mod time;

use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::*;
use polars_arrow::datatypes::DataType::*;
use polars_arrow::datatypes::{IntervalUnit, TimeUnit};
use polars_arrow::scalar::PrimitiveScalar;

#[test]
fn test_add() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
    let result = add(&a, &b);
    let expected = Int32Array::from(&[None, None, None, Some(12)]);
    assert_eq!(expected, result.as_ref());
}

#[test]
fn test_add_scalar() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let b: PrimitiveScalar<i32> = Some(1i32).into();
    let result = add_scalar(&a, &b);
    let expected = Int32Array::from(&[None, Some(7), None, Some(7)]);
    assert_eq!(expected, result.as_ref());
}

#[test]
fn consistency() {
    let datatypes = vec![
        Null,
        Boolean,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float32,
        Float64,
        Timestamp(TimeUnit::Second, None),
        Timestamp(TimeUnit::Millisecond, None),
        Timestamp(TimeUnit::Microsecond, None),
        Timestamp(TimeUnit::Nanosecond, None),
        Time64(TimeUnit::Microsecond),
        Time64(TimeUnit::Nanosecond),
        Date32,
        Time32(TimeUnit::Second),
        Time32(TimeUnit::Millisecond),
        Date64,
        Utf8,
        LargeUtf8,
        Binary,
        LargeBinary,
        Duration(TimeUnit::Second),
        Duration(TimeUnit::Millisecond),
        Duration(TimeUnit::Microsecond),
        Duration(TimeUnit::Nanosecond),
        Interval(IntervalUnit::MonthDayNano),
    ];

    let cases = datatypes.clone().into_iter().zip(datatypes.into_iter());

    cases.for_each(|(lhs, rhs)| {
        let lhs_a = new_empty_array(lhs.clone());
        let rhs_a = new_empty_array(rhs.clone());
        if can_add(&lhs, &rhs) {
            add(lhs_a.as_ref(), rhs_a.as_ref());
        }
        if can_sub(&lhs, &rhs) {
            sub(lhs_a.as_ref(), rhs_a.as_ref());
        }
        if can_mul(&lhs, &rhs) {
            mul(lhs_a.as_ref(), rhs_a.as_ref());
        }
        if can_div(&lhs, &rhs) {
            div(lhs_a.as_ref(), rhs_a.as_ref());
        }
        if can_rem(&lhs, &rhs) {
            rem(lhs_a.as_ref(), rhs_a.as_ref());
        }
    });
}

#[test]
fn test_neg() {
    let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
    let result = neg(&a);
    let expected = Int32Array::from(&[None, Some(-6), None, Some(-6)]);
    assert_eq!(expected, result.as_ref());
}

#[test]
fn test_neg_dict() {
    let a = DictionaryArray::try_from_keys(
        UInt8Array::from_slice([0, 0, 1]),
        Int8Array::from_slice([1, 2]).boxed(),
    )
    .unwrap();
    let result = neg(&a);
    let expected = DictionaryArray::try_from_keys(
        UInt8Array::from_slice([0, 0, 1]),
        Int8Array::from_slice([-1, -2]).boxed(),
    )
    .unwrap();
    assert_eq!(expected, result.as_ref());
}
