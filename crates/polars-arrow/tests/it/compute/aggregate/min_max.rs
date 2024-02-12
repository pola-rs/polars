use polars_arrow::array::*;
use polars_arrow::compute::aggregate::{
    max_binary, max_boolean, max_primitive, max_string, min_binary, min_boolean, min_primitive,
    min_string,
};
use polars_arrow::datatypes::DataType;

#[test]
fn test_primitive_array_min_max() {
    let a = Int32Array::from_slice([5, 6, 7, 8, 9]);
    assert_eq!(5, min_primitive(&a).unwrap());
    assert_eq!(9, max_primitive(&a).unwrap());
}

#[test]
fn test_primitive_array_min_max_with_nulls() {
    let a = Int32Array::from(&[Some(5), None, None, Some(8), Some(9)]);
    assert_eq!(5, min_primitive(&a).unwrap());
    assert_eq!(9, max_primitive(&a).unwrap());
}

#[test]
fn test_primitive_min_max_1() {
    let a = Int32Array::from(&[None, None, Some(5), Some(2)]);
    assert_eq!(Some(2), min_primitive(&a));
    assert_eq!(Some(5), max_primitive(&a));
}

#[test]
fn decimal() {
    let a = Int128Array::from(&[None, None, Some(5), Some(2)]);
    assert_eq!(Some(2), min_primitive(&a));
    assert_eq!(Some(5), max_primitive(&a));
}

#[test]
fn min_max_f32() {
    let a = Float32Array::from(&[None, None, Some(5.0), Some(2.0)]);
    assert_eq!(Some(2.0), min_primitive(&a));
    assert_eq!(Some(5.0), max_primitive(&a));
}

#[test]
fn min_max_f64() {
    let a = Float64Array::from(&[None, None, Some(5.0), Some(2.0)]);
    assert_eq!(Some(2.0), min_primitive(&a));
    assert_eq!(Some(5.0), max_primitive(&a));
}

#[test]
fn min_max_f64_large() {
    // in simd, f64 has 8 lanes, thus > 8 covers the branch with lanes
    let a = Float64Array::from(&[
        None,
        None,
        Some(8.0),
        Some(2.0),
        None,
        None,
        Some(5.0),
        Some(2.0),
        None,
        None,
        Some(8.0),
        Some(2.0),
        None,
        None,
        Some(5.0),
        Some(2.0),
        None,
        None,
        Some(8.0),
        Some(2.0),
        None,
        None,
        Some(5.0),
        Some(2.0),
    ]);
    assert_eq!(Some(2.0), min_primitive(&a));
    assert_eq!(Some(8.0), max_primitive(&a));
}

#[test]
fn min_max_f64_nan_only() {
    let a = Float64Array::from(&[None, Some(f64::NAN)]);
    assert!(min_primitive(&a).unwrap().is_nan());
    assert!(max_primitive(&a).unwrap().is_nan());
}

#[test]
fn min_max_f64_nan() {
    let a = Float64Array::from(&[None, Some(1.0), Some(f64::NAN)]);
    assert_eq!(Some(1.0), min_primitive(&a));
    assert_eq!(Some(1.0), max_primitive(&a));
}

#[test]
fn min_max_f64_edge_cases() {
    let a: Float64Array = (0..100).map(|_| Some(f64::NEG_INFINITY)).collect();
    assert_eq!(Some(f64::NEG_INFINITY), min_primitive(&a));
    assert_eq!(Some(f64::NEG_INFINITY), max_primitive(&a));

    let a: Float64Array = (0..100).map(|_| Some(f64::MIN)).collect();
    assert_eq!(Some(f64::MIN), min_primitive(&a));
    assert_eq!(Some(f64::MIN), max_primitive(&a));

    let a: Float64Array = (0..100).map(|_| Some(f64::MAX)).collect();
    assert_eq!(Some(f64::MAX), min_primitive(&a));
    assert_eq!(Some(f64::MAX), max_primitive(&a));

    let a: Float64Array = (0..100).map(|_| Some(f64::INFINITY)).collect();
    assert_eq!(Some(f64::INFINITY), min_primitive(&a));
    assert_eq!(Some(f64::INFINITY), max_primitive(&a));
}

#[test]
fn test_string_min_max_with_nulls() {
    let a = Utf8Array::<i32>::from([Some("b"), None, None, Some("a"), Some("c")]);
    assert_eq!(Some("a"), min_string(&a));
    assert_eq!(Some("c"), max_string(&a));
}

#[test]
fn test_string_min_max_all_nulls() {
    let a = Utf8Array::<i32>::from([None::<&str>, None]);
    assert_eq!(None, min_string(&a));
    assert_eq!(None, max_string(&a));
}

#[test]
fn test_string_min_max_no_null() {
    let a = Utf8Array::<i32>::from([Some("abc"), Some("abd"), Some("bac"), Some("bbb")]);
    assert_eq!(Some("abc"), min_string(&a));
    assert_eq!(Some("bbb"), max_string(&a));
}

#[test]
fn test_string_min_max_1() {
    let a = Utf8Array::<i32>::from([None, None, Some("b"), Some("a")]);
    assert_eq!(Some("a"), min_string(&a));
    assert_eq!(Some("b"), max_string(&a));
}

#[test]
fn test_boolean_min_max_empty() {
    let a = BooleanArray::new_empty(DataType::Boolean);
    assert_eq!(None, min_boolean(&a));
    assert_eq!(None, max_boolean(&a));
}

#[test]
fn test_boolean_min_max_all_null() {
    let a = BooleanArray::from(&[None, None]);
    assert_eq!(None, min_boolean(&a));
    assert_eq!(None, max_boolean(&a));
}

#[test]
fn test_boolean_min_max_no_null() {
    let a = BooleanArray::from(&[Some(true), Some(false), Some(true)]);
    assert_eq!(Some(false), min_boolean(&a));
    assert_eq!(Some(true), max_boolean(&a));
}

#[test]
fn test_boolean_min_max() {
    let a = BooleanArray::from(&[Some(true), Some(true), None, Some(false), None]);
    assert_eq!(Some(false), min_boolean(&a));
    assert_eq!(Some(true), max_boolean(&a));

    let a = BooleanArray::from(&[None, Some(true), None, Some(false), None]);
    assert_eq!(Some(false), min_boolean(&a));
    assert_eq!(Some(true), max_boolean(&a));

    let a = BooleanArray::from(&[Some(false), Some(true), None, Some(false), None]);
    assert_eq!(Some(false), min_boolean(&a));
    assert_eq!(Some(true), max_boolean(&a));
}

#[test]
fn test_boolean_min_max_smaller() {
    let a = BooleanArray::from(&[Some(false)]);
    assert_eq!(Some(false), min_boolean(&a));
    assert_eq!(Some(false), max_boolean(&a));

    let a = BooleanArray::from(&[None, Some(false)]);
    assert_eq!(Some(false), min_boolean(&a));
    assert_eq!(Some(false), max_boolean(&a));

    let a = BooleanArray::from(&[None, Some(true)]);
    assert_eq!(Some(true), min_boolean(&a));
    assert_eq!(Some(true), max_boolean(&a));

    let a = BooleanArray::from(&[Some(true)]);
    assert_eq!(Some(true), min_boolean(&a));
    assert_eq!(Some(true), max_boolean(&a));
}

#[test]
fn test_binary_min_max_with_nulls() {
    let a = BinaryArray::<i32>::from([Some(b"b"), None, None, Some(b"a"), Some(b"c")]);
    assert_eq!(Some("a".as_bytes()), min_binary(&a));
    assert_eq!(Some("c".as_bytes()), max_binary(&a));
}

#[test]
fn test_binary_min_max_no_null() {
    let a = BinaryArray::<i32>::from([
        Some("abc".as_bytes()),
        Some(b"acd"),
        Some(b"aabd"),
        Some(b""),
    ]);
    assert_eq!(Some("".as_bytes()), min_binary(&a));
    assert_eq!(Some("acd".as_bytes()), max_binary(&a));
}

#[test]
fn test_binary_min_max_all_nulls() {
    let a = BinaryArray::<i32>::from([None::<&[u8]>, None]);
    assert_eq!(None, min_binary(&a));
    assert_eq!(None, max_binary(&a));
}

#[test]
fn test_binary_min_max_1() {
    let a = BinaryArray::<i32>::from([None, None, Some(b"b"), Some(b"a")]);
    assert_eq!(Some("a".as_bytes()), min_binary(&a));
    assert_eq!(Some("b".as_bytes()), max_binary(&a));
}

#[test]
fn test_max_not_lexi() {
    let values = [0, 10, 0, 0, 0, 0, 0, 0, 1, 0];
    let arr = Int64Array::from_slice(values);

    let maximum = 10;
    let out = max_primitive(&arr).unwrap();
    assert_eq!(out, maximum);
}
