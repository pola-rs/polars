use polars_arrow::array::*;
use polars_arrow::compute::contains::contains;

// disable wrapping inside literal vectors used for test data and assertions
#[rustfmt::skip::macros(vec)]
// Expected behaviour:
// contains([1, 2, null], 1) = true
// contains([1, 2, null], 3) = false
// contains([1, 2, null], null) = null
// contains(null, 1) = null
#[test]
fn test_contains() {
    let data = vec![
        Some(vec![Some(1), Some(2), None]),
        Some(vec![Some(1), Some(2), None]),
        Some(vec![Some(1), Some(2), None]),
        None,
    ];
    let values = Int32Array::from(&[Some(1), Some(3), None, Some(1)]);
    let expected = BooleanArray::from(vec![
        Some(true),
        Some(false),
        None,
        None
    ]);

    let mut a = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    a.try_extend(data).unwrap();
    let a: ListArray<i32> = a.into();

    let result = contains(&a, &values).unwrap();

    assert_eq!(result, expected);
}

// disable wrapping inside literal vectors used for test data and assertions
#[rustfmt::skip::macros(vec)]
#[test]
fn test_contains_binary() {
    let data = vec![
        Some(vec![Some(b"a"), Some(b"b"), None]),
        Some(vec![Some(b"a"), Some(b"b"), None]),
        Some(vec![Some(b"a"), Some(b"b"), None]),
        None,
    ];
    let values = BinaryArray::<i32>::from([Some(b"a"), Some(b"c"), None, Some(b"a")]);
    let expected = BooleanArray::from(vec![
        Some(true),
        Some(false),
        None,
        None
    ]);

    let mut a = MutableListArray::<i32, MutableBinaryArray<i32>>::new();
    a.try_extend(data).unwrap();
    let a: ListArray<i32> = a.into();

    let result = contains(&a, &values).unwrap();

    assert_eq!(result, expected);
}
