use arrow::array::growable::{Growable, GrowableBinary};
use arrow::array::BinaryArray;

#[test]
fn no_offsets() {
    let array = BinaryArray::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);

    let mut a = GrowableBinary::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    assert_eq!(a.len(), 2);

    let result: BinaryArray<i32> = a.into();

    let expected = BinaryArray::<i32>::from([Some("bc"), None]);
    assert_eq!(result, expected);
}

/// tests extending from a variable-sized (strings and binary) array
/// with an offset and nulls
#[test]
fn with_offsets() {
    let array = BinaryArray::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);
    let array = array.sliced(1, 3);

    let mut a = GrowableBinary::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 0, 3);
    }
    assert_eq!(a.len(), 3);

    let result: BinaryArray<i32> = a.into();

    let expected = BinaryArray::<i32>::from([Some("bc"), None, Some("defh")]);
    assert_eq!(result, expected);
}

#[test]
fn test_string_offsets() {
    let array = BinaryArray::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);
    let array = array.sliced(1, 3);

    let mut a = GrowableBinary::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 0, 3);
    }
    assert_eq!(a.len(), 3);

    let result: BinaryArray<i32> = a.into();

    let expected = BinaryArray::<i32>::from([Some("bc"), None, Some("defh")]);
    assert_eq!(result, expected);
}

#[test]
fn test_multiple_with_validity() {
    let array1 = BinaryArray::<i32>::from_slice([b"hello", b"world"]);
    let array2 = BinaryArray::<i32>::from([Some("1"), None]);

    let mut a = GrowableBinary::new(vec![&array1, &array2], false, 5);

    unsafe {
        a.extend(0, 0, 2);
    }
    unsafe {
        a.extend(1, 0, 2);
    }
    assert_eq!(a.len(), 4);

    let result: BinaryArray<i32> = a.into();

    let expected = BinaryArray::<i32>::from([Some("hello"), Some("world"), Some("1"), None]);
    assert_eq!(result, expected);
}

#[test]
fn test_string_null_offset_validity() {
    let array = BinaryArray::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);
    let array = array.sliced(1, 3);

    let mut a = GrowableBinary::new(vec![&array], true, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    a.extend_validity(1);
    assert_eq!(a.len(), 3);

    let result: BinaryArray<i32> = a.into();

    let expected = BinaryArray::<i32>::from([None, Some("defh"), None]);
    assert_eq!(result, expected);
}
