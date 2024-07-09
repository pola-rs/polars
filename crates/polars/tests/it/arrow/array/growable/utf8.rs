use arrow::array::growable::{Growable, GrowableUtf8};
use arrow::array::Utf8Array;

/// tests extending from a variable-sized (strings and binary) array w/ offset with nulls
#[test]
fn validity() {
    let array = Utf8Array::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);

    let mut a = GrowableUtf8::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }

    let result: Utf8Array<i32> = a.into();

    let expected = Utf8Array::<i32>::from([Some("bc"), None]);
    assert_eq!(result, expected);
}

/// tests extending from a variable-sized (strings and binary) array
/// with an offset and nulls
#[test]
fn offsets() {
    let array = Utf8Array::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);
    let array = array.sliced(1, 3);

    let mut a = GrowableUtf8::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 0, 3);
    }
    assert_eq!(a.len(), 3);

    let result: Utf8Array<i32> = a.into();

    let expected = Utf8Array::<i32>::from([Some("bc"), None, Some("defh")]);
    assert_eq!(result, expected);
}

#[test]
fn offsets2() {
    let array = Utf8Array::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);
    let array = array.sliced(1, 3);

    let mut a = GrowableUtf8::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 0, 3);
    }
    assert_eq!(a.len(), 3);

    let result: Utf8Array<i32> = a.into();

    let expected = Utf8Array::<i32>::from([Some("bc"), None, Some("defh")]);
    assert_eq!(result, expected);
}

#[test]
fn multiple_with_validity() {
    let array1 = Utf8Array::<i32>::from_slice(["hello", "world"]);
    let array2 = Utf8Array::<i32>::from([Some("1"), None]);

    let mut a = GrowableUtf8::new(vec![&array1, &array2], false, 5);

    unsafe {
        a.extend(0, 0, 2);
    }
    unsafe {
        a.extend(1, 0, 2);
    }
    assert_eq!(a.len(), 4);

    let result: Utf8Array<i32> = a.into();

    let expected = Utf8Array::<i32>::from([Some("hello"), Some("world"), Some("1"), None]);
    assert_eq!(result, expected);
}

#[test]
fn null_offset_validity() {
    let array = Utf8Array::<i32>::from([Some("a"), Some("bc"), None, Some("defh")]);
    let array = array.sliced(1, 3);

    let mut a = GrowableUtf8::new(vec![&array], true, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    a.extend_validity(1);
    assert_eq!(a.len(), 3);

    let result: Utf8Array<i32> = a.into();

    let expected = Utf8Array::<i32>::from([None, Some("defh"), None]);
    assert_eq!(result, expected);
}
