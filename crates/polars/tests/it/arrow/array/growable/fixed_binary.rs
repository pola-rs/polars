use arrow::array::growable::{Growable, GrowableFixedSizeBinary};
use arrow::array::FixedSizeBinaryArray;

/// tests extending from a variable-sized (strings and binary) array w/ offset with nulls
#[test]
fn basic() {
    let array =
        FixedSizeBinaryArray::from_iter(vec![Some(b"ab"), Some(b"bc"), None, Some(b"de")], 2);

    let mut a = GrowableFixedSizeBinary::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    assert_eq!(a.len(), 2);

    let result: FixedSizeBinaryArray = a.into();

    let expected = FixedSizeBinaryArray::from_iter(vec![Some("bc"), None], 2);
    assert_eq!(result, expected);
}

/// tests extending from a variable-sized (strings and binary) array
/// with an offset and nulls
#[test]
fn offsets() {
    let array =
        FixedSizeBinaryArray::from_iter(vec![Some(b"ab"), Some(b"bc"), None, Some(b"fh")], 2);
    let array = array.sliced(1, 3);

    let mut a = GrowableFixedSizeBinary::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 0, 3);
    }
    assert_eq!(a.len(), 3);

    let result: FixedSizeBinaryArray = a.into();

    let expected = FixedSizeBinaryArray::from_iter(vec![Some(b"bc"), None, Some(b"fh")], 2);
    assert_eq!(result, expected);
}

#[test]
fn multiple_with_validity() {
    let array1 = FixedSizeBinaryArray::from_iter(vec![Some("hello"), Some("world")], 5);
    let array2 = FixedSizeBinaryArray::from_iter(vec![Some("12345"), None], 5);

    let mut a = GrowableFixedSizeBinary::new(vec![&array1, &array2], false, 5);

    unsafe {
        a.extend(0, 0, 2);
    }
    unsafe {
        a.extend(1, 0, 2);
    }
    assert_eq!(a.len(), 4);

    let result: FixedSizeBinaryArray = a.into();

    let expected =
        FixedSizeBinaryArray::from_iter(vec![Some("hello"), Some("world"), Some("12345"), None], 5);
    assert_eq!(result, expected);
}

#[test]
fn null_offset_validity() {
    let array = FixedSizeBinaryArray::from_iter(vec![Some("aa"), Some("bc"), None, Some("fh")], 2);
    let array = array.sliced(1, 3);

    let mut a = GrowableFixedSizeBinary::new(vec![&array], true, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    a.extend_validity(1);
    assert_eq!(a.len(), 3);

    let result: FixedSizeBinaryArray = a.into();

    let expected = FixedSizeBinaryArray::from_iter(vec![None, Some("fh"), None], 2);
    assert_eq!(result, expected);
}

#[test]
fn sized_offsets() {
    let array =
        FixedSizeBinaryArray::from_iter(vec![Some(&[0, 0]), Some(&[0, 1]), Some(&[0, 2])], 2);
    let array = array.sliced(1, 2);
    // = [[0, 1], [0, 2]] due to the offset = 1

    let mut a = GrowableFixedSizeBinary::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 1);
    }
    unsafe {
        a.extend(0, 0, 1);
    }
    assert_eq!(a.len(), 2);

    let result: FixedSizeBinaryArray = a.into();

    let expected = FixedSizeBinaryArray::from_iter(vec![Some(&[0, 2]), Some(&[0, 1])], 2);
    assert_eq!(result, expected);
}

/// to, as_box, as_arc
#[test]
fn as_box() {
    let array =
        FixedSizeBinaryArray::from_iter(vec![Some(b"ab"), Some(b"bc"), None, Some(b"de")], 2);
    let mut a = GrowableFixedSizeBinary::new(vec![&array], false, 0);
    unsafe {
        a.extend(0, 1, 2);
    }

    let result = a.as_box();
    let result = result
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();

    let expected = FixedSizeBinaryArray::from_iter(vec![Some("bc"), None], 2);
    assert_eq!(&expected, result);
}

/// as_arc
#[test]
fn as_arc() {
    let array =
        FixedSizeBinaryArray::from_iter(vec![Some(b"ab"), Some(b"bc"), None, Some(b"de")], 2);
    let mut a = GrowableFixedSizeBinary::new(vec![&array], false, 0);
    unsafe {
        a.extend(0, 1, 2);
    }

    let result = a.as_arc();
    let result = result
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();

    let expected = FixedSizeBinaryArray::from_iter(vec![Some("bc"), None], 2);
    assert_eq!(&expected, result);
}
